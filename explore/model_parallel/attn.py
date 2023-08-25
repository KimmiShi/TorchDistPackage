import torch
from torch import nn as nn
from torch.nn.parameter import Parameter
from einops import rearrange

try:
    from flash_attn.flash_attention import FlashAttention
    from flash_attn.modules.mlp import FusedMLP
    from flash_attn.ops.rms_norm import DropoutAddRMSNorm
except:
    print("flash attention is not installed.")

from tp_utils import get_tp_group, set_tp_group, mylinear, RowParallelLinear, ColParallelLinear, _ReduceFromModelParallelRegion

def _split_heads(tensor, num_heads, attn_head_size):
    """
    Splits hidden_size dim into attn_head_size and num_heads
    """
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 use_flash_attn=False, causal=False,
                 fuse_dal=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = mylinear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = mylinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            self.causal = causal
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)



    def _naive_attn(self, x):
        B, N, DIM = x.shape
        # print("qkv out:",  self.qkv(x), flush=True)

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)  # make torchscript happy (cannot use tensor as tuple)
        q= _split_heads(q, self.num_heads, self.head_dim)
        k= _split_heads(k, self.num_heads, self.head_dim)
        v= _split_heads(v, self.num_heads, self.head_dim)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        # attn = attn - attn.max(-1)[0].unsqueeze(-1)  # in case of overflow for fp16
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, DIM)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):
        if self.use_subln:
            qk = self.qk(x)
            v = self.v(x)
            qkv = torch.cat([qk, v], dim=-1)
        else:
            qkv = self.qkv(x)

        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads)

        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            if self.qk_normalization_head_merged:
                if self.fuse_dal:
                    q = self.q_norm(q.flatten(-2, -1))[0].view(q.shape)
                    k = self.k_norm(k.flatten(-2, -1))[0].view(k.shape)
                else:
                    q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
                    k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            else:
                q = self.q_norm(q)
                k = self.k_norm(k)
                if self.fuse_dal:
                    q = q[0]
                    k = k[0]
            qkv = torch.stack([q, k, v], dim=2)

        if self.k_centering:
            # B, N, H, D
            qkv[:, :, 1] = qkv[:, :, 1] - qkv[:, :, 1].mean(1, keepdim=True)
        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=self.causal
        )
        outs = self.proj(rearrange(context, "b s h d -> b s (h d)"))
        outs = self.proj_drop(outs)
        return outs

    def forward(self, x):
        x = self._naive_attn(x) if not self.use_flash_attn else self._flash_attn(x)
        return x


class TpAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 tp_group = None,
                 use_flash_attn=False, causal=False,
                 fuse_dal=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        set_tp_group(tp_group)
        self.tp_size = torch.distributed.get_world_size(get_tp_group())
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.head_num_per_partition = self.num_heads//self.tp_size

        self.qkv = ColParallelLinear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = RowParallelLinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            self.causal = causal
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)

        self.fuse_dal = fuse_dal


    def _naive_attn(self, x):
        B, N, DIM = x.shape     # DIM=self.head_dim*self.num_heads
        # print("tp qkv out:",  self.qkv(x), flush=True)
        qkv_out = self.qkv(x)       # B,N,3*DIM
        # qkv_out_reshaped = qkv_out.reshape(B, N, self.head_num_per_partition, 3*self.head_dim)

        q, k, v = qkv_out.chunk(3, dim=-1) # B.N.DIM
        q= _split_heads(q, self.head_num_per_partition, self.head_dim)
        k= _split_heads(k, self.head_num_per_partition, self.head_dim)
        v= _split_heads(v, self.head_num_per_partition, self.head_dim)

        # B,N,3*DIM -> B,N,num_heads, 3*head_dim)
        # # split into: B,N,num_heads,head_dim
        # q,k,v = qkv_out_reshaped.split(qkv_out.shape[-1]//self.tp_size, -1)  # B,N,DIM
        # # reshape to B,N, num_heads, head_dim


        # q = q.permute(0,2,3,1)
        # k = k.permute(0,2,3,1)
        # v = v.permute(0,2,3,1)

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads//self.tp_size, DIM // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)


        attn = ((q * self.scale) @ k.transpose(-2, -1))
        # attn = attn - attn.max(-1)[0].unsqueeze(-1)  # in case of overflow for fp16
        # import pdb;pdb.set_trace()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, DIM//self.tp_size)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x):
        x = self._naive_attn(x) #if not self.use_flash_attn else self._flash_attn(x)
        return x