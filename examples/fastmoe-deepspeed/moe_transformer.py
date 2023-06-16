import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from einops import rearrange

from timm.models.layers.helpers import to_2tuple
from timm.models.layers.drop import DropPath

from .norm_layers import LayerNorm

try:
    from flash_attn.flash_attention import FlashAttention
    from flash_attn.modules.mlp import FusedMLP
except ImportError:
    UserWarning("flash_attn is not installed, please make sure use_flash_attn and use_fused_mlp are both False.")

try:
    from flash_attn.ops.layer_norm import DropoutAddLayerNorm
    from flash_attn.ops.rms_norm import DropoutAddRMSNorm
except ImportError:
    UserWarning("flash_attn layer_norm is not installed.")


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 use_flash_attn=False, causal=False, use_subln=False, norm_layer=LayerNorm,
                 qk_normalization=False, k_centering=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.use_subln = use_subln

        if use_subln:
            self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
            self.sub_norm = norm_layer(dim)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.sub_norm = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            self.causal = causal
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)

        self.qk_normalization = qk_normalization
        self.q_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_normalization else nn.Identity()

        self.k_centering = k_centering

    def _naive_attn(self, x, key_padding_mask=None):
        B, N, C = x.shape
        if self.use_subln:
            qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
            v = self.v(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).squeeze(0)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # qk normalization proposed in ViT-22B paper
        q = self.q_norm(q)
        k = self.k_norm(k)
        if isinstance(q, tuple) and len(q) == 2:
            q = q[0]
            k = k[0]

        if self.k_centering:
            # B, H, N, D
            k = k - k.mean(-2, keepdim=True)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        if key_padding_mask is not None:
            raise NotImplementedError # TODO
        attn = attn - attn.max(-1)[0].unsqueeze(-1) # in case of overflow for fp16
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.sub_norm(x)
        if isinstance(x, tuple) and len(x) == 2:
            x = x[0]
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
            # v1
            # qkv[:, :, 0] = self.q_norm(qkv[:, :, 0])
            # qkv[:, :, 1] = self.k_norm(qkv[:, :, 1])

            # v2
            q, k, v = qkv.unbind(2)
            ori_shape = q.shape
            q = self.q_norm(q.flatten(-2, -1))
            k = self.k_norm(k.flatten(-2, -1))
            if isinstance(q, tuple) and len(q) == 2:
                q = q[0]
                k = k[0]
            q = q.view(ori_shape)
            k = k.view(ori_shape)

            qkv = torch.stack([q, k, v], dim=2)

        if self.k_centering:
            # B, N, H, D
            qkv[:, :, 1] = qkv[:, :, 1] - qkv[:, :, 1].mean(1, keepdim=True)
        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=self.causal
        )
        context = self.sub_norm(rearrange(context, "b s h d -> b s (h d)"))
        if isinstance(context, tuple) and len(context) == 2:
            context = context[0]
        outs = self.proj(context)
        outs = self.proj_drop(outs)
        return outs


    def forward(self, x, key_padding_mask=None):
        x = self._naive_attn(x) if not self.use_flash_attn else self._flash_attn(x, key_padding_mask=key_padding_mask)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        # output_type = x.dtype
        # out = x.float().mul_(self.gamma.float()) if self.inplace else x.float() * self.gamma.float()
        # return out.to(dtype=output_type)
        out = x.mul_(self.gamma) if self.inplace else x * self.gamma
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.,
                 use_subln=False, norm_layer=LayerNorm):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

        self.sub_norm = norm_layer(hidden_features) if use_subln else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.sub_norm(x)
        if isinstance(x, tuple) and len(x) == 2:
            x = x[0]
        x = self.fc2(x)
        x = self.drop2(x)
        return x


import fmoe
from fmoe import FMoE
from fmoe.linear import FMoELinear

def mark_as_moe_params(module):
    for param in module.parameters():
        setattr(param, "is_moe", True)
        setattr(param, "allreduce", False)


class MlpExpert(nn.Module):
    def __init__(self, num_expert=1, in_features=None, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.,
                 use_subln=False, norm_layer=LayerNorm, group=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        self.num_expert = num_expert
        rank = torch.distributed.get_rank(group)
        self.fc1 = FMoELinear(num_expert, in_features, hidden_features, bias=bias[0], rank=rank)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = FMoELinear(num_expert, hidden_features, out_features, bias=bias[1], rank=rank)
        self.drop2 = nn.Dropout(drop_probs[1])

        self.sub_norm = norm_layer(hidden_features) if use_subln else nn.Identity()

    def forward(self, x, cnt):
        # print("rank: ", torch.distributed.get_rank(), "cnt:", cnt)
        x = self.fc1(x, cnt)
        x = self.act(x)
        x = self.drop1(x)
        x = self.sub_norm(x)
        if isinstance(x, tuple) and len(x) == 2:
            x = x[0]
        x = self.fc2(x, cnt)
        x = self.drop2(x)
        return x

class MoEMLP(FMoE):
    def __init__(
        self, num_expert=1, mlp_layer=Mlp, d_model=1, top_k=1, moe_group=None, expert_kwargs={}
    ):
        world_size = torch.distributed.get_world_size(moe_group)
        super().__init__(
            num_expert=num_expert,
            d_model=d_model,
            world_size=world_size,
            mp_group=None,
            top_k=top_k,
            moe_group=moe_group,
            # gate = fmoe.gates.GShardGate,
        )
        expert_kwargs['group'] = moe_group
        self.experts = mlp_layer(**expert_kwargs)
        self.experts_fused = True
        self.num_experts = num_expert
        self.init_gate_weight()
        mark_as_moe_params(self.experts)

    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output = super().forward(inp)
        return output.reshape(original_shape)

    def init_gate_weight(self):
        for p in self.gate.parameters():
            torch.nn.init.normal_(p)

class TransformerEncoderBlock(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm,
            use_flash_attn=False, use_fused_mlp=False, fused_mlp_heuristic=1, with_cp=False,
            use_subln=False, res_post_norm=False, remove_pre_norm=False, post_norm=False,
            qk_normalization=False, k_centering=False):
        super().__init__()
        self.norm1 = norm_layer(dim) if not remove_pre_norm else nn.Identity()
        self.attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                  use_flash_attn=use_flash_attn, causal=False, use_subln=use_subln, norm_layer=norm_layer,
                                  qk_normalization=qk_normalization, k_centering=k_centering)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim) if not remove_pre_norm else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_fused_mlp:
            """
            checkpoint_lvl (increasing lvl means slower but more memory saving):
                0: no recomputation in the bwd
                1: recompute gelu_out in the bwd
                2: recompute pre_act and gelu_out in the bwd
            heuristic:
                -1: don't fuse gemm + gelu (separate kernel)
                0..4: use this heuristic for the algo section in the fused gemm + gelu
                'auto': heuristic will be picked automatically:
                    For CUDA >= 11.8, we set heuristic=0 for both fp16 and bf16 for best perf.
                    For CUDA <= 11.7, we set heuristic=1 for fp16 and heuristic=-1 for bf16.
            """
            assert not use_subln
            self.mlp = FusedMLP(in_features=dim, hidden_features=mlp_hidden_dim, heuristic=fused_mlp_heuristic)
        else:
            # param cnt: a:125,980,032 f: 81,936,000
            # self.mlp0 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, use_subln=use_subln, norm_layer=norm_layer)
            # param cnt: 251 968 002
            from torchdistpackage import tpc
            self.mlp = MoEMLP(num_expert=1, mlp_layer=MlpExpert, d_model=dim, top_k=1,
                              moe_group = tpc.get_group('moe_ep'),
                              expert_kwargs=dict(
                                    num_expert=1, in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, use_subln=use_subln, norm_layer=norm_layer
            ))

            # print(sum([p.numel() for p in self.mlp.parameters()]))
            # print([hasattr(p, 'is_moe') for p in self.mlp.parameters()])
            # import pdb;pdb.set_trace()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.with_cp = with_cp

        self.res_post_norm1 = norm_layer(dim) if res_post_norm else nn.Identity()
        self.res_post_norm2 = norm_layer(dim) if res_post_norm else nn.Identity()

        self.post_norm = norm_layer(dim) if post_norm else nn.Identity()

        self.fused_dal = False

        # if isinstance(self.norm1, DropoutAddRMSNorm) or isinstance(self.norm1, DropoutAddLayerNorm):
        #     self.fused_dal = True
        # else:
        #     self.fused_dal = False

    def forward(self, x, residual=None):
        def _inner_forward(x, residual=None):
            if self.fused_dal:
                x, residual = self.norm1(x, residual)
                x = self.drop_path1(self.ls1(self.res_post_norm1(self.attn(x))))
                x, residual = self.norm2(x, residual)
                x = self.drop_path2(self.ls2(self.res_post_norm2(self.mlp(x))))
                if not isinstance(self.post_norm, nn.Identity):
                    x, _ = self.post_norm(x, residual)
                    residual = None
                return x, residual
            else:
                assert residual is None
                x = x + self.drop_path1(self.ls1(self.res_post_norm1(self.attn(self.norm1(x)))))
                x = x + self.drop_path2(self.ls2(self.res_post_norm2(self.mlp(self.norm2(x)))))
                x = self.post_norm(x)
                return x
        if self.with_cp:
            if residual is None:
                return checkpoint.checkpoint(_inner_forward, x)
            else:
                return checkpoint.checkpoint(_inner_forward, x, residual)
        else:
            return _inner_forward(x, residual=residual)


class TransformerEncoder(nn.Module):

    def __init__(self,
                 depth=12,
                 drop_path_type='linear', # 'linear' or 'uniform'
                 **kwargs):
        super().__init__()

        drop_path = kwargs.pop('drop_path', 0.0)
        if drop_path_type == 'linear':
            drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, depth)]
        elif drop_path_type == 'uniform':
            drop_path_rates = [drop_path for _ in range(depth)]

        self.blocks = nn.ModuleList([TransformerEncoderBlock(**kwargs, drop_path=drop_path_rates[i]) for i in range(depth)])

    def forward(self, x):

        for block in self.blocks:
            if isinstance(x, tuple) and (len(x) == 2):
                x = block(*x)
            else:
                x = block(x)

        if isinstance(x, tuple) and (len(x) == 2):
            x, residual = x
            if residual is not None:
                x = x + residual

        return x