import math
from functools import partial
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
# from timm.models.layers import to_2tuple, trunc_normal_
# from timm.models.layers import DropPath

# from timm.models.registry import register_model
from einops import rearrange

from torch import Tensor, Size
from typing import Union, List
try:
    from flash_attn.flash_attention import FlashAttention
    from flash_attn.modules.mlp import FusedMLP
    from flash_attn.ops.rms_norm import DropoutAddRMSNorm
except:
    print("flash attention is not installed.")

def setup_distributed_slurm(backend="nccl", port=None):
    import os
    import subprocess
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        local_rank = rank % num_gpus

        addr = subprocess.getoutput(
            f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            port = 54647
            os.environ["MASTER_PORT"] = str(port)
        else:
            port = int(os.environ["MASTER_PORT"])
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = rank % num_gpus

    dist.init_process_group(rank=rank, world_size=world_size, backend=backend)

    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    print(f"dist init done, world_size = {dist.get_world_size()}")
    return rank, world_size, port, addr

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


_shape_t = Union[int, List[int], Size]



class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                 device=None, dtype=None, force_fp32=True):
        self.force_fp32 = force_fp32
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input):
        # return super(LayerNorm, self).forward(input.float())
        if self.force_fp32:
            output_type = input.dtype
            return F.layer_norm(
                input.float(), self.normalized_shape, self.weight.float() if self.weight is not None else None,
                self.bias.float() if self.bias is not None else None, self.eps).to(dtype=output_type)
        else:
            return F.layer_norm(
                input, self.normalized_shape, self.weight if self.weight is not None else None,
                self.bias if self.bias is not None else None, self.eps)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False, force_fp32=True):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        self.force_fp32 = force_fp32

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        if self.force_fp32:
            output_type = x.dtype
            out = x.float().mul_(self.gamma.float()) if self.inplace else x.float() * self.gamma.float()
            return out.to(dtype=output_type)
        else:
            out = x.mul_(self.gamma) if self.inplace else x * self.gamma
            return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 use_flash_attn=False, causal=False, use_subln=False, norm_layer=LayerNorm,
                 qk_normalization=False, k_centering=False, qk_normalization_head_merged=False,
                 fuse_dal=False):
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
        self.qk_normalization_head_merged = qk_normalization_head_merged
        qk_norm_dim = dim if qk_normalization_head_merged else head_dim
        self.q_norm = norm_layer(qk_norm_dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(qk_norm_dim) if qk_normalization else nn.Identity()
        self.fuse_dal = fuse_dal

        self.k_centering = k_centering

    def _naive_attn(self, x):
        B, N, C = x.shape
        if self.use_subln:
            qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k = qk.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
            v = self.v(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).squeeze(0)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # qk normalization proposed in ViT-22B paper
        if self.qk_normalization:
            if self.qk_normalization_head_merged:
                # B, H, N, D
                B_, H_, N_, D_ = q.shape
                q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
                k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            else:
                q = self.q_norm(q)
                k = self.k_norm(k)

        if self.k_centering:
            # B, H, N, D
            k = k - k.mean(-2, keepdim=True)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn - attn.max(-1)[0].unsqueeze(-1)  # in case of overflow for fp16
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.sub_norm(x)
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


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.,
                 use_subln=False, norm_layer=LayerNorm):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # bias = to_2tuple(bias)
        drop_probs = drop

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs)

        self.sub_norm = norm_layer(hidden_features) if use_subln else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.sub_norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm, use_flash_attn=False,
            use_fused_mlp=False, fused_mlp_heuristic=1, with_cp=False, use_subln=False,
            res_post_norm=False, remove_pre_norm=False, post_norm=False, qk_normalization=False,
            k_centering=False, qk_normalization_head_merged=False, layernorm_no_force_fp32=False,
            layerscale_no_force_fp32=False, fuse_dal=False):
        super().__init__()

        if layernorm_no_force_fp32 and (not fuse_dal):
            norm_layer = partial(norm_layer, force_fp32=False)

        self.norm1 = norm_layer(dim) if not remove_pre_norm else nn.Identity()
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                              use_flash_attn=use_flash_attn, causal=False, use_subln=use_subln, norm_layer=norm_layer,
                              qk_normalization=qk_normalization, k_centering=k_centering,
                              qk_normalization_head_merged=qk_normalization_head_merged,
                              fuse_dal=fuse_dal)
        self.ls1 = LayerScale(dim, init_values=init_values,
                              force_fp32=(not layerscale_no_force_fp32)) if init_values else nn.Identity()
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
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                           use_subln=use_subln, norm_layer=norm_layer)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.with_cp = with_cp

        self.res_post_norm1 = norm_layer(dim) if res_post_norm else nn.Identity()
        self.res_post_norm2 = norm_layer(dim) if res_post_norm else nn.Identity()

        self.post_norm = norm_layer(dim) if post_norm else nn.Identity()

        self.fuse_dal = fuse_dal

    def forward(self, x, residual=None):

        def _inner_forward(x, residual=None):
            if self.fuse_dal:
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


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size,img_size)
        patch_size = (patch_size,patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., init_values=None, use_mean_pooling=True,
                 init_scale=0.001, use_checkpoint=True, stop_grad_conv1=True,
                 use_abs_pos_emb=True,
                 subln=False,
                 xattn=False,
                 use_fused_mlp: bool = True,
                 fused_mlp_heuristic: int = 1,
                 post_norm_every_six_blocks: bool = False,
                 qk_normalization_head_merged: bool = True,
                 k_centering: bool = False,
                 qk_normalization: bool = True,
                 use_fused_rmsnorm: bool = True,
                 layernorm_no_force_fp32: bool = True,
                 layerscale_no_force_fp32: bool = True,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.use_checkpoint = use_checkpoint
        self.stop_grad_conv1 = stop_grad_conv1

        norm_layer_for_blocks = partial(DropoutAddRMSNorm, eps=1e-6, prenorm=True)
        # norm_layer_for_blocks = partial(LayerNorm, eps=1e-6, prenorm=True)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer_for_blocks,
                  drop_path=dpr[i], init_values=init_values, attn_drop=attn_drop_rate,
                  use_flash_attn=xattn, use_fused_mlp=use_fused_mlp,
                  fused_mlp_heuristic=fused_mlp_heuristic,
                  with_cp=use_checkpoint,
                  use_subln=subln,
                  # res_post_norm=args.res_post_norm,
                  # remove_pre_norm=args.remove_pre_norm,
                  post_norm=(post_norm_every_six_blocks and (i + 1) % 6 == 0),
                  qk_normalization=qk_normalization,
                  k_centering=k_centering,
                  qk_normalization_head_merged=qk_normalization_head_merged,
                  layernorm_no_force_fp32=layernorm_no_force_fp32,
                  layerscale_no_force_fp32=layerscale_no_force_fp32,
                  fuse_dal=use_fused_rmsnorm)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else nn.LayerNorm(embed_dim)
        self.fc_norm = nn.LayerNorm(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # if self.pos_embed is not None:
        #     trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        # # trunc_normal_(self.mask_token, std=.02)
        # if isinstance(self.head, nn.Linear):
        #     trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_vit_blocks(self, blocks, x):
        residual = None
        for blk in blocks:
            if isinstance(x, tuple) and len(x) == 2:
                x, residual = x
            x = blk(x, residual=residual)
        if isinstance(x, tuple) and len(x) == 2:
            x, residual = x
            if residual is not None:
                x = x + residual
        return x

    def forward_features(self, x, return_patch_tokens=False):
        x = self.patch_embed(x.type(self.dtype))
        if self.stop_grad_conv1:
            x = x.detach()
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.forward_vit_blocks(self.blocks, x)

        x = self.norm(x)
        if self.fc_norm is not None:
            t = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(t)
            else:
                return self.fc_norm(t.mean(1))
        else:
            if return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

    def forward(self, x, return_patch_tokens=False):
        x = self.forward_features(x, return_patch_tokens=return_patch_tokens)
        x = self.head(x)
        return x


def process_checkpoint(ckpt):
    new_ckpt = {}
    for k, v in ckpt['module'].items():
        if "bamboo" in k or "predictor" in k or "decoder" in k or "loss" in k:
            continue
        if "mask_token" in k or "clip_projector" in k:
            continue
        if "transformer." in k or "grad_norm_square" in k:
            continue
        if "target_pos_embed" in k or "norm3" in k:
            continue
        if "text_projection" in k or "logit_scale" in k:
            continue
        new_k = k.replace("clip.transformer.", "transformer.")
        new_k = new_k.replace("clip.text_projection", "text_projection")
        new_k = new_k.replace("clip.logit_scale", "logit_scale")

        new_ckpt[new_k] = v
    return new_ckpt


def vit_patch14_flashattn(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1200, depth=12, num_heads=25, mlp_ratio=4, qkv_bias=False,
        xattn=True,
        naiveswiglu=False,
        subln=False,
        rope=False,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        new_ckpt = process_checkpoint(checkpoint)
        message = model.load_state_dict(new_ckpt, strict=False)
        print(message)
    return model

import deepspeed

def create_moe_param_groups(model):
    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

    parameters = {'params': [p for p in model.parameters()], 'name': 'parameters'}

    return split_params_into_different_moe_groups_for_optimizer(parameters)

def ds_cfg():
    return {
        "train_batch_size": 128,
        "gradient_accumulation_steps": 1,
        "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.00015
        }
        },
        "bf16": {
        "enabled": True,
        "auto_cast": True
        },
        "zero_optimization": {
        "stage": 0,
        "contiguous_gradients": True,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
        },
        "steps_per_print": 50
        }

if __name__ == "__main__":
    from deepspeed.utils import groups
    from deepspeed.comm.comm import init_distributed

    # need to initialize torch.distributed, this line depends on slurm, you can switch to another way
    setup_distributed_slurm()
  
    init_distributed(dist_backend='nccl')
    ep_size=1
    groups._create_expert_and_data_parallel(ep_size)

    model = vit_patch14_flashattn(init_values=1e-5, pretrained=False).cuda().bfloat16()
    for block in model.blocks:
        original_mlp = block.mlp
        feat_in = original_mlp.fc1.weight.shape[1]
        block.mlp = deepspeed.moe.layer.MoE(hidden_size=feat_in, expert=block.mlp,k=1,
                                            num_experts=ep_size, ep_size=ep_size, use_fmoe=True,
                                            drop_tokens=True).cuda().bfloat16()

    parameters = create_moe_param_groups(model)
    model, optimizer, _, _ = deepspeed.initialize(config=ds_cfg(),
                                                  model=model,
                                                  model_parameters=parameters)


    for i in range(2):
        out=model(torch.rand(2,3,224,224).cuda().bfloat16())
        # out.mean().backward
        model.backward(out.mean())
        # model.step()
        torch.cuda.synchronize()
        print("finished step ", i)
