import torch
import torch.distributed as dist

from attn import Attention,TpAttention

from torchdistpackage import fix_rand, setup_distributed_slurm

setup_distributed_slurm()
fix_rand()

def test_attn(nh=8, in_dim=1024, drop=0., seq_len=128):
    attn = Attention(in_dim, num_heads=nh, attn_drop=drop, proj_drop=drop).cuda()
    opt = torch.optim.AdamW(attn.parameters())


    tp_attn = TpAttention(in_dim, num_heads=nh, attn_drop=drop, proj_drop=drop).cuda()
    opt_tp = torch.optim.AdamW(tp_attn.parameters())

    tp_attn.qkv.init_weight_from_full_attn(attn.qkv.weight)
    tp_attn.proj.init_weight_from_full(attn.proj.weight)


    for _ in range(2):
        inp = torch.rand((32, seq_len, in_dim)).cuda()

        opt_tp.zero_grad()
        opt.zero_grad()

        out = attn(inp)
        tp_out = tp_attn(inp)

        assert torch.allclose(out, tp_out)
        print("fwd passed")

        out.sum().backward()
        tp_out.sum().backward()

        grad_out_buffer_2 = [torch.empty_like(tp_attn.proj.linear.weight.grad) for _ in range(dist.get_world_size())]
        dist.all_gather(grad_out_buffer_2, tp_attn.proj.linear.weight.grad)
        fc2_grad_full = torch.cat(grad_out_buffer_2, dim=0)

        if not torch.allclose(attn.proj.weight.grad, fc2_grad_full):
            import pdb;pdb.set_trace()
        print("bwd passed")

        opt.step()
        opt_tp.step()



test_attn(nh=8, in_dim=1024, seq_len=128)