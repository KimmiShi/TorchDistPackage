
import torch
import torch.distributed as dist

from transformer import Transformer

from torchdistpackage import fix_rand, setup_distributed_slurm

setup_distributed_slurm()
fix_rand()


def test_model(dim, depth, nh=2):
    model = Transformer(dim, depth=depth, num_heads=nh, tensor_parallel=False, sequence_parallel=False).cuda().bfloat16()
    # tp_model = Transformer(dim, depth=depth, num_heads=nh, tensor_parallel=True, sequence_parallel=False).cuda().bfloat16()
    tp_model = Transformer(dim, depth=depth, num_heads=nh, tensor_parallel=True, sequence_parallel=False).cuda().bfloat16()
    sp_model = Transformer(dim, depth=depth, num_heads=nh, tensor_parallel=True, sequence_parallel=True).cuda().bfloat16()

    opt = torch.optim.AdamW(model.parameters())
    tp_opt = torch.optim.AdamW(tp_model.parameters())

    for ind in range(len(model.blocks)):
        tp_model.blocks[ind].init_from_full(model.blocks[ind])
        sp_model.blocks[ind].init_from_full(model.blocks[ind])

    for _ in range(10):
        inp = torch.rand((32, 1024, dim)).cuda().bfloat16()

        opt.zero_grad()

        out = model(inp)
        tp_out = tp_model(inp)
        sp_out = sp_model(inp)
        import pdb;pdb.set_trace()
        # print(out)
        out.mean().backward()
        tp_out.mean().backward()

        opt.step()
        tp_opt.step()

test_model(4, 2)