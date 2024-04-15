
import torch
import torch.distributed as dist

from torchdistpackage.parallel.tensor_parallel.transformer import Transformer

from torchdistpackage import fix_rand, setup_distributed_slurm

setup_distributed_slurm()
fix_rand()


def test_model(dim, depth, nh=2, dtype=torch.float, seqlen=1024, b=1):
    model = Transformer(dim, depth=depth, num_heads=nh, tensor_parallel=False, sequence_parallel=False).cuda().to(dtype)
    tp_model = Transformer(dim, depth=depth, num_heads=nh, tensor_parallel=True, sequence_parallel=False).cuda().to(dtype)
    # tp_model = Transformer(dim, depth=depth, num_heads=nh, tensor_parallel=True, sequence_parallel=True).cuda().to(dtype)


    opt = torch.optim.AdamW(model.parameters())
    tp_opt = torch.optim.AdamW(tp_model.parameters())

    for ind in range(len(model.blocks)):
        tp_model.blocks[ind].init_from_full(model.blocks[ind])
        # sp_model.blocks[ind].init_from_full(model.blocks[ind])

    for _ in range(10):
        inp = torch.rand((b, seqlen, dim)).cuda().to(dtype)

        opt.zero_grad()

        out = model(inp)
        tp_out = tp_model(inp)
        # import pdb;pdb.set_trace()
        assert torch.allclose(out, tp_out, rtol=1e-3, atol=1e-02)

        import pdb;pdb.set_trace()
        # TODO: fix this mis alignment
        assert torch.allclose(out, tp_out, rtol=1e-2, atol=1e-02)
        assert torch.allclose(out, tp_out, rtol=1e-04, atol=1e-04)
        assert torch.allclose(out, tp_out, rtol=1e-05, atol=1e-05)
        out.mean().backward()
        tp_out.mean().backward()

        opt.step()
        tp_opt.step()

# test_model(1024, 8)
test_model(4, 2, seqlen=5)