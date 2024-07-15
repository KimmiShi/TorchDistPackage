import torch
import torch.distributed as dist

from torchdistpackage import setup_distributed
from torchdistpackage import fix_rand

from torchdistpackage.parallel import TpMlp, Mlp



def test_mlp(dim1=1024, in_feat=2048, outfeat=2048, hiddenfeat=8192):
    input = torch.rand((dim1, in_feat)).cuda()
    dist.broadcast(input, 0)

    mlp = Mlp(in_feat, hiddenfeat, outfeat).cuda()
    tp_mlp = TpMlp(in_feat, hiddenfeat, outfeat).cuda()

    tp_mlp.fc2.init_weight_from_full(mlp.fc2.weight)
    tp_mlp.fc1.init_weight_from_full(mlp.fc1.weight)

    mlp_out = mlp(input)

    tp_mlp_out = tp_mlp(input)

    assert torch.allclose(mlp_out, tp_mlp_out)
    print("fwd passed")

    mlp_out.mean().backward()
    tp_mlp_out.mean().backward()

    grad_out_buffer_2 = [torch.empty_like(tp_mlp.fc2.linear.weight.grad) for _ in range(dist.get_world_size())]
    dist.all_gather(grad_out_buffer_2, tp_mlp.fc2.linear.weight.grad)
    fc2_grad_full = torch.cat(grad_out_buffer_2, dim=0)

    assert torch.allclose(mlp.fc2.weight.grad, fc2_grad_full)

    grad_out_buffer_1 = [torch.empty_like(tp_mlp.fc1.linear.weight.grad) for _ in range(dist.get_world_size())]
    dist.all_gather(grad_out_buffer_1, tp_mlp.fc1.linear.weight.grad)
    fc1_grad_full = torch.cat(grad_out_buffer_1, dim=1)
    assert torch.allclose(mlp.fc1.weight.grad, fc1_grad_full, rtol=1e-04, atol=1e-04)
    print("bwd passed")



if __name__=='__main__':
    fix_rand()
    setup_distributed()
    test_mlp()
