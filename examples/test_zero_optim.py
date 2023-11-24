import copy
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import timm

from torchdistpackage import setup_distributed_slurm, NaiveDDP, fix_rand,Bf16ZeroOptimizer


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, input):
        out = self.fc1(input)
        return self.fc2(out)


setup_distributed_slurm()
rank = dist.get_rank()


def test_ddp(model, input_shape, rtol=1e-6):
    model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.00015)
    optim = Bf16ZeroOptimizer(optim, overlap_comm=True)
    my_ddp_model = model


    model2 = copy.deepcopy(model)
    torch_ddp_model = DDP(model2, broadcast_buffers=False)
    optim2 = torch.optim.Adam(torch_ddp_model.parameters(), lr=0.00015)

    for i in range(10):
        # make sure initial param is equal
        for p1, p2 in zip(my_ddp_model.parameters(), torch_ddp_model.parameters()):
            if not torch.allclose(p1, p2, rtol=rtol, atol=rtol):
                import pdb;pdb.set_trace()
                assert False, "model param not equal"

        x = torch.rand(input_shape).cuda() + rank #（should make sure inputs are the same)

        optim.zero_grad()
        optim2.zero_grad()

        out = my_ddp_model(x)
        out.mean().backward()
        if hasattr(my_ddp_model, "reduce_gradients"):
            my_ddp_model.reduce_gradients()

        out2 = torch_ddp_model(x)
        out2.mean().backward()

        if not torch.allclose(out2, out, rtol=rtol, atol=rtol):
            import pdb;pdb.set_trace()
            assert False, "model output not equal"


        optim.step()
        optim2.step()
        if rank == 0:
            print("passed round -- ", i)


def test_simple_module():
    model = MyModule().cuda()
    test_ddp(model, [3, 10])


def test_timm_resnet():
    model = timm.create_model("resnet50", pretrained=False).cuda()
    test_ddp(model, [4, 3, 224, 224], rtol=1e-7)


def test_timm_vit():
    model = timm.create_model("vit_large_patch16_224_in21k", pretrained=True).cuda()
    test_ddp(model, [4, 3, 224, 224], rtol=1e-3)


fix_rand()

test_simple_module()
test_timm_resnet()
# test_timm_vit()   # timm vit's backward is not the same even two ddp model
