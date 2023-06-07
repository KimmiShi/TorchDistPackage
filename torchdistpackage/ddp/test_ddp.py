import copy
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import timm

from torchdistpackage import setup_distributed_slurm, NaiveDDP


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


def test_ddp(model, input_shape):
    model = model.cuda()
    model2 = copy.deepcopy(model)

    # my_ddp_model = NaiveDDP(model, sync=True, gradient_as_bucket_view=False)
    my_ddp_model = DistributedDataParallel(model, None, True,True)
    torch_ddp_model = DDP(model2, broadcast_buffers=False)

    optim = torch.optim.Adam(my_ddp_model.parameters(), lr=0.00015)
    optim2 = torch.optim.Adam(torch_ddp_model.parameters(), lr=0.00015)

    for i in range(10):
        # make sure initial param is equal
        for p1, p2 in zip(my_ddp_model.parameters(), torch_ddp_model.parameters()):
            if (p1.data -p2.data).sum() > 1e-5:
                assert False, "model param not equal"

        x = torch.rand(input_shape).cuda() + rank

        optim.zero_grad()
        optim2.zero_grad()

        out = my_ddp_model(x)
        out.sum().backward()
        if hasattr(my_ddp_model, "reduce_gradients"):
            my_ddp_model.reduce_gradients()

        out2 = torch_ddp_model(x)
        out2.sum().backward()

        assert torch.allclose(out2, out)

        compare grads
        for p1, p2 in zip(my_ddp_model.parameters(), torch_ddp_model.parameters()):
            if (p1.grad- p2.grad).sum() > 1e-6:
                if not torch.allclose(p1.grad, p2.grad):
                    import pdb;pdb.set_trace()
                    assert False

        optim.step()
        optim2.step()
        if rank == 0:
            print("passed round -- ", i)

def test_simple_module():
    model = MyModule().cuda()
    test_ddp(model, [3,10])

def test_timm_resnet():
    model = timm.create_model(
            'resnet50',
            pretrained=False).cuda()
    test_ddp(model, [4, 3, 224, 224])

def test_timm_vit():
    model = timm.create_model(
            'vit_large_patch16_224_in21k',
            pretrained=False).cuda()
    test_ddp(model, [4, 3, 224, 224])

fix_random_seed()

test_simple_module()
test_timm_resnet()
# test_timm_vit()