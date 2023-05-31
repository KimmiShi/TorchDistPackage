import copy
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
device = rank % torch.cuda.device_count()
torch.cuda.set_device(device)

model = MyModule().cuda()
model2 = copy.deepcopy(model)

my_ddp_model = NaiveDDP(model, sync=False, gradient_as_bucket_view=True)
torch_ddp_model = DDP(model2)

for _ in range(3):
    # make sure initial param is equal
    for p1, p2 in zip(my_ddp_model.parameters(), torch_ddp_model.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            assert False, "model param not equal"

    x = torch.rand(3, 10).cuda() + rank
    out = my_ddp_model(x)
    out.sum().backward()
    my_ddp_model.reduce_gradients()

    out2 = torch_ddp_model(x)
    out2.sum().backward()

    assert torch.allclose(out2, out)

    # compare grads
    for p1, p2 in zip(my_ddp_model.parameters(), torch_ddp_model.parameters()):
        if p1.grad.ne(p2.grad).sum() > 0:
            import pdb

            pdb.set_trace()
            if not torch.allclose(p1.grad, p2.grad):
                assert False

    my_ddp_model.zero_grad()
    torch_ddp_model.zero_grad()
    if rank == 0:
        print("passed round -- ")
