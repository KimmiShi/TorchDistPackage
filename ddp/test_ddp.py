import torch
import torch.nn as nn

import slurm_dist_init.launch_from_slurm.setup_distributed_slurm
import ddp.naive_ddp.SdxDdp

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, input):
        out = self.fc1(input)
        return self.fc2(out)

setup_distributed_slurm()

model = MyModule().cuda()

model = SdxDdp(model)

for _ in range(5):
    x = torch.rand(3,10).cuda()
    out = model(x)
    out.sum().backward()
