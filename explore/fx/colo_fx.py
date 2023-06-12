import torch
from torch.fx import symbolic_trace
import colossalai

from colossalai.fx.passes.meta_info_prop import MetaInfoProp

BATCH_SIZE = 2
DIM_IN = 4
DIM_HIDDEN = 16
DIM_OUT = 16
model = torch.nn.Sequential(
    torch.nn.Linear(DIM_IN, DIM_HIDDEN),
    torch.nn.Linear(DIM_HIDDEN, DIM_OUT),
    )
input_sample = torch.rand(BATCH_SIZE, DIM_IN)
gm = symbolic_trace(model)
interp = MetaInfoProp(gm)
interp.run(input_sample)
print(interp.summary(unit='kb'))    # don't panic if some statistics are 0.00 MB
