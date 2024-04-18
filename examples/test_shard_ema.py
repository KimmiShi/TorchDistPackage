import torch
from copy import deepcopy
from collections import OrderedDict

from torchdistpackage import ShardedEMA

from torchdistpackage import setup_distributed, test_comm, fix_rand

fix_rand()

@torch.no_grad()
def update_params(model):
    for name, param in model.named_parameters():
        param.add_(torch.randn_like(param))

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    ema_params = OrderedDict(ema_model.named_parameters())
    if isinstance(model, DDP):
        model = model.module
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def test_ema(model):
    ema = deepcopy(model)
    ema.requires_grad_(False)

    sd_ema = ShardedEMA(model)

    for iter in range(100):
        update_params(model)

        update_ema(ema, model, decay=0.999)

        sd_ema.update(model, decay=0.999)

    # compare

    sd_ema_params = sd_ema.state_dict_cpu()

    if torch.distributed.get_rank()==0:
        for name, gt_param in ema.named_parameters():
            sd_param = sd_ema_params[name]
            if not torch.equal(gt_param.cpu(), sd_param):
                print(name, (gt_param.cpu()-sd_param).sum())
            assert torch.equal(gt_param.cpu(), sd_param)

    print("=========== test passed ===========")

import torchvision.models as models

setup_distributed()
test_comm()

model = models.resnet50().cuda()

test_ema(model)
