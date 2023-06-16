# this file demostrates how to use fmoe with deepspeed zero2 training a vit

import torch
from models.backbones.vit import *

from torchdistpackage import setup_distributed_slurm,tpc
from torchdistpackage import moe_dp_iter_step,create_moe_dp_hooks

from torch.nn.parallel import DistributedDataParallel

import os
import deepspeed
from deepspeed.utils import groups
from deepspeed.comm.comm import init_distributed#,_EXPERT_PARALLEL_GROUP

expert_parallel_size_=2
def get_moe_params(model):
    moe_params = {}
    for name,param in model.named_parameters():
        if hasattr(param, 'is_moe'):
            moe_params[name] = param
    return moe_params

def split_moe_param_group(model):
    group_name = f"ep_size_{expert_parallel_size_}"
    moe_params = []
    non_moe_params = []
    for name,param in model.named_parameters():
        if hasattr(param, 'is_moe'):
            moe_params.append(param)

            setattr(param, 'group_name', group_name)
        else:
            non_moe_params.append(param)
    return [dict(params=non_moe_params),dict(params=moe_params, moe=True, name=group_name)]


rank, world_size, port, addr=setup_distributed_slurm()
dist_config = [('data', int(os.environ["SLURM_NTASKS"]))]
tpc.setup_process_groups(dist_config)
tpc.build_moe_groups(moe_ep_size=expert_parallel_size_)

init_distributed(dist_backend='nccl')
groups._create_expert_and_data_parallel(expert_parallel_size_)

torch.cuda.set_device(int(os.environ['SLURM_PROCID']) % torch.cuda.device_count())

# config = 'i'
image_size = 196
patch_size = 14
batch_size = 512
mask_ratio = 0.5
for config in ['ff']:
    print(f"[config={config}, image_size={image_size}, patch_size={patch_size}, batch_size={batch_size}, mask_ratio={mask_ratio}]")
    # model = fordebug_onelayer_vit(config=config, image_size=image_size, patch_size=patch_size)
    model = vit_6b(config=config, image_size=image_size, patch_size=patch_size)
    x = torch.zeros((batch_size, 3, image_size, image_size), dtype=torch.bfloat16).cuda().uniform_(-1, 1)*(rank+1)
    model = model.to(dtype=torch.bfloat16).cuda()
    param_groups = split_moe_param_group(model)
    optimizer = torch.optim.AdamW(param_groups)
    # moe_opt =  torch.optim.AdamW(param_groups[1:])

    # import pdb;pdb.set_trace()
    moe_params = get_moe_params(model)
    model._ddp_params_and_buffers_to_ignore = list(moe_params.keys())
    # create_moe_dp_hooks(moe_params, tpc.get_group('moe_dp'), tpc.get_ranks_in_group('moe_dp')[0])

    # model = DistributedDataParallel(model)

    use_ds=True
    model, optimizer, _, _ = deepspeed.initialize(config='/mnt/petrelfs/shidongxing/work/pytorch-image-models-private/configs/adam_zero2_bf16.json',
                                                  model=model,
                                                  optimizer=optimizer)
    num_steps = 5
    warm_steps = 10
    import time
    for _ in range(num_steps):
        if _ == warm_steps:
            tic = time.time()
        y = model(x, mask_ratio=mask_ratio)
        loss = y.mean()
        # moe_opt.zero_grad()
        # print(moe_params['encoder.blocks.21.mlp.experts.fc2.bias'])
        if use_ds:
            model.backward(loss)
            # moe_dp_iter_step()
            # torch.cuda.synchronize()
            model.step()
            # moe_opt.step()
            torch.cuda.synchronize()
            # import pdb;pdb.set_trace()
            print(model.encoder.blocks[22].mlp.experts.fc1.weight[0][30], flush=True)
            # print(moe_params['encoder.blocks.21.mlp.experts.fc2.bias'])
        else:
            optimizer.zero_grad()
            loss.backward()
            torch.cuda.synchronize()
            optimizer.step()
    # iter_time = (time.time() - tic) / (num_steps - warm_steps)
    # fps = x.shape[0] / iter_time
    # print("FPS:", fps)
    print("MAX MEM", torch.cuda.max_memory_allocated()/1e9, rank)