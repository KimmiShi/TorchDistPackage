import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import functools

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from torchdistpackage import setup_distributed_slurm
import sdto

def enable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def train(model, vae, optimizer_class, batchsize, use_zero=False, use_amp=True, h=512, w=512, is_xl=False,
          input_grad=False):
    dt=torch.float32

    if not is_xl:
        timesteps = torch.arange(batchsize, dtype=torch.int64).cuda()+100
        prompt_embeds = torch.rand([batchsize,77,768], dtype=dt).cuda()
        time_ids = torch.rand([batchsize,6], dtype=dt).cuda()
        text_embeds = torch.rand([batchsize,1280], dtype=dt).cuda()
        encoder_hidden_states = torch.rand([batchsize,77,768], dtype=dt).cuda()

    else:
        timesteps = torch.arange(batchsize, dtype=torch.int64).cuda()+100
        prompt_embeds = torch.rand([batchsize,77,2048], dtype=dt).cuda()
        time_ids = torch.rand([batchsize,6], dtype=dt).cuda()
        text_embeds = torch.rand([batchsize,1280], dtype=dt).cuda()


    # model.to(dt)
    model.cuda()

    model_input = torch.rand([batchsize, 3, h, w], dtype=torch.float32).cuda()

    # noisy_model_input = torch.rand([batchsize, 4, h//8, w//8], dtype=dt, device='cuda').requires_grad_(input_grad)
    unet_added_conditions = {
        "time_ids": time_ids,
        "text_embeds": text_embeds
    }


    # model.enable_gradient_checkpointing()
    # model.enable_xformers_memory_efficient_attention()
    model, vae, opt, _ = sdto.compile(model, vae, compile_vae=True)

    perf_times = []
    if use_zero:
        model = FSDP(model, sharding_strategy=torch.distributed.fsdp.ShardingStrategy.SHARD_GRAD_OP)
        opt =optimizer_class(model.parameters())
    else:
        opt =optimizer_class(model.parameters())

    from torch.profiler import profile, record_function, ProfilerActivity
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=5, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./bnb_int8'),
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True)
    # prof.start()
    for ind in range(10):
        torch.cuda.synchronize()
        beg = time.time()

        # noisy_model_input = vae.encode(model_input).latent_dist.sample().mul_(0.18215)
        noisy_model_input = sdto.sliced_vae(vae, model_input, use_autocast=True)

        with torch.autocast(dtype=torch.float16, device_type='cuda', enabled=use_amp):
            if is_xl:
                model_pred = model(
                            noisy_model_input, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions
                        ).sample
                # print("after fwd", torch.cuda.memory_allocated()/1e9, torch.cuda.max_memory_allocated()/1e9)
                # import pdb;pdb.set_trace()
                loss = F.mse_loss(model_pred.float(), torch.rand_like(model_pred).float(), reduction="mean")
            else:
                model_pred = model(noisy_model_input, timesteps, encoder_hidden_states).sample
                # print("after fwd", torch.cuda.memory_allocated()/1e9, torch.cuda.max_memory_allocated()/1e9)
                # import pdb;pdb.set_trace()
                loss = F.mse_loss(model_pred.float(), torch.rand_like(model_pred).float(), reduction="mean")

        loss.backward()
        # print("after bwd", torch.cuda.max_memory_allocated()/1e9)

        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
        # prof.step()
        if ind>5:
           perf_times.append(time.time()-beg)
        beg=time.time()
    print("max mem", torch.cuda.max_memory_allocated()/1e9)
    print(perf_times)

enable_tf32()
rank, world_size, port, addr=setup_distributed_slurm()

pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"

# pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"

unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet"
).cuda()
unet.train()
vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae").cuda()

# print(unet)
# optimizer_class = FusedAdam
optimizer_class = functools.partial(torch.optim.Adam,fused = True)

train(unet, vae, optimizer_class, 96,
      use_amp=False, use_zero=False, h=576, w=1024, is_xl ='xl' in pretrained_model_name_or_path,
      input_grad=True)