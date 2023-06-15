import os
import time

import matplotlib.pyplot as plt
import torch
from timm.models import create_model
from timm.models.vision_transformer import VisionTransformer
from torch.cuda.amp import GradScaler, autocast

from functorch.compile import memory_efficient_fusion
from functorch.compile import min_cut_rematerialization_partition
from functorch.compile import ts_compile
from functorch.compile import aot_function, \
    make_boxed_func, ts_compile
from functorch.compile import ts_compile, aot_module

def train_aot(model,input, amp_enable=False, warmup=False):
    # aot_compiled_mod = aot_module(model,
    #                               fw_compiler=ts_compile,
    #                               bw_compiler=ts_compile)
    aot_compiled_mod=memory_efficient_fusion(model)
    for i in range(0, 3):
        with autocast(enabled=amp_enable):
            out=aot_compiled_mod(input)
        out.sum().backward()
    torch.cuda.synchronize()
    time_start=time.time()
    for i in range(0, 10):
        with autocast(enabled=amp_enable):
            out=aot_compiled_mod(input)
        out.sum().backward()

    torch.cuda.synchronize()
    if not warmup:
        print(f'time used: {time.time()-time_start}')
    return time.time()-time_start


def train(model, input, amp_enable=False, warmup=False):
    for i in range(0, 3):
        with autocast(enabled=amp_enable):
            out=model(input)
        out.sum().backward()
    torch.cuda.synchronize()
    time_start=time.time()
    for i in range(0, 10):
        with autocast(enabled=amp_enable):
            out=model(input)
        out.sum().backward()

    torch.cuda.synchronize()
    if not warmup:
        print(f'time used: {time.time()-time_start}')
    return time.time()-time_start


def train_g(model, input, amp_enable=False, warmup=False):
    # Placeholders used for capture
    static_input = torch.randn_like(input, device='cuda')
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with autocast(enabled=amp_enable):
            out=model(static_input)
        out.sum().backward()

    static_input.copy_(input)
    torch.cuda.synchronize()
    time_start=time.time()
    for i in range(0, 5):
        g.replay()

    torch.cuda.synchronize()
    if not warmup:
        print(f'time used: {time.time()-time_start}')

def train_g_2(model, input, amp_enable=False, warmup=False):
    torch.cuda.synchronize()
    model_g = torch.cuda.make_graphed_callables(model, (input,))
    time_start=time.time()
    for i in range(0, 5):
        with autocast(enabled=amp_enable):
            out=model_g(input)
        out.sum().backward()

    torch.cuda.synchronize()
    print(f'time used: {time.time()-time_start}')

def run_train_funcs(train, model, input):
    # warmup, ignore
    train(model, input, warmup=True)

    # torch.backends.cuda.matmul.allow_tf32=False
    # torch.backends.cudnn.allow_tf32=False
    os.environ['NVIDIA_TF32_OVERRIDE']='0'
    print('----train with fp32----')
    train(model, input)
    print('----train with autocast----')
    train(model, input, amp_enable=True)
    print('----train with tf32----')
    os.environ['NVIDIA_TF32_OVERRIDE']='1'
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    train(model, input)

print(torch.backends.cuda.matmul.allow_tf32)
print(torch.backends.cudnn.allow_tf32)

def enable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def disable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True


enable_tf32()
model=create_model('vit_large_patch16_224', #vit_large_patch16_384
                   pretrained=False,
                   num_classes=None,
                   drop_rate=0,
                   drop_path_rate=0.3)
model.cuda().train()

def gen_inp(batch=64):
    return torch.rand([batch, 3, 224, 224])

# train(model, gen_inp().cuda())

# train(model, gen_inp().cuda())

train_aot(model, gen_inp().cuda())

print("max mem", torch.cuda.max_memory_allocated()/(1e9))