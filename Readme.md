# TorchDistPackage

TorchDistPackage provides some easy-to-use modules and tools for Distributed Training in PyTorch.

It is under construction. Welcome to use and contribute.

[主要特性介绍-中文](./Intro.md)

# 安装使用

- install
```sh
git clone https://github.com/KimmiShi/TorchDistPackage.git
cd TorchDistPackage
pip install -e . # or pip install . --user
```

- simple example
```py
import torch
from torchdistpackage import setup_distributed,test_comm,tpc

# init torch disttributed
setup_distributed()

# init process groups
pp_size=2
tp_size=2
dist_config = [('data',world_size/(2*pp_size)), ('pipe',pp_size), ('tensor',tp_size)]
tpc.setup_process_groups(dist_config)

# test communication in groups
tmp = torch.rand([100,1024]).cuda()

# collective
dist.broadcast(tmp, tpc.get_ranks_in_group('model')[0], tpc.get_group('model'))

# p2p
if tpc.is_first_in_pipeline_group():
    dist.send(tmp, tpc.get_next_global_rank('pipe'))
if tpc.is_last_in_pipeline_group():
    dist.recv(tmp, tpc.get_prev_global_rank('pipe'))

```

# 特性介绍

## 0. 简单的纯Python实现DDP - Simple DDP Module in PyTorch

example: [TestNaiveDdp](./torchdistpackage/ddp/test_ddp.py)

**Highlights:**

- Python only implementation. Easy to understand and debug.
- overlaps grad reduce with compute like [TorchDDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- For Pipeline Parallelism, only reduce grad at the last micro-batch; and could still overlap comm, which is better than [ColossalAI impl](https://github.com/hpcaitech/ColossalAI/blob/2a951955ade14fd067bc5bee34a5ff7e57513ac6/colossalai/initialize.py#L385).

**Drawbacks/TODO:**

- the all-reduce launch seems to take more time than TorchDDP in some model


## 1. 从slurm初始化torch distributed - torch_launch_from_slurm
[torch dist init from slurm](./torchdistpackage/dist/launch_from_slurm.py)

[example](#安装使用)


## 2. 灵活的通信组划分 - Flexible process group initialization for Mixed Parallelism

详见[主要特性介绍](./Intro.md)


## 3. 流水并行相关 - For Pipeline Parallelism
- 自定义fwd_fn,bwd_fn的1F1B调度器 [pipeline scheduler](./torchdistpackage/parallel/pipeline_parallel/pipeline_sched.py)
- pipeline model partition [流水并行模型切分](./torchdistpackage/parallel/pipeline_parallel/pipeline_helper.py)

[使用示例](./torchdistpackage/parallel/pipeline_parallel/pipeline.md)
[测例参考](./examples/model_parallel/test_pipeline.py)

## 4. MoE-数据并行
在专家并行(Expert Parallel)的基础上，支持 MoE 数据并行：即复制一些expert，相同的expert之间做数据并行（初始参数广播，梯度平均），不同的expert之间做专家并行。

[使用示例](./torchdistpackage/ddp/moe_dp.md)


## 5. Tensor Parallel & Sequence Parallel
简单的TP实现。

[测例参考](./examples/model_parallel/test_transformer.py)

## 6. Hybrid ZeRO / 节点内ZeRO - 加速ZeRO多卡训练速度

详见[主要特性介绍](./Intro.md)


## 7. 分片EMA - sharded EMA
节省EMA的显存消耗，见[sharded ema example](./examples/test_shard_ema.py)

# TOOLS 工具类
## 1. model profiler
分级时间和显存消耗
[参考](./torchdistpackage/tools/module_profile.md)

