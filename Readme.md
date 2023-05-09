# TorchDistPackage

TorchDistPackage provides some easy-to-use modules and tools for Distributed Training in PyTorch.

It is under construction. Welcome to use and contribute.

[主要特性介绍](./Intro.md)

# Simple DDP Module in PyTorch

code and example: [NaiveDdp](./torchdistpackage/ddp/naive_ddp.py)

**Highlights:**

- Python only implementation. Easy to understand and debug.
- overlaps grad reduce with compute like [TorchDDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- For Pipeline Parallelism, only reduce grad at the last micro-batch; and could still overlap comm, which is better than [ColossalAI impl](https://github.com/hpcaitech/ColossalAI/blob/2a951955ade14fd067bc5bee34a5ff7e57513ac6/colossalai/initialize.py#L385).

**Drawbacks/TODO:**

- the all-reduce launch seems to have taken more time than TorchDDP


# 小特性介绍

## 1. 工具： 从slurm初始化torch distributed
[torch dist init from slurm](./torchdistpackage/dist/launch_from_slurm.py)

## 2. 组件：通信组划分 Flexible process group initialization for Mixed Parallelism

**Highlights:**

Generally, DataParallel Group placement is at the most 'outside', and ModelParallel group is located 'inside'.

However, in some cases, we would like the DataParallel Group to be placed 'inside'. [topo.py](./dist_init/topo.py) enables us to place DP,TP,PP groups in the order we want it to be.

> "inside" mean the ranks in this group are more likely to be adjacent
>
> "outside" means the ranks in this group are more likely to be at different GPU nodes.

### example

> place `data parallel` group inside-node, and `pipeline parallel` group cross-node.
>
> enables faster ZeRO communication which uses `data parallel` group and slower `pipeline parallel` P2P communication.

```
# world_size=16
# pp_size=2

    dist_config = [('pipe',pp_size), ('tensor',2), ('data',world_size/(2*pp_size))]
    tpc.setup_process_groups(dist_config)
```

output:
```
group pipe, ranks: [0, 8]
group pipe, ranks: [1, 9]
group pipe, ranks: [2, 10]
group pipe, ranks: [3, 11]
group pipe, ranks: [4, 12]
group pipe, ranks: [5, 13]
group pipe, ranks: [6, 14]
group pipe, ranks: [7, 15]
group tensor, ranks: [0, 4]
group tensor, ranks: [8, 12]
group tensor, ranks: [1, 5]
group tensor, ranks: [9, 13]
group tensor, ranks: [2, 6]
group tensor, ranks: [10, 14]
group tensor, ranks: [3, 7]
group tensor, ranks: [11, 15]
group data, ranks: [0, 1, 2, 3]
group data, ranks: [4, 5, 6, 7]
group data, ranks: [8, 9, 10, 11]
group data, ranks: [12, 13, 14, 15]
group model, ranks: [0, 4, 8, 12]
group model, ranks: [1, 5, 9, 13]
group model, ranks: [2, 6, 10, 14]
group model, ranks: [3, 7, 11, 15]
```

## 3. 组件：流水并行相关
- 自定义fwd_fn,bwd_fn的1F1B调度器 [pipeline scheduler](./torchdistpackage/parallel/pipeline_sched.py)
- pipeline model partition [流水并行模型切分](./torchdistpackage/parallel/pipeline_helper.py)