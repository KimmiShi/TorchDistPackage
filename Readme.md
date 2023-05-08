# TorchDistPackage

TorchDistPackage provides some easy-to-use modules and tools for Distributed Training in PyTorch.

It is under construction. Welcome to use and contribute.

# Simple DDP Module in PyTorch

code and example: [NaiveDdp](./ddp)

**Highlights:**

- Python only implementation. Easy to understand and debug.
- overlaps grad reduce with compute like [TorchDDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- For Pipeline Parallelism, only reduce grad at the last micro-batch; and could still overlap comm, which is better than [ColossalAI impl](https://github.com/hpcaitech/ColossalAI/blob/2a951955ade14fd067bc5bee34a5ff7e57513ac6/colossalai/initialize.py#L385).

**Drawbacks/TODO:**

- the all-reduce launch seems to have taken more time than TorchDDP


# Toolkit

## 1. [torch dist init from slurm](./slurm_dist_init/)

## 2. Flexible process group initialize for Mixed Parallelism

**References**:
- https://github.com/hpcaitech/ColossalAI


**Highlights:**

Generally, DataParallel Group placement is at the most 'outside', and ModelParallel group is located 'inside'.

However, in some cases, we would like the DataParallel Group to be placed 'inside'. [topo.py](./dist_init/topo.py) enables us to place DP,TP,PP groups in the order we want it to be.

> "inside" mean the ranks in this group are more likely to be adjacent
>
> "outside" means the ranks in this group are more likely to be at different GPU nodes.

