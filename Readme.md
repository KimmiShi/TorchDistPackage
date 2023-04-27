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

- [slurm init](./slurm_dist_init/)

