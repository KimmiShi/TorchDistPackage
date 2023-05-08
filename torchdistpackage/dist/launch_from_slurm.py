import os
import subprocess

import torch
import torch.distributed as dist

# TODO: this func is not exmamined
def find_free_port():
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])

def setup_distributed_slurm(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        local_rank = rank % num_gpus

        addr = subprocess.getoutput(
            f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            port = 54647
            os.environ["MASTER_PORT"] = str(port)
        else:
            port = int(os.environ["MASTER_PORT"])
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = rank % num_gpus

    dist.init_process_group(rank=rank, world_size=world_size, backend=backend)

    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    print(f"dist init done, world_size = {dist.get_world_size()}")
    return rank, world_size, port, addr