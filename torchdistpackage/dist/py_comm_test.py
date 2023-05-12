import os

import torch
import torch.distributed as dist
import time
import functools

from torchdistpackage import setup_distributed_slurm

# reference: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
# algbw = Size/time
# bus_bw = algbw * fraction * (n-1)/n
mode_2_frac = dict(
    all_reduce = 2,
    all_gather = 1,
    reduce_scatter = 1,
)

def test_collection(ele_num_total, mode = 'all_reduce', group=None):
    comm_op = eval(f"dist.{mode}")
    ws = dist.get_world_size(group)

    ele_num = ele_num_total
    if mode == "all_gather":
        ele_num = int(ele_num_total // dist.get_world_size(group))
    tensor = torch.randn(ele_num).half().cuda()
    if mode == "all_gather":
        tensor_list = [torch.randn(ele_num).half().cuda() for _ in range(ws)]
        # import pdb;pdb.set_trace()
        comm_op = functools.partial(comm_op, tensor_list)

    comm_op(tensor, group=group)
    comm_op(tensor, group=group)
    dist.barrier()
    torch.cuda.synchronize()
    bw=0
    frac = mode_2_frac[mode]
    num_repeat = 1

    dist.barrier()
    torch.cuda.synchronize()
    beg=time.perf_counter()
    for _ in range(num_repeat):
        comm_op(tensor, group=group)
    dist.barrier()
    torch.cuda.synchronize()

    time_avg = (time.perf_counter()-beg)/num_repeat
    algbw = (ele_num_total*2/1e9)/time_avg # GB/s
    bw = algbw * frac * ((ws-1)/ws)
    bw = round(bw, 3)
    time_avg = round(time_avg, 3)
    if dist.get_rank()==0:
        print(f"{mode} repeat={num_repeat}, bandwidth:{bw} GB/s time_avg:{time_avg} s, numel={tensor.numel()}")

    torch.cuda.synchronize()
    return bw,time_avg


def test_all2all_balanced(ele_num, group=None):
    tensor = torch.ones(ele_num).cuda() * dist.get_rank()

    output = torch.empty_like(tensor)
    dist.all_to_all_single(output, tensor, group=group)

    dist.barrier()
    torch.cuda.synchronize()

    num_repeat = 1
    beg=time.perf_counter()
    for _ in range(num_repeat):
        dist.all_to_all_single(output, tensor, group=group)
    dist.barrier()
    torch.cuda.synchronize()
    time_avg = (time.perf_counter()-beg)/num_repeat

    if dist.get_rank()==0:
        print(f"all2all_balanced repeat={num_repeat}, time_avg:{time_avg} s, numel={tensor.numel()}")


if __name__=="__main__":
    setup_distributed_slurm()

    test_collection(1801705472*2, mode='all_gather')
