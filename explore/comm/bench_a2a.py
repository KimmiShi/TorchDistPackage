import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Any, Tuple
from torch import Tensor
import time
from torchdistpackage import setup_distributed, NaiveDDP, fix_rand

class AllToAll(torch.autograd.Function):
    """
    All to all communication
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        output_split_sizes=None,
        input_split_sizes=None,
        group: torch.distributed.ProcessGroup = None,
        async_op=False,
    ) -> Tensor:  # type: ignore

        ctx.input_shape = inputs.shape
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group

        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return inputs, None

        inputs = inputs.contiguous()
        out = (
            torch.empty_like(inputs)
            if output_split_sizes is None
            else inputs.new_empty(size=[sum(output_split_sizes)] + list(inputs.size()[1:]))
        )
        handle = torch.distributed.all_to_all_single(
            out,
            inputs,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=async_op,
        )

        # if async_op=False, handle will be None
        return out, handle

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor, _) -> Tuple[None, Tensor]:
        if ctx.needs_input_grad[0]:
            # Bypass the function if we are using only 1 GPU.
            world_size = torch.distributed.get_world_size(group=ctx.group)
            if world_size == 1:
                return grad_output, None, None, None, None

            grad_output = grad_output.contiguous()
            out = torch.empty(ctx.input_shape, device=grad_output.device, dtype=grad_output.dtype)
            torch.distributed.all_to_all_single(
                out,
                grad_output,
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
                group=ctx.group,
            )
            return out, None, None, None, None
        return None, None, None, None, None


def all_to_all(x, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False):
    return AllToAll.apply(x, output_split_sizes, input_split_sizes, group, async_op)

def test_all2all(group=None, s=4096,d=2048,k=8, even=True):
    inp = torch.rand((s*k, d), dtype=torch.bfloat16, device='cuda')

    for _ in range(4):
        all_to_all(inp, group=group)

    torch.cuda.synchronize()
    dist.barrier()
    t1=time.time()

    rpt=10
    for _ in range(rpt):
        all_to_all(inp, group=group)

    torch.cuda.synchronize()
    dist.barrier()

    cost = round((time.time()-t1)/rpt,6)

    algo_bw = round(inp.numel()*2/cost/1024**3,3)

    if dist.get_rank()==0:
        print("ws:",dist.get_world_size(group),s,d,k)
        print("timecost,algobw:(GB/s):", cost, algo_bw)

setup_distributed()

test_all2all(s=512,d=4096)