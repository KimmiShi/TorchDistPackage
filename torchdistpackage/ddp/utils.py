import torch
import torch.distributed as dist
from torch.distributed import ReduceOp


glb_reduce_stream = torch.cuda.Stream()

def reduce_grad(grad, group, reduce_op):
    global glb_reduce_stream
    glb_reduce_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(glb_reduce_stream):
        dist.all_reduce(grad, group=group, async_op=False, op=reduce_op)

def register_reduce_hook_for_param(p, group, reduce_op=ReduceOp.AVG):
    if p.requires_grad:
        if dist.get_world_size(group) <= 1:
            return
        p_tmp = p.expand_as(p)
        grad_acc = p_tmp.grad_fn.next_functions[0][0]
        grad_acc.register_hook(reduce_grad(p.grad.data, group, reduce_op))
