import torch
from torch import nn as nn
from torch.nn.parameter import Parameter

import torch.distributed as dist

TP_GROUP=None
def get_tp_group():
    global TP_GROUP
    return TP_GROUP

def set_tp_group(group):
    global TP_GROUP
    TP_GROUP=group

def get_tensor_model_parallel_world_size():
    return dist.get_world_size(get_tp_group())

class mylinear(nn.Module):
    def __init__(self, fin, fout, bias=True):
        super(mylinear, self).__init__()
        self.weight = Parameter(torch.rand((fin, fout)))
        self.bias =None
        if bias:
            self.bias = Parameter(torch.zeros(fout))

    def forward(self, x):
        out = torch.matmul(x, self.weight)
        if self.bias is not None:
            out +=self.bias
        return out

class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        torch.distributed.all_reduce(input_, group=get_tp_group())
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def _reduce_scatter_along_first_dim(input_):
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert dim_size[0] % world_size == 0, \
        "First dimension of the tensor should be divisible by tensor parallel size"

    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=torch.cuda.current_device())
    torch.distributed._reduce_scatter_base(output, input_.contiguous(),
                                           group=get_tp_group())
    return output

def _gather_along_first_dim(input_):
    """Gather tensors and concatinate along the first dimension."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=torch.cuda.current_device())
    torch.distributed._all_gather_base(output, input_.contiguous(),
                                       group=get_tp_group())

    return output
def _split_along_first_dim(input_):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size()[0]
    assert dim_size % world_size == 0, \
        "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    rank = torch.distributed.get_rank(get_tp_group())
    dim_offset = rank * local_dim_size

    output = input_[dim_offset:dim_offset+local_dim_size].contiguous()

    return output

class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, tensor_parallel_output_grad=True):
        return _gather_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_, tensor_parallel_output_grad=True):
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        return _gather_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad

        # If the computation graph after the gather operation is
        # in the tensor parallel mode, output gradients need to reduce
        # scattered and whereas if the computation is duplicated,
        # output gradients need to be scattered.
        if tensor_parallel_output_grad:
            return _reduce_scatter_along_first_dim(grad_output), None
        else:
            return _split_along_first_dim(grad_output), None

def gather_from_sequence_parallel_region(input_, tensor_parallel_output_grad=True):
    return _GatherFromSequenceParallelRegion.apply(input_, tensor_parallel_output_grad)
def reduce_scatter_to_sequence_parallel_region(input_):
    return _ReduceScatterToSequenceParallelRegion.apply(input_)

class ColParallelLinear(nn.Module):
    def __init__(self, fin, fout, bias=True):
        super(ColParallelLinear, self).__init__()
        tp_group=get_tp_group()
        self.tp_world_size = torch.distributed.get_world_size(tp_group)
        assert fout%self.tp_world_size==0
        self.fout = int(fout/self.tp_world_size)

        self.linear = mylinear(fin, self.fout, bias)


    def forward(self, x):
        """
        1. automatically split fout
        2. do linear compute
        3. the output is split , no communication
        """
        out = self.linear(x)
        return out

    def init_weight_from_full(self, fullwt):
        cur_rank = torch.distributed.get_rank(get_tp_group())
        start_ind = cur_rank*self.fout
        end_ind = (cur_rank+1)*self.fout
        slice = fullwt[:,start_ind:end_ind]
        with torch.no_grad():
            self.linear.weight.copy_(slice)

    def init_weight_from_full_attn(self, fullwt):
        cur_rank = torch.distributed.get_rank(get_tp_group())
        ws = torch.distributed.get_world_size(get_tp_group())
        dim=fullwt.shape[0]
        dim3=fullwt.shape[1]
        fullwts = fullwt.split(dim3//3, dim=-1) # (q,k,v)
        splits = []
        for wt in fullwts:
            splits.append(wt.split(wt.shape[-1]//ws, dim=-1)[cur_rank])

        cat_full = torch.cat(splits, dim=-1)

        with torch.no_grad():
            # org_shape = self.linear.weight.shape
            # self.linear.weight = self.linear.weight.reshape(dim, nh//ws, dim3//nh)
            self.linear.weight.copy_(cat_full)

    @property
    def weight(self):
        return self.linear.weight

class RowParallelLinear(nn.Module):
    def __init__(self, fin, fout, bias=True, sequence_parallel=False):
        super(RowParallelLinear, self).__init__()
        tp_group=get_tp_group()
        self.tp_world_size = torch.distributed.get_world_size(tp_group)
        assert fin%self.tp_world_size==0
        self.fin = int(fin/self.tp_world_size)
        self.linear = mylinear(self.fin, fout, bias)
        self.sequence_parallel = sequence_parallel


    def forward(self, x):
        """
        1. automatically split fout
        2. do linear compute
        3. the output is allreduced
        """
        out = self.linear(x)
        if not self.sequence_parallel:
            out = _ReduceFromModelParallelRegion.apply(out)
        else:
            out = reduce_scatter_to_sequence_parallel_region(out)
        return out

    def init_weight_from_full(self, fullwt):
        cur_rank = torch.distributed.get_rank(get_tp_group())
        start_ind = cur_rank*self.fin
        end_ind = (cur_rank+1)*self.fin
        slice = fullwt[start_ind:end_ind]
        with torch.no_grad():
            self.linear.weight.copy_(slice)

    @property
    def weight(self):
        return self.linear.weight