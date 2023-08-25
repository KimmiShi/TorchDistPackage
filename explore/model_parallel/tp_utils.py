import torch
from torch import nn as nn
from torch.nn.parameter import Parameter

TP_GROUP=None
def get_tp_group():
    global TP_GROUP
    return TP_GROUP

def set_tp_group(group):
    global TP_GROUP
    TP_GROUP=group

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

class ColParallelLinear(nn.Module):
    def __init__(self, fin, fout, bias):
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

class RowParallelLinear(nn.Module):
    def __init__(self, fin, fout, bias=True):
        super(RowParallelLinear, self).__init__()
        tp_group=get_tp_group()
        self.tp_world_size = torch.distributed.get_world_size(tp_group)
        assert fin%self.tp_world_size==0
        self.fin = int(fin/self.tp_world_size)
        self.linear = mylinear(self.fin, fout, bias)


    def forward(self, x):
        """
        1. automatically split fout
        2. do linear compute
        3. the output is allreduced
        """
        out = self.linear(x)
        out = _ReduceFromModelParallelRegion.apply(out)
        return out

    def init_weight_from_full(self, fullwt):
        cur_rank = torch.distributed.get_rank(get_tp_group())
        start_ind = cur_rank*self.fin
        end_ind = (cur_rank+1)*self.fin
        slice = fullwt[start_ind:end_ind]
        with torch.no_grad():
            self.linear.weight.copy_(slice)