from functools import partial

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
    def __init__(self, fin, fout, bias, tp_group=None):
        super(ColParallelLinear, self).__init__()
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

    def init_weight_from(self, fullwt):
        cur_rank = torch.distributed.get_rank(get_tp_group())
        start_ind = cur_rank*self.fout
        end_ind = (cur_rank+1)*self.fout
        slice = fullwt[:,start_ind:end_ind]
        with torch.no_grad():
            self.linear.weight.copy_(slice)

class RowParallelLinear(nn.Module):
    def __init__(self, fin, fout, bias, tp_group=None):
        super(RowParallelLinear, self).__init__()
        self.tp_world_size = torch.distributed.get_world_size(tp_group)
        assert fout%self.tp_world_size==0
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

    def init_weight_from(self, fullwt):
        cur_rank = torch.distributed.get_rank(get_tp_group())
        start_ind = cur_rank*self.fin
        end_ind = (cur_rank+1)*self.fin
        slice = fullwt[start_ind:end_ind]
        with torch.no_grad():
            self.linear.weight.copy_(slice)



class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

        From timm, but modified to run with tensor parallel and sequence parallel.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            tp_group = None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = bias

        set_tp_group(tp_group)
        self.fc1 = mylinear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = mylinear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class TpMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

        From timm, but modified to run with tensor parallel and sequence parallel.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            tp_group = None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = bias

        set_tp_group(tp_group)
        self.fc1 = ColParallelLinear(in_features, hidden_features, bias=bias, tp_group=tp_group)
        self.act = act_layer()
        self.fc2 = RowParallelLinear(hidden_features, out_features, bias=bias, tp_group=tp_group)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x