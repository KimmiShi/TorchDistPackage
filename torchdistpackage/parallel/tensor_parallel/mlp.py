import torch
from torch import nn as nn

# from torchdistpackage.parallel import *
from .tp_utils import get_tp_group, set_tp_group, TpLinear, RowParallelLinear, ColParallelLinear, \
    gather_from_sequence_parallel_region

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
        self.fc1 = TpLinear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = TpLinear(hidden_features, out_features, bias=bias)
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
            sequence_parallel=False
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = bias
        self.sequence_parallel = sequence_parallel

        set_tp_group(tp_group)
        self.fc1 = ColParallelLinear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = RowParallelLinear(hidden_features, out_features, bias=bias, sequence_parallel=sequence_parallel)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        if self.sequence_parallel:
            # assume input tensor is sequence parallel
            x = gather_from_sequence_parallel_region(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x