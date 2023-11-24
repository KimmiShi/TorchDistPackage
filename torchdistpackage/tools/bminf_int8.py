import torch

import bminf

from .module_replace import replace_all_module

def if_replace_linear(name, module):
    return isinstance(module, (torch.nn.Linear))

def get_new_module(name, module):
    new_linear = bminf.QuantizedLinear(module)
    return new_linear

def replace_linear_by_bminf(model):
    replace_all_module(model, if_replace_linear, get_new_module)