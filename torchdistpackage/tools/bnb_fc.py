import torch
import functools

from .module_replace import replace_all_module

def if_replace_linear(name, module):
    return isinstance(module, (torch.nn.Linear))

def get_new_module(name, module, bnb_kwargs={}):
    from bitsandbytes.nn import Linear8bitLt

    old_shape = module.weight.shape
    indim = old_shape[1]
    outdim = old_shape[0]
    o_dtype = module.weight.dtype
    o_device = module.weight.dtype
    new_linear = Linear8bitLt(indim, outdim, bias=module.bias is not None, **bnb_kwargs).to(o_device).to(o_dtype)
    new_linear.weight.data.copy_(module.weight)
    if module.bias is not None:
        new_linear.bias.data.copy_(module.bias)
    return new_linear

def replace_linear_by_bnb(model, bnb_kwargs={'has_fp16_weights':False,'threshold':6.0}):
    replace_all_module(model, if_replace_linear, functools.partial(get_new_module, bnb_kwargs=bnb_kwargs))