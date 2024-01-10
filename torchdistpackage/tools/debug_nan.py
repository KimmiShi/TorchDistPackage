import torch

def check_tensor_inf_nan(inp):
    if torch.isnan(inp).any() or torch.isinf(inp).any():
        return False
    return True

def check_tensors(inputs):
    if isinstance(inputs, torch.Tensor):
        return check_tensor_inf_nan(inputs)
    elif isinstance(inputs, tuple) or isinstance(inputs, list):
        ret=True
        for tmp in inputs:
            if isinstance(tmp, torch.Tensor):
                ret = check_tensor_inf_nan(tmp) and ret
        return ret
    elif hasattr(inputs, 'sample'):
        return check_tensors(inputs.sample)
    else:
        print("unhandled type:", type(inputs))
        return True


def check_model_params(model):
    for name,param in model.named_parameters():
        if not check_tensor_inf_nan(param):
            print("model param:", name, " contains nan/inf!")
            # import pdb;pdb.set_trace()
            return False

# register_forward_hook
@torch.no_grad()
def fwd_hook_wrapper(module_name=''):
    def check_value(module, args, output):
        if not check_tensors(args):
            print(module_name, "input contains nan/inf!")
        if not check_tensors(output):
            # import pdb;pdb.set_trace()
            print(module_name, "output contains nan/inf!")


    return check_value

# model.register_full_backward_hook
def bwd_hook_wrapper(module_name=''):
    def check_value(module, grad_input, grad_output):
        if not check_tensors(grad_output):
            print(module_name, "grad_output contains nan/inf!")
        if not check_tensors(grad_input):
            print(module_name, "grad_input contains nan/inf!")
            # import pdb;pdb.set_trace()
    return check_value

# detect param update nan

def register_abnormal_val_hooks(model):
    fwd_hooks = {}
    bwd_hooks = {}
    for name, module in model.named_modules():
        fwd_hooks[name] = module.register_forward_hook(fwd_hook_wrapper(name))
        bwd_hooks[name] = module.register_forward_hook(bwd_hook_wrapper(name))
    return fwd_hooks, bwd_hooks

