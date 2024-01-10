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
    ret=True
    for name,param in model.named_parameters():
        if not check_tensor_inf_nan(param):
            print("model param:", name, " contains nan/inf!")
            # import pdb;pdb.set_trace()
            ret = False
    return ret

def print_tensor_info(module_name, output, msg):
    if isinstance(output, torch.Tensor):
        print(module_name, msg, output.max())
    elif isinstance(output, tuple) or isinstance(output, list):
        for i, t in enumerate(output):
            print_tensor_info(module_name, t, msg)
            # print(module_name, f"output_{i}_max", t.max())

@torch.no_grad
def get_max(output):
    max=0
    if isinstance(output, torch.Tensor):
        return output.max().item()
    elif isinstance(output, tuple) or isinstance(output, list):
        for i, t in enumerate(output):
            if get_max(t)>max:
                max = get_max(t)
    return max

def find_big_change(module_name, args, output):
    out_max = get_max(output)
    in_max = get_max(args)
    if in_max!=0 and out_max / in_max >= 2 and out_max>1000:
        print(module_name, out_max, in_max)

# register_forward_hook
@torch.no_grad()
def fwd_hook_wrapper(module_name=''):
    def check_value(module, args, output):
        if not check_tensors(args):
            print(module_name, "input contains nan/inf!")
        if not check_tensors(output):
            import pdb;pdb.set_trace()
            print(module_name, "output contains nan/inf!")

        # if 'conv_shortcut' in module_name:
        # print_tensor_info(module_name, args, "input_max")
        # print_tensor_info(module_name, output, "output_max")
        find_big_change(module_name, args, output)

    return check_value


# model.register_full_backward_hook
def bwd_hook_wrapper(module_name=''):
    def check_value(module, grad_input, grad_output):
        if not check_tensors(grad_output):
            print(module_name, "grad_output contains nan/inf!")
            import pdb;pdb.set_trace()
        if not check_tensors(grad_input):
            print(module_name, "grad_input contains nan/inf!")
            # import pdb;pdb.set_trace()
    return check_value

# detect param update nan

def register_debug_nan_hooks(model):
    fwd_hooks = {}
    bwd_hooks = {}
    for name, module in model.named_modules():
        fwd_hooks[name] = module.register_forward_hook(fwd_hook_wrapper(name))
        # bwd_hooks[name] = module.register_full_backward_hook(bwd_hook_wrapper(name))
    return fwd_hooks, bwd_hooks

