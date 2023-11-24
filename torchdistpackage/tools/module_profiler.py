import torch

import time
import re



""" Usage:

from torchdistpackage import report_prof, register_profile_hooks

infos = register_profile_hooks(model)

report_prof(infos, topn=-1, max_depth=4, min_mem=50)

"""

def get_dt_size(dtype):
    if dtype==torch.float:
        return 4
    elif dtype==torch.float16 or dtype==torch.bfloat16:
        return 2
    elif dtype==torch.int8:
        return 8
    elif dtype in [torch.int64,]:
        return 8
    else:
        import pdb;pdb.set_trace()
        return 0
        raise NotImplementedError

def count_tensor_size(output):
    if isinstance(output, torch.Tensor):
        return output.numel()*get_dt_size(output.dtype)
    elif isinstance(output, (list, tuple)):
        return sum([count_tensor_size(t) for t in output])

    else:
        # import pdb;pdb.set_trace()
        return 0
        raise NotImplementedError

def output_same_as_input(output, args):
    if isinstance(output, torch.Tensor):
        if id(output)==id(args):
            return True
        if isinstance(args, (tuple, list)):
            return len(args)==1 and id(output)==id(args[0])
    else:
        return False

def get_level(name):
    if name=="root":
        return 0
    dot_cnt = name.count('.')
    postfix_cnt = len(re.findall('\.[0-9]\.', name)) + len(re.findall('\.[0-9]$', name))
    return dot_cnt-postfix_cnt+1

# register_forward_pre_hook
@torch.no_grad()
def fwd_pre_hook_wrapper(module_name='root', infos={}):
    def begin_fwd_profile(module, args):
        torch.cuda.synchronize()
        beg = time.perf_counter()
        infos[module_name] = {}
        infos[module_name]['fwd_beg'] = beg
        infos[module_name]['fwd_beg_mem'] = torch.cuda.memory_allocated()

    return begin_fwd_profile

# register_forward_hook
@torch.no_grad()
def fwd_hook_wrapper(module_name='root', infos={}):
    def end_fwd_profile(module, args, output):
        torch.cuda.synchronize()
        end = time.perf_counter()
        mem = torch.cuda.memory_allocated()

        infos[module_name]['fwd_time'] = (end-infos[module_name].pop('fwd_beg'))*1000
        infos[module_name]['fwd_mem'] = (mem-infos[module_name].pop('fwd_beg_mem'))/1e6
        if not output_same_as_input(output, args):
            activation_mem_diff = count_tensor_size(output) - count_tensor_size(args)

            infos[module_name]['fwd_mem'] -= activation_mem_diff/1e6

    return end_fwd_profile

def register_profile_hooks(model,infos={}):
    for name, module in model.named_modules():
        if name=='':
            name = 'root'
        module.register_forward_hook(fwd_hook_wrapper(name, infos))
        module.register_forward_pre_hook(fwd_pre_hook_wrapper(name, infos))
    return infos


def divide_by_layer(infos):
    levels_map = {}
    for k, v in infos.items():
        if k=='root':
            level = 0
            levels_map[level] = {k:v}
        elif isinstance(k, str):
            try:
                level = get_level(k)
            except:
                import pdb;pdb.set_trace()
                pass
            if level not in levels_map:
                levels_map[level] = {}
            levels_map[level][k] =v
        else:
            import pdb;pdb.set_trace()
            pass

    return levels_map

def sort_mem_time_ratio(infos, topn=20, max_depth=5, min_mem=50, sort=True):
    levels = divide_by_layer(infos)
    for l, metas in levels.items():
        if l > max_depth:
            break
        print("\nlevel:", l)
        if not sort:
            for k, time_mem in metas.items():
                print(f"{k}: MEM: {round(metas[k]['fwd_mem'])} MB; Time: {round(metas[k]['fwd_time'], 4)} ms")
        else:
            mem_time_ratio = {}
            for name, time_mem in metas.items():
                if time_mem['fwd_mem']<min_mem:
                    continue
                ratio = time_mem['fwd_mem']/time_mem['fwd_time']
                ratio = ratio
                mem_time_ratio[name] = round(ratio, 3)
            sorted_list = sorted(mem_time_ratio.items(), key=lambda x:x[1], reverse=True)
            show_infos = []
            for k, ratio in sorted_list[:topn]:
                show_infos.append(f"{k}: MEM: {round(metas[k]['fwd_mem'])} MB; Time: {round(metas[k]['fwd_time'], 4)} ms")
            for line in show_infos:
                print(line)



report_prof = sort_mem_time_ratio

def get_model_profile(model, args: tuple, kwargs={}, sort=True, topn=20, max_depth=5, min_mem=50):
    """run the model and print the TIME & GPU MEM consumption

    Args:
        model (torch.nn.Module): the model to run
        args (tuple): the model input tensors
        kwargs (dict, optional): model input kwargs. Defaults to {}.
        sort (bool, optional): sort the print sequence by mem/time or not. If False, ALL modules' info will be printed. Defaults to True.
        topn (int, optional): how many module to print if sort is True. Defaults to 20.
        max_depth (int, optional): the max depth of module to print. Defaults to 5.
        min_mem (int, optional): memory consumption low than min_mem MB will be ignored. Defaults to 50.

    Returns:
        dict: all the modules and their memory and time statistics
    """
    # wamup
    _ = model(*args, **kwargs)

    torch.cuda.synchronize()

    infos = register_profile_hooks(model)

    out = model(*args, **kwargs)

    report_prof(infos, sort=sort, topn=topn, max_depth=max_depth, min_mem=min_mem)

    return infos