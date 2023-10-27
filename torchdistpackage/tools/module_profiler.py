import torch
import time

def get_dt_size(dtype):
    if dtype==torch.float:
        return 4
    elif dtype==torch.float16 or dtype==torch.bfloat16:
        return 2
    elif dtype==torch.int8:
        return 8
    else:
        import pdb;pdb.set_trace()
        raise NotImplementedError

def count_tensor_size(output):
    if isinstance(output, torch.Tensor):
        return output.numel()*get_dt_size(output.dtype)
    elif isinstance(output, (list, tuple)):
        return sum([count_tensor_size(t) for t in output])
    else:
        raise NotImplementedError

def output_same_as_input(output, args):
    if isinstance(output, torch.Tensor):
        if id(output)==id(args):
            return True
        if isinstance(args, (tuple, list)):
            return len(args)==1 and id(output)==id(args[0])
    else:
        return False


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

def register_profile_hooks(model):
    infos={}
    for name, module in model.named_modules():
        if name=='':
            name = 'root'
        module.register_forward_hook(fwd_hook_wrapper(name, infos))
        module.register_forward_pre_hook(fwd_pre_hook_wrapper(name, infos))
    return infos


def divide_by_layer(infos):
    levels_map = {}

    def get_level(name):
        return name.count('.')

    for k, v in infos.items():
        if k=='root':
            pass
        else:
            level = get_level(k)
            if level not in levels_map:
                levels_map[level] = {}
            levels_map[level][k] =v

    return levels_map

def sort_mem_time_ratio(infos, topn=20):
    levels = divide_by_layer(infos)
    for l, metas in levels.items():
        print("level:", l)
        mem_time_ratio = {}
        for name, time_mem in metas.items():
            ratio = time_mem['fwd_mem']/time_mem['fwd_time']
            mem_time_ratio[name] = round(ratio, 3)
        sorted_list = sorted(mem_time_ratio.items(), key=lambda x:x[1], reverse=True)
        print(sorted_list[:topn])



report_prof = sort_mem_time_ratio

def test():
    from torchvision.models import resnet18
    class FFNNET(torch.nn.Module):
        def __init__(self, indim, outdim):
            super().__init__()
            hidden_dim=4*indim
            LoRACompatibleLinear=torch.nn.Linear
            self.fc1 = LoRACompatibleLinear(indim, hidden_dim)
            self.fc2 = LoRACompatibleLinear(hidden_dim, outdim)

            self.indim = indim
            self.outdim = outdim
            self.hidden_dim = hidden_dim

        # def set_lora(self):
        #     self.fc1.requires_grad_(False)
        #     self.fc2.requires_grad_(False)
        #     self.fc1_lora = LoRALinearLayer(self.indim, self.hidden_dim, device='cuda')
        #     self.fc2_lora = LoRALinearLayer(self.hidden_dim, self.outdim, device='cuda')

        #     self.fc1.set_lora_layer(self.fc1_lora)
        #     self.fc2.set_lora_layer(self.fc2_lora)

        def forward(self, x):
            x =self.fc1(x)
            x=self.fc2(x)
            return x

    model = resnet18().cuda()
    input=torch.rand(64,3,224,224, device='cuda')

    model(input)
    model(input)

    infos = register_profile_hooks(model)

    out = model(input)

    sort_mem_time_ratio(infos)