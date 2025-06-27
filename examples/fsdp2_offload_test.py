import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from torch import distributed as dist
import os

def setup_distributed(timeout=None) -> None:

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    except KeyError as e:
        raise RuntimeError(f"Could not find {e} in the torch environment")

    # initialize the default process group
    host = os.getenv("MASTER_ADDR", "localhost")
    port = os.getenv("MASTER_PORT", "2222")
    init_method = f"tcp://{host}:{port}"
    if rank==0:
        print(f"Init Distributed Env, init_method:{init_method}, rank:{rank}, world_size:{world_size}")
    # TODO: unify the init_process_group for both vllm and sglang when stable version finished

    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend='cpu:gloo,cuda:nccl',
        init_method=init_method,
        timeout=timeout,
    )
def fsdp_wrap(model_to_wrap):
    from transformers.trainer_pt_utils import get_module_class_from_name
    try:
        from torch.distributed.fsdp import (
            fully_shard,
            register_fsdp_forward_method,
            MixedPrecisionPolicy,
            CPUOffloadPolicy,
            OffloadPolicy,
            FSDPModule,
        )
    except ImportError:
        from torch.distributed._composable.fsdp import (
            fully_shard,
            register_fsdp_forward_method,
            MixedPrecisionPolicy,
            CPUOffloadPolicy,
            OffloadPolicy,
            FSDPModule,
        )
    fsdp_kwargs = {
            "reshard_after_forward": True,
        }
    transformer_cls_names_to_wrap = getattr(model_to_wrap, "_no_split_modules", [])
    transformer_cls_to_wrap = list()
    vit_transformer_cls = list()
    for layer_class in transformer_cls_names_to_wrap:
        transformer_cls = get_module_class_from_name(model_to_wrap, layer_class)
        if transformer_cls is not None:
            transformer_cls_to_wrap.append(transformer_cls)

    for cls_to_wrap in transformer_cls_to_wrap:
        for module in model_to_wrap.modules():
            if isinstance(module, cls_to_wrap):

                fully_shard(module, **fsdp_kwargs)

    for name, module in model_to_wrap.named_modules():
        if 'lm_head' in name:
            fully_shard(module, **fsdp_kwargs)

    fully_shard(model_to_wrap, **fsdp_kwargs)

    return model_to_wrap

@torch.no_grad()
def offload_model(models, empty_cache: bool = True):
    def offload_single(model):
        if not isinstance(model, torch.nn.Module):
            return
        # model.to(torch.device("cpu"))
        for param in model.parameters():
            param.data = param.data.to(torch.device("cpu"), non_blocking=True)
            # param = param.to(torch.device("cpu"), non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(torch.device("cpu"), non_blocking=True)

    if isinstance(models, (list, tuple)):
        for model in models:
            offload_single(model)
    else:
        offload_single(models)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

@torch.no_grad()
def reload_model(models):
    device = torch.cuda.current_device()
    def reload_single(model):
        if not isinstance(model, torch.nn.Module):
            return
        # model.to(device)
        for param in model.parameters():
            param.data = param.data.to(device, non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(device, non_blocking=True)

    if isinstance(models, (list, tuple)):
        for model in models:
            reload_single(model)
    else:
        reload_single(models)


def report_memory(prefix=""):
    usable, total = torch.cuda.mem_get_info()
    used = round((total - usable) / 1e9, 2)
    print(f"MEMORY STATUS: {prefix}, USED={used} GB," f"ALLOCATED={torch.cuda.memory_allocated() / 1e9:.2f} GB")


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.ModuleList([torch.nn.Linear(8192, 8192) for _ in range(40)])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

setup_distributed()

pretrain_name_or_path=""

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrain_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
            )
# torch.cuda.memory._record_memory_history(max_entries=100000)

# model = DummyModel()
model = fsdp_wrap(model)

report_memory("before .cuda")
model.cuda()
report_memory("after .cuda")

offload_model(model)
# offload_model_to_cpu(model)
report_memory("after offload model")

# torch.cuda.memory._dump_snapshot('test_offload_mem')

if dist.get_rank()==0:
    print(torch.cuda.memory_summary())



