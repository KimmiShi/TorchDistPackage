import torch
from torch import nn
import torch.nn.functional as F

from torchdistpackage import get_model_profile

class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        # return hidden_states #* self.gelu(gate)
        # import pdb;pdb.set_trace()
        # return torch.matmul(hidden_states, gate.transpose(-1,-2))
        # return hidden_states.matmul(gate.transpose(-1,-2))
        gate=gate.transpose(-1,-2)
        out=hidden_states@gate
        return out


def test_flops_ds():
    model = GEGLU(1280, 5120).cuda().half()
    inp = torch.rand(32, 1280, 1280).cuda().half()

    from deepspeed.profiling.flops_profiler import get_model_profile
    flops, macs, params = get_model_profile(model=model,
                                    input_shape=(32, 1280, 1280),
                                    print_profile=True,
                                    detailed=True,
                                    warm_up=3,
    )


def test_get_profile_ffn():
    model = GEGLU(1280, 5120).cuda().half()
    inp = torch.rand(32, 1280, 1280).cuda().half()

    get_model_profile(model, (inp,), sort=False)

def test_get_profile_resnet():
    import torchvision.models as models

    model = models.resnet18().cuda().half()
    inp = torch.rand(32, 3, 224, 224).cuda().half()

    get_model_profile(model, (inp,), sort=False)

def test_get_profile_resnet_sorted():
    import torchvision.models as models

    model = models.resnet18().cuda().half()
    inp = torch.rand(32, 3, 224, 224).cuda().half()

    get_model_profile(model, (inp,), sort=True, min_mem=5)

# test_flops_ds()

# test_get_profile_ffn()

# test_get_profile_resnet()

test_get_profile_resnet_sorted()