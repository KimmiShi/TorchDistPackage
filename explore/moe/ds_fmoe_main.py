# Use this version of deepspeed:
# https://github.com/KimmiShi/DeepSpeed/tree/fmoe_v0.9.0

# Use this version of fastmoe:
# https://github.com/KimmiShi/fastmoe

from timm.models import create_model
import deepspeed


def main():
    model = create_model(...)
    args.ep = 4


    # replace vit.mlp with MoE layer
    from deepspeed.utils import groups
    from deepspeed.comm.comm import init_distributed
    init_distributed(dist_backend='nccl')
    groups._create_expert_and_data_parallel(args.ep)
    for block in model.encoder.blocks:
        original_mlp = block.mlp
        feat_in = original_mlp.fc1.weight.shape[1]
        block.mlp = deepspeed.moe.layer.MoE(hidden_size=feat_in, expert=block.mlp, num_experts=args.ep,
                                            ep_size=args.ep, use_fmoe=True)

    def create_moe_param_groups(model):
        from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
        parameters = {'params': [p for p in model.parameters()], 'name': 'parameters'}
        return split_params_into_different_moe_groups_for_optimizer(parameters)
    parameters = create_moe_param_groups(model)
    optimizer = create_optimizer_v2(parameters, **optimizer_kwargs(cfg=args))
    model, optimizer, _, _ = deepspeed.initialize(args=args,
                                                    model=model,
                                                    optimizer=optimizer)
