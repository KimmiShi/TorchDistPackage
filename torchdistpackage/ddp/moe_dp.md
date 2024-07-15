

```py
# 1. setup moe dp group
from torchdistpackage import tpc, setup_distributed

setup_distributed()
pp_size=2
dist_config = [('data',world_size/(2*pp_size)), ('pipe',pp_size), ('tensor',2)]
tpc.setup_process_groups(dist_config)
tpc.build_moe_groups(moe_dp_size=4)


# 2. build all moe params into a dict
moe_params = build_moe_params_for_module(model)

# 3. create moe_dp_hooks
from tpc import moe_dp_iter_step,create_moe_dp_hooks
create_moe_dp_hooks(moe_params, tpc.get_group('moe_dp'), tpc.get_ranks_in_group('moe_dp')[0])

# 4 (optional). for moe with pipeline parallel, we can set num_grad_acc_iter to do reduce only once,
# and we have to call moe_dp_iter_step every iter
moe_dp_iter_step()

```