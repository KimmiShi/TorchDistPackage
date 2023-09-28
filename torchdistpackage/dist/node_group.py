import torch.distributed as dist

def setup_node_groups(num_per_node=8):
    """every node is build as a comm group

    Args:
        num_per_node (int, optional): num gpu per node. Defaults to 8.

    Returns:
        torch comm group: returns None if world size is illeagal to divide into nodes

    Usage example:
    ```
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, gradient_as_bucket_view=True)
        opt = ZeroRedundancyOptimizer(model.parameters(),
                                    optimizer_class=torch.optim.AdamW,
                                    process_group = setup_node_groups(),
                                    parameters_as_bucket_view=False, fused=True)
    ```

    """
    world_size = dist.get_world_size()
    if world_size % num_per_node != 0 or world_size <= num_per_node:
        return None
    num_nodes = world_size//num_per_node
    for node_ind in range(num_nodes):
        ranks_in_node = [r + node_ind*num_per_node for r in range(0, num_per_node)]
        new_group = dist.new_group(ranks_in_node)
        if dist.get_rank() in ranks_in_node:
            ret = new_group
            print(dist.get_rank(), ranks_in_node)
    return ret

