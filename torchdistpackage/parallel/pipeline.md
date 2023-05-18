# Example: partition a Resnet into stages

```py
    layer_list = [
        'conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool',
        lambda x: torch.flatten(x, 1), 'fc'
    ]
    flat_model = flatten_model(model, layer_list , return_list=True)
    md_list = partition_uniform(flat_model)
    model = torch.nn.Sequential(*md_list)

    model = DDP(model)
```

# Example: Train Resnet

```py
from torchdistpackage.parallel.pipeline_sched import forward_backward, forward_eval

    # adapted from DeiT
    for samples, targets in data_loader:
        dist.broadcast(samples, tpc.get_ranks_in_group('pipe')[0], tpc.get_group('pipe'))
        dist.broadcast(targets, tpc.get_ranks_in_group('pipe')[0], tpc.get_group('pipe'))

        loss=torch.rand(1).cuda()

        inputs = []
        if tpc.is_first_in_pipeline_group():
            inputs.append(samples)
        else:
            inputs.append(targets)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        def bwd_fn(loss):
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        def fwd_fn(inputs):
            with torch.cuda.amp.autocast():
                if tpc.is_first_in_pipeline_group():
                    out = model(inputs)
                    return out
                else:
                    out = model(inputs[0])
                    loss = criterion(samples, out, inputs[1])
                    # dealwith loss
                    # ...
                    return loss

        num_microbatches=1
        out = forward_backward(
            optimizer,
            fwd_fn,
            bwd_fn,
            inputs,
            num_microbatches=num_microbatches,
            forward_only=False,
            dtype=torch.float16,
            scatter_gather_tensors=False)

```