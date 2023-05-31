# Pipeline Usage

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
from torchdistpackage.parallel.clip_grad_parallel import clip_grad_norm_

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

        # this line is not necessary, forward_backward will do it.
        optimizer.zero_grad()

        # an example of customized backward_fn
        def bwd_fn(output_obj, output_obj_grad):
            if output_obj_grad is None:
                model.backward(output_obj)
            else:
                model.backward_grad(output_obj, output_obj_grad)

        def fwd_fn(inputs):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                if tpc.is_first_in_pipeline_group():
                    out = model(inputs)
                    return out
                else:
                    out = model(inputs[0])
                    loss = criterion(samples, out, inputs[1])
                    # deal with loss ...
                    return loss

        num_microbatches=4
        forward_backward(
            optimizer,
            fwd_fn,
            None,   # for simple backward func like loss.backward, we can pass None
            inputs,
            num_microbatches=num_microbatches,
            forward_only=False,
            dtype=torch.bfloat16,
            scatter_gather_tensors=False)

        total_norm = clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

```

# Example: Train Resnet -- Evaluation

```py
for images, target in data_loader:
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    dist.broadcast(target, tpc.get_ranks_in_group('pipe')[0], tpc.get_group('pipe'))

    inputs = []
    if tpc.is_first_in_pipeline_group():
        inputs = images

    def eval_fwd(img):
        with torch.cuda.amp.autocast():
            return model(img)

    output = forward_backward(
        None,
        eval_fwd,
        None,
        inputs,
        num_microbatches=1,
        forward_only=True,
        dtype=torch.float16,
        scatter_gather_tensors=False)

    if tpc.is_last_in_pipeline_group():
        # compute output
        with torch.cuda.amp.autocast():
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

```

