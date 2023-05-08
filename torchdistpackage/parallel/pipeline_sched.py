import torch
from torchdistpackage import tpc

def _forward_step_in_forward_backward(self, input_obj_from_prev, ind, micro_bs, fwd_fn, extra_inputs = []):

    cur_inputs = []
    if input_obj_from_prev is not None:
        if isinstance(input_obj_from_prev, torch.Tensor):
            cur_inputs.append(input_obj_from_prev)
        elif isinstance(input_obj_from_prev, list) or isinstance(input_obj_from_prev, tuple):
            cur_inputs.append(*input_obj_from_prev)

    for inp in extra_inputs:
        cur_inputs.append(inp[ind * micro_bs : (ind + 1) * micro_bs])


    # inputs made of two parts: the prev stage output, and given mini-batch ( that should be split into micro-batches)
    fwd_fn(*cur_inputs)


def _backward_step_in_forward_backward(self, input_obj, output_obj, output_obj_grad, backward_fn):
    # Retain the grad on the input_obj.
    if input_obj is not None:
        if isinstance(input_obj, torch.Tensor):
            input_obj.retain_grad()
        else:
            for in_tensor in input_obj:
                if in_tensor is not None:
                    in_tensor.retain_grad()
    if output_obj_grad is None:
        backward_fn(output_obj)
    else:
        torch.autograd.backward(tensors=output_obj, grad_tensors=output_obj_grad)

    # Collect the grad of the input_obj.
    input_obj_grad = None
    if input_obj is not None:
        if isinstance(input_obj, torch.Tensor):
            input_obj_grad = input_obj.grad
        else:
            input_obj_grad = []
            for in_tensor in input_obj:
                input_obj_grad.append(in_tensor.grad)

    return input_obj_grad

def forward_backward(self, optimizer, fwd_fn, bwd_fn, num_microbatches=1, forward_only = False, dtype=torch.bfloat16, scatter_gather_tensors=False):

    num_warmup_microbatches = \
        (tpc.get_group_size('pipe')
            - tpc.get_group_rank('pipe') - 1)

    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Input, output tensors only need to be saved when doing backward passes
    input_objs = None
    output_objs = None
    if not forward_only:
        input_objs = []
        output_objs = []

    # Used for tensor meta information communication
    ft_shapes = None
    bt_shapes = None
    fs_checker = True
    optimizer.zero_grad()

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        if not tpc.is_first_in_pipeline_group:
            ft_shapes = comm.recv_obj_meta(ft_shapes)
        input_obj = comm.recv_forward(ft_shapes, dtype=dtype, scatter_gather_tensors=scatter_gather_tensors)

        output_obj = _forward_step_in_forward_backward(input_obj, i)
        if not tpc.is_last_in_pipeline_group():
            if isinstance(output_obj, torch.Tensor):
                bt_shapes = output_obj.shape
            else:
                bt_shapes = []
                for out_tensor in output_obj:
                    bt_shapes.append(out_tensor.shape)
            fs_checker = comm.send_obj_meta(output_obj, fs_checker)
        comm.send_forward(output_obj, scatter_gather_tensors=scatter_gather_tensors)

        if not forward_only:
            input_objs.append(input_obj)
            output_objs.append(output_obj)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        if not tpc.is_first_in_pipeline_group():
            ft_shapes = comm.recv_obj_meta(ft_shapes)
        input_obj = comm.recv_forward(ft_shapes, dtype=dtype, scatter_gather_tensors=scatter_gather_tensors)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = (i == (num_microbatches_remaining - 1))

        output_obj = _forward_step_in_forward_backward(input_obj, i+num_warmup_microbatches)
        if forward_only:
            comm.send_forward(output_obj, scatter_gather_tensors=scatter_gather_tensors)

            if not last_iteration:
                input_obj = comm.recv_forward(ft_shapes, dtype=dtype, scatter_gather_tensors=scatter_gather_tensors)

        else:
            output_obj_grad = comm.send_forward_recv_backward(output_obj,
                                                                bt_shapes,
                                                                dtype=dtype,
                                                                scatter_gather_tensors=scatter_gather_tensors)

            # Add input_obj and output_obj to end of list.
            input_objs.append(input_obj)
            output_objs.append(output_obj)

            # Pop output_obj and output_obj from the start of the list for
            # the backward pass.
            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)

            input_obj_grad = _backward_step_in_forward_backward(input_obj, output_obj, output_obj_grad)

            if last_iteration:
                input_obj = None
                comm.send_backward(input_obj_grad, scatter_gather_tensors=scatter_gather_tensors)
            else:
                input_obj = comm.send_backward_recv_forward(input_obj_grad,
                                                            ft_shapes,
                                                            dtype=dtype,
                                                            scatter_gather_tensors=scatter_gather_tensors)

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):
            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)
            output_obj_grad = comm.recv_backward(bt_shapes,
                                                    dtype=dtype,
                                                    scatter_gather_tensors=scatter_gather_tensors)

            input_obj_grad = _backward_step_in_forward_backward(input_obj, output_obj, output_obj_grad)

            comm.send_backward(input_obj_grad, scatter_gather_tensors=scatter_gather_tensors)

    return
