# Adapted from https://github.com/hpcaitech/ColossalAI

from typing import List, Tuple, Union
import torch
import torch.distributed as dist

from torchdistpackage import tpc
from functools import reduce
import operator


TensorShape = Union[torch.Size, List[int], Tuple[int]]


def get_current_device() -> torch.device:
    """
    Returns currently selected device (gpu/cpu).
    If cuda available, return gpu, otherwise return cpu.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        return torch.device("cpu")


def send_meta_helper(obj, next_rank, tensor_kwargs):
    send_shape = torch.tensor(obj.size(), **tensor_kwargs)
    send_ndims = torch.tensor(len(obj.size()), **tensor_kwargs)
    dist.send(send_ndims, next_rank)
    dist.send(send_shape, next_rank)


def send_obj_meta(obj, need_meta=True, next_rank=None) -> bool:
    """Sends obj meta information before sending a specific obj.
    Since the recipient must know the shape of the obj in p2p communications,
    meta information of the obj should be sent before communications. This function
    synchronizes with :func:`recv_obj_meta`.

    Args:
        obj (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): obj to be sent.
        need_meta (bool, optional): If False, meta information won't be sent.
        next_rank (int): The rank of the next member in pipeline parallel group.

    Returns:
        bool: False
    """
    if need_meta:
        if next_rank is None:
            next_rank = tpc.get_next_global_rank("pipe")
        # import pdb;pdb.set_trace()

        tensor_kwargs = {"dtype": torch.long, "device": get_current_device()}
        if isinstance(obj, torch.Tensor):
            send_obj_nums = torch.tensor(1, **tensor_kwargs)
            dist.send(send_obj_nums, next_rank)
            send_meta_helper(obj, next_rank, tensor_kwargs)
        else:
            send_obj_nums = torch.tensor(len(obj), **tensor_kwargs)
            dist.send(send_obj_nums, next_rank)
            for tensor_to_send in obj:
                send_meta_helper(tensor_to_send, next_rank, tensor_kwargs)

    return False


def recv_meta_helper(prev_rank, tensor_kwargs):
    recv_ndims = torch.empty((), **tensor_kwargs)
    dist.recv(recv_ndims, prev_rank)
    recv_shape = torch.empty(recv_ndims, **tensor_kwargs)
    dist.recv(recv_shape, prev_rank)
    return recv_shape


def recv_obj_meta(obj_shape, prev_rank=None) -> torch.Size:
    """Receives obj meta information before receiving a specific obj.
    Since the recipient must know the shape of the obj in p2p communications,
    meta information of the obj should be received before communications. This function
    synchronizes with :func:`send_obj_meta`.

    Args:
        obj_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the obj to be received.
        prev_rank (int): The rank of the source of the obj.

    Returns:
        Union[:class:`torch.Size`, List[:class:`torch.Size`]]: The shape of the obj to be received.
    """
    if obj_shape is None:
        if prev_rank is None:
            prev_rank = tpc.get_prev_global_rank("pipe")

        tensor_kwargs = {"dtype": torch.long, "device": get_current_device()}
        recv_obj_nums = torch.empty((), **tensor_kwargs)
        # import pdb;pdb.set_trace()

        dist.recv(recv_obj_nums, prev_rank)
        if recv_obj_nums.item() == 1:
            recv_shape = recv_meta_helper(prev_rank, tensor_kwargs)
            obj_shape = torch.Size(recv_shape)
        else:
            obj_shape = []
            for i in range(recv_obj_nums.item()):
                recv_shape = recv_meta_helper(prev_rank, tensor_kwargs)
                obj_shape.append(torch.Size(recv_shape))

    return obj_shape


def split_tensor_into_1d_equal_chunks(
    tensor: torch.Tensor, new_buffer=False
) -> torch.Tensor:
    """Break a tensor into equal 1D chunks.

    Args:
        tensor (:class:`torch.Tensor`): Tensor to be split before communication.
        new_buffer (bool, optional): Whether to use a new buffer to store sliced tensor.

    Returns:
        :class:`torch.Tensor`: The split tensor
    """
    partition_size = torch.numel(tensor) // tpc.get_group_size("tensor")
    start_index = partition_size * tpc.get_group_rank("tensor")
    end_index = start_index + partition_size
    if new_buffer:
        data = torch.empty(
            partition_size,
            dtype=tensor.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
        data.copy_(tensor.view(-1)[start_index:end_index])
    else:
        data = tensor.view(-1)[start_index:end_index]
    return data


def gather_split_1d_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Opposite of above function, gather values from model parallel ranks.

    Args:
        tensor (:class:`torch.Tensor`): Tensor to be gathered after communication.
    Returns:
        :class:`torch.Tensor`: The gathered tensor.
    """
    world_size = tpc.get_group_size("tensor")
    numel = torch.numel(tensor)
    numel_gathered = world_size * numel
    gathered = torch.empty(
        numel_gathered,
        dtype=tensor.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    chunks = [gathered[i * numel : (i + 1) * numel] for i in range(world_size)]
    dist.all_gather(chunks, tensor, group=tpc.get_group("tensor"))
    return gathered


def _get_tensor_shape(
    tensor_shape: TensorShape, chunk_tensor: bool = False
) -> Tuple[TensorShape, bool]:
    """get the exact tensor shape when communicating and return whether the tensor is a chunk

    Args:
        tensor_shape (:class:`torch.Size`): shape of tensor
        chunk_tensor (bool, optional): whether to chunk tensor, defaults to False

    Returns:
        Tuple[Union[:class:`torch.Size`, List[int], Tuple[int]], bool]: exact tensor shape, whether to chunk tensor
    """
    if chunk_tensor:
        tensor_chunk_shape = reduce(operator.mul, tensor_shape, 1)
        tensor_parallel_world_size = tpc.get_group_size("tensor")
        if tensor_chunk_shape % tensor_parallel_world_size == 0:
            tensor_chunk_shape = tensor_chunk_shape // tensor_parallel_world_size
        else:
            tensor_chunk_shape = tensor_shape
            chunk_tensor = False
    else:
        tensor_chunk_shape = tensor_shape
    return tensor_chunk_shape, chunk_tensor


def create_recv_buffer_with_shapes(recv_shapes, dtype, scatter_gather_tensors):
    if isinstance(recv_shapes, torch.Size):
        recv_chunk_shape, recv_split = _get_tensor_shape(
            recv_shapes, scatter_gather_tensors
        )
        buffer_recv = torch.empty(
            recv_chunk_shape,
            requires_grad=True,
            device=get_current_device(),
            dtype=dtype,
        )
        return buffer_recv, recv_split
    buffer_recv = []
    for recv_shape in recv_shapes:
        recv_chunk_shape, recv_split = _get_tensor_shape(
            recv_shape, scatter_gather_tensors
        )
        tensor_recv = torch.empty(
            recv_chunk_shape,
            requires_grad=True,
            device=get_current_device(),
            dtype=dtype,
        )
        buffer_recv.append(tensor_recv)
    return buffer_recv, recv_split


def process_object_to_send(object_send, scatter_gather_tensors):
    if isinstance(object_send, torch.Tensor):
        send_split = _get_tensor_shape(object_send.shape, scatter_gather_tensors)[1]
        if send_split:
            object_send = split_tensor_into_1d_equal_chunks(object_send)
        return object_send

    object_send_list = []
    for tensor_send in object_send:
        send_split = _get_tensor_shape(tensor_send.shape, scatter_gather_tensors)[1]
        if send_split:
            object_send_list.append(split_tensor_into_1d_equal_chunks(tensor_send))
        else:
            object_send_list.append(tensor_send)
    object_send = tuple(object_send_list)

    return object_send


def filling_ops_queue(obj, comm_op, comm_rank, ops_queue):
    if isinstance(obj, torch.Tensor):
        op_to_add = dist.P2POp(comm_op, obj, comm_rank)
        ops_queue.append(op_to_add)
    else:
        for tensor_to_comm in obj:
            op_to_add = dist.P2POp(comm_op, tensor_to_comm, comm_rank)
            ops_queue.append(op_to_add)


def _communicate(
    object_send_next: Union[torch.Tensor, List[torch.Tensor]] = None,
    object_send_prev: Union[torch.Tensor, List[torch.Tensor]] = None,
    recv_prev: bool = False,
    recv_next: bool = False,
    recv_prev_shape: Union[torch.Size, List[torch.Size]] = None,
    recv_next_shape: Union[torch.Size, List[torch.Size]] = None,
    prev_rank: int = None,
    next_rank: int = None,
    dtype: torch.dtype = None,
    scatter_gather_tensors: bool = False,
) -> Tuple[Union[torch.Tensor, List[torch.Tensor]]]:
    """
    Adapted from megatron.p2p_communication.
    Communicate tensors between stages. Used as helper method in other
    communication methods that are used in pipeline schedule.
    Takes the following arguments:
        object_send_next (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): tensor to send to next rank (no tensor sent if
                          set to None).
        object_send_prev (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev (bool): boolean for whether tensor should be received from
                   previous rank.
        recv_next (bool): boolean for whether tensor should be received from
                   next rank.
        recv_prev_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): shape of the tensor to be received from the previous stage, defaults to None.
        recv_next_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): shape of the tensor to be received from the next stage, defaults to None.
        prev_rank (int): the rank of the previous pipeline stage, defaults to None,
        next_rank (int): the rank of the next pipeline stage, defaults to None,
        dtype (torch.dtype): data type of intermediate buffers, defaults to None
        scatter_gather_tensors (bool): whether to scatter and gather tensor between pipeline stages, defaults to False
    Returns:
        Tuple[Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]]: returns tensor_recv_prev, tensor_recv_next
    """

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    if recv_prev:
        assert recv_prev_shape is not None
        tensor_recv_prev, recv_prev_split = create_recv_buffer_with_shapes(
            recv_prev_shape, dtype, scatter_gather_tensors
        )

    if recv_next:
        assert recv_next_shape is not None
        tensor_recv_next, recv_next_split = create_recv_buffer_with_shapes(
            recv_next_shape, dtype, scatter_gather_tensors
        )

    if object_send_prev is not None or recv_prev:
        if prev_rank is None:
            prev_rank = tpc.get_prev_global_rank("pipe")

    if object_send_next is not None or recv_next:
        if next_rank is None:
            next_rank = tpc.get_next_global_rank("pipe")

    if object_send_prev is not None:
        object_send_prev = process_object_to_send(
            object_send_prev, scatter_gather_tensors
        )

    if object_send_next is not None:
        object_send_next = process_object_to_send(
            object_send_next, scatter_gather_tensors
        )

    ops = []
    if object_send_prev is not None:
        filling_ops_queue(object_send_prev, dist.isend, prev_rank, ops)

    if tensor_recv_prev is not None:
        filling_ops_queue(tensor_recv_prev, dist.irecv, prev_rank, ops)

    if tensor_recv_next is not None:
        filling_ops_queue(tensor_recv_next, dist.irecv, next_rank, ops)

    if object_send_next is not None:
        filling_ops_queue(object_send_next, dist.isend, next_rank, ops)

    if len(ops) > 0:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
    # To protect against race condition when using batch_isend_irecv().
    torch.cuda.synchronize()

    if recv_prev and recv_prev_split:
        if isinstance(tensor_recv_prev, torch.Tensor):
            tensor_recv_prev = (
                gather_split_1d_tensor(tensor_recv_prev)
                .view(recv_prev_shape)
                .requires_grad_()
            )
        else:
            for index in range(len(tensor_recv_prev)):
                tensor_recv_prev[index] = (
                    gather_split_1d_tensor(tensor_recv_prev[index])
                    .view(recv_prev_shape[index])
                    .requires_grad_()
                )

    if recv_next and recv_next_split:
        if isinstance(tensor_recv_next, torch.Tensor):
            tensor_recv_next = (
                gather_split_1d_tensor(tensor_recv_next)
                .view(recv_next_shape)
                .requires_grad_()
            )
        else:
            for index in range(len(tensor_recv_next)):
                tensor_recv_next[index] = (
                    gather_split_1d_tensor(tensor_recv_next[index])
                    .view(recv_next_shape[index])
                    .requires_grad_()
                )

    return tensor_recv_prev, tensor_recv_next


def recv_forward(
    input_tensor_shape, prev_rank=None, dtype=torch.float, scatter_gather_tensors=False
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Copy the forward output from the previous stage in pipeline as the input tensor of this stage.
    Args:
        input_tensor_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor to be received.
        prev_rank (int, optional): The rank of the source of the tensor.
    Returns:
        Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: The input tensor or input tensor list.
    """
    if tpc.is_first_in_pipeline_group():
        input_tensor = None
    else:
        input_tensor, _ = _communicate(
            recv_prev=True,
            recv_prev_shape=input_tensor_shape,
            prev_rank=prev_rank,
            dtype=dtype,
            scatter_gather_tensors=scatter_gather_tensors,
        )
    return input_tensor


def recv_backward(
    output_grad_shape, next_rank=None, dtype=torch.float, scatter_gather_tensors=False
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Copy the gradient tensor from the next stage in pipeline as the input gradient of this stage.
    Args:
        output_grad_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor to be received.
        next_rank (int, optional): The rank of the source of the tensor.
    Returns:
        Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: The input gradient tensor or gradident tensor list.
    """
    if tpc.is_last_in_pipeline_group():
        output_tensor_grad = None
    else:
        _, output_tensor_grad = _communicate(
            recv_next=True,
            recv_next_shape=output_grad_shape,
            next_rank=next_rank,
            dtype=dtype,
            scatter_gather_tensors=scatter_gather_tensors,
        )
    return output_tensor_grad


def send_forward(output_tensor, next_rank=None, scatter_gather_tensors=False) -> None:
    """Sends the input tensor to the next stage in pipeline.
    Args:
        output_tensor (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor to be sent.
        next_rank (int, optional): The rank of the recipient of the tensor.
    """
    if not tpc.is_last_in_pipeline_group():
        _communicate(
            object_send_next=output_tensor,
            next_rank=next_rank,
            scatter_gather_tensors=scatter_gather_tensors,
        )


def send_backward(
    input_tensor_grad, prev_rank=None, scatter_gather_tensors=False
) -> None:
    """Sends the gradient tensor to the previous stage in pipeline.
    Args:
        input_tensor_grad (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor to be sent
        prev_rank (int, optional): The rank of the recipient of the tensor
    """
    if not tpc.is_first_in_pipeline_group():
        _communicate(
            object_send_prev=input_tensor_grad,
            prev_rank=prev_rank,
            scatter_gather_tensors=scatter_gather_tensors,
        )


def send_forward_recv_backward(
    output_tensor,
    output_grad_shape,
    recv_next=True,
    next_rank=None,
    dtype=torch.float,
    scatter_gather_tensors=False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Batched communication operation. Sends the input tensor to the
    next stage in pipeline, while receives the gradient tensor from the
    next stage in pipeline as the input gradient tensor of this stage.
    Args:
        output_tensor (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor to be sent.
        output_grad_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor to be received.
    Returns:
        Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: The input gradient tensor.
    """
    if tpc.is_last_in_pipeline_group():
        output_tensor_grad = None
    else:
        _, output_tensor_grad = _communicate(
            object_send_next=output_tensor,
            recv_next=recv_next,
            recv_next_shape=output_grad_shape,
            next_rank=next_rank,
            dtype=dtype,
            scatter_gather_tensors=scatter_gather_tensors,
        )
    return output_tensor_grad


def send_backward_recv_forward(
    input_tensor_grad,
    input_tensor_shape,
    recv_prev=True,
    prev_rank=None,
    dtype=torch.float,
    scatter_gather_tensors=False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Batched communication operation. Sends the gradient tensor to the
    previous stage in pipeline, while receives the output tensor from the
    previous stage in pipeline as the input of this stage.
    Args:
        input_tensor_grad (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor to be sent.
        input_tensor_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor to be received.
    Returns:
        Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: The input tensor.
    """
    if tpc.is_first_in_pipeline_group():
        input_tensor = None
    else:
        input_tensor, _ = _communicate(
            object_send_prev=input_tensor_grad,
            recv_prev=recv_prev,
            recv_prev_shape=input_tensor_shape,
            prev_rank=prev_rank,
            dtype=dtype,
            scatter_gather_tensors=scatter_gather_tensors,
        )
    return input_tensor


def send_forward_recv_forward(
    output_tensor,
    input_tensor_shape,
    recv_prev=True,
    prev_rank=None,
    next_rank=None,
    dtype=torch.float,
    scatter_gather_tensors=False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Batched communication operation. Sends the input tensor to the
    next stage in pipeline, while receives the output tensor from the
    previous stage in pipeline as the input of this stage.
    Args:
        output_tensor (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor to be sent.
        input_tensor_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor to be received.
    Returns:
        Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: The input tensor.
    """
    input_tensor, _ = _communicate(
        object_send_next=output_tensor,
        recv_prev=recv_prev,
        recv_prev_shape=input_tensor_shape,
        prev_rank=prev_rank,
        next_rank=next_rank,
        dtype=dtype,
        scatter_gather_tensors=scatter_gather_tensors,
    )
    return input_tensor


def send_backward_recv_backward(
    input_tensor_grad,
    output_grad_shape,
    recv_next=True,
    prev_rank=None,
    next_rank=None,
    dtype=torch.float,
    scatter_gather_tensors=False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Batched communication operation. Sends the gradient tensor to the
    previous stage in pipeline, while receives the gradient tensor from the
    next member in pipeline as the input of this stage.
    Args:
        input_tensor_grad (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor to be sent.
        output_grad_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor to be received.
    Returns:
        Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: The input gradient tensor.
    """
    _, output_tensor_grad = _communicate(
        object_send_prev=input_tensor_grad,
        recv_next=recv_next,
        recv_next_shape=output_grad_shape,
        prev_rank=prev_rank,
        next_rank=next_rank,
        dtype=dtype,
        scatter_gather_tensors=scatter_gather_tensors,
    )
    return output_tensor_grad


def send_forward_backward_recv_forward_backward(
    output_tensor,
    input_tensor_grad,
    input_tensor_shape,
    output_grad_shape,
    recv_prev=True,
    recv_next=True,
    prev_rank=None,
    next_rank=None,
    dtype=torch.float,
    scatter_gather_tensors=False,
) -> Tuple[Union[torch.Tensor, List[torch.Tensor]]]:
    """Batched communication operation. Sends the input tensor to the next stage in pipeline and
    the gradient tensor to the previous stage, while receives the input gradient tensor from the
    next stage and the input tensor from the previous stage.
    Args:
        output_tensor (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor sent to the next.
        input_tensor_grad (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor sent to the previous.
        input_tensor_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor received from the previous.
        output_grad_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor received from the next.
    Returns:
        Tuple(Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]], Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): (the input tensor, the input gradient tensor)
    """
    input_tensor, output_tensor_grad = _communicate(
        object_send_next=output_tensor,
        object_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        recv_prev_shape=input_tensor_shape,
        recv_next_shape=output_grad_shape,
        prev_rank=prev_rank,
        next_rank=next_rank,
        dtype=dtype,
        scatter_gather_tensors=scatter_gather_tensors,
    )
    return input_tensor, output_tensor_grad
