import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

from mindspore.communication.management import GlobalComm, init, get_group_size, get_rank


# FIXME: valid global variables in mindspore static graph
sp_group = None
sp_size = None
sp_rank = None
_is_distribute = False


def is_distribute():
    return _is_distribute


def initialize_parall_group(ring_degree=1, ulysses_degree=1):

    global _is_distribute

    world_size = 1
    rank_id = 0
    if ring_degree > 1 or ulysses_degree > 1:
        init()
        world_size = get_group_size()
        rank_id = get_rank()
        print(f"init_environment, rank_id: {rank_id}, world_size: {world_size}")

        ms.reset_auto_parallel_context()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=world_size,
        )

        _is_distribute = True

    global sp_group
    global sp_size
    global sp_rank

    if ring_degree > 1:
        raise NotImplementedError
    elif ulysses_degree > 1:
        if ulysses_degree == world_size:
            sp_group = GlobalComm.WORLD_COMM_GROUP
            sp_size = world_size
            sp_rank = rank_id
        else:
            from mindspore.communication import create_group

            g_id = rank_id // ulysses_degree
            s_id, e_id = g_id * ulysses_degree, (g_id + 1) * ulysses_degree
            comm_group = f"sub_sp_group_{g_id}"
            create_group(comm_group, [_i for _i in range(s_id, e_id)])
            
            sp_size = ulysses_degree
            sp_rank = rank_id % ulysses_degree
            sp_group = comm_group
    else:
        sp_size = 1
        sp_rank = 0
        sp_group = None


    # dist.init_process_group("nccl")
    # xfuser.core.distributed.init_distributed_environment(
    #     rank=dist.get_rank(), 
    #     world_size=dist.get_world_size()
    # )
    #
    # xfuser.core.distributed.initialize_model_parallel(
    #     sequence_parallel_degree=dist.get_world_size(),
    #     ring_degree=ring_degree,
    #     ulysses_degree=ulysses_degree,
    # )

def get_parallel_group():
    # return xfuser.core.distributed.get_world_group()
    return get_group_size()

def get_sequence_parallel_world_size():
    # return xfuser.core.distributed.parallel_state.get_sequence_parallel_world_size()
    return sp_size

def get_sequence_parallel_rank():
    # return xfuser.core.distributed.parallel_state.get_sequence_parallel_rank()
    return sp_rank

def get_sp_group():
    # return xfuser.core.distributed.parallel_state.get_sp_group()
    return sp_group

def parallel_forward(fn_):
    def wrapTheFunction(_, hidden_states, *args, **kwargs):
        if kwargs['parallel']:            
            hidden_states = ops.chunk(hidden_states, get_sequence_parallel_world_size(), axis=-2)[get_sequence_parallel_rank()]
            kwargs['attn_mask'] = ops.chunk(kwargs['attn_mask'], get_sequence_parallel_world_size(), axis=-2)[get_sequence_parallel_rank()]
        output = fn_(_, hidden_states, *args, **kwargs)

        if kwargs['parallel']:
            # output = get_sp_group().all_gather(output.contiguous(), dim=-2)
            output = sp_all_gather(output, dim=-2)

        return output
     
    return wrapTheFunction


def sp_all_gather(input_: Tensor, dim: int = 0):
    
    # w/o sp
    if get_sp_group() is None:
        return input_

    # w/ sp
    world_size = get_sequence_parallel_world_size()
    input_size = input_.shape

    output = ops.AllGather(group=get_sp_group())(input_)  # e.g. (2, 8) -> (4, 8)
    
    dim = -2
    if dim < 0:
        dim += output.ndim
    if dim != 0:
        _shape = [world_size, input_size[0]//world_size, ] + input_size[1:]
        output_tensor = output_tensor.reshape(_shape)
        output_tensor = output_tensor.movedim(0, dim)

    input_size[dim] = input_size[dim] * world_size
    # Reshape
    output_tensor = output_tensor.reshape(input_size)
    return output_tensor
