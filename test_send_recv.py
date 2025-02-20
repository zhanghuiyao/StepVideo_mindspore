import numpy as np

import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.communication import init, get_rank


_shape = (2, 3)
send_recv_map = {
    0: {'tag': 0, "src_dst_id": 2}, 2: {'tag': 0, "src_dst_id": 0}, #0->2
    1: {'tag': 1, "src_dst_id": 3}, 3: {'tag': 1, "src_dst_id": 1}, #1->3
}


def check_ops_comm():

    init()

    tag, src_dst_id = send_recv_map[get_rank()]["tag"], send_recv_map[get_rank()]["src_dst_id"]

    if get_rank() in [0, 1]:
        print("RANK {get_rank()}, sending...")
        _tensor = Tensor(np.random.randn(*_shape), ms.float32)
        send_op = ops.Send(sr_tag=tag, dest_rank=src_dst_id)
        send_op(_tensor)
        print(f"RANK {get_rank()}, send success, {_tensor=}")
    elif get_rank() in [2, 3]:
        print("RANK {get_rank()}, recving...")
        recv_op = ops.Receive(sr_tag=tag, src_rank=src_dst_id, shape=_shape, dtype=ms.float32)
        _tensor = recv_op()
        print(f"RANK {get_rank()}, recv success, {_tensor=}")


def check_mint_comm():
    from mindspore.mint.distributed import init_process_group
    from mindspore.mint.distributed import send, recv, get_rank

    init()
    init_process_group()

    tag, src_dst_id = send_recv_map[get_rank()]["tag"], send_recv_map[get_rank()]["src_dst_id"]

    if get_rank() in [0, 1]:
        print("RANK {get_rank()}, sending...")
        _tensor = Tensor(np.random.randn(*_shape), ms.float32).to(ms.bfloat16)
        send(_tensor, src_dst_id)
        print(f"RANK {get_rank()}, send success, {_tensor=}")
    elif get_rank() in [2, 3]:
        print("RANK {get_rank()}, recving...")
        _tensor = ops.zeros(_shape, ms.bfloat16)
        recv(_tensor, src_dst_id)
        print(f"RANK {get_rank()}, recv success, {_tensor=}")



# check_ops_comm()
check_mint_comm()
