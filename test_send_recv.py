import numpy as np

import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.communication import init, get_rank


_shape = (2, 3)
send_recv_map = {
    0: {'tag': 0, "src_dst_id": 2}, 2: {'tag': 0, "src_dst_id": 0}, #0->2
    1: {'tag': 1, "src_dst_id": 3}, 3: {'tag': 1, "src_dst_id": 1}, #1->3
    
    4: {'tag': None, "src_dst_id": None},
    5: {'tag': None, "src_dst_id": None},
}
send_recv_map_2 = {
    # captioner ->
    4: [{'tag': 40, "src_dst_id": 0}, {'tag': 41, "src_dst_id": 1}], 0: {'tag': 40, "src_dst_id": 4}, 1: {'tag': 41, "src_dst_id": 4},
    
    # vae <-
    5: {'tag': 35, "src_dst_id": 3},                                 3: {'tag': 35, "src_dst_id": 5},
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
        
        _, prefetch_src_dst_id = send_recv_map_2[get_rank()]["tag"], send_recv_map_2[get_rank()]["src_dst_id"]

        # prefetch
        print(f"prefetch: RANK {get_rank()}, recving...")
        _tensor = ops.zeros(_shape, ms.bfloat16)
        recv(_tensor, prefetch_src_dst_id)
        print(f"prefetch: RANK {get_rank()}, recv success, {_tensor=}")

        # base
        print(f"RANK {get_rank()}, sending...")
        _tensor = Tensor(np.random.randn(*_shape), ms.float32).to(ms.bfloat16)
        send(_tensor, src_dst_id)
        print(f"RANK {get_rank()}, send success, {_tensor=}")
    elif get_rank() in [2, 3]:
        
        # base
        print(f"RANK {get_rank()}, recving...")
        _tensor = ops.zeros(_shape, ms.bfloat16)
        recv(_tensor, src_dst_id)
        print(f"RANK {get_rank()}, recv success, {_tensor=}")

        # post-send
        if get_rank() == 3:
            _, post_src_dst_id = send_recv_map_2[get_rank()]["tag"], send_recv_map_2[get_rank()]["src_dst_id"]

            print(f"post: RANK {get_rank()}, sending...")
            _tensor = Tensor(np.random.randn(*_shape), ms.float32).to(ms.bfloat16)
            send(_tensor, post_src_dst_id)
            print(f"post: RANK {get_rank()}, send success, {_tensor=}")

    elif get_rank() == 4:
        
        # pre-send
        prefetch_list = send_recv_map_2[get_rank()]
        for i, prefetch in enumerate(prefetch_list):
            _, prefetch_src_dst_id = prefetch["tag"], prefetch["src_dst_id"]
            print(f"prefetch send: {i=}, RANK {get_rank()}->{prefetch_src_dst_id}, sending...")
            _tensor = Tensor(np.random.randn(*_shape), ms.float32).to(ms.bfloat16)
            send(_tensor, src_dst_id)
            print(f"prefetch send: {i=}, RANK {get_rank()}->{prefetch_src_dst_id}, send success, {_tensor=}")

    elif get_rank() == 5:
        
        # post-recv
        _, post_src_dst_id = send_recv_map_2[get_rank()]["tag"], send_recv_map_2[get_rank()]["src_dst_id"]
        print(f"post: RANK {get_rank()}, recving...")
        _tensor = ops.zeros(_shape, ms.bfloat16)
        recv(_tensor, post_src_dst_id)
        print(f"post: RANK {get_rank()}, recv success, {_tensor=}")


# check_ops_comm()
check_mint_comm()
