# Copyright 2025 StepFun Inc. All Rights Reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

from mindone.transformers.mindspore_adapter.attention import FlashAttention2


class FlashSelfAttention(FlashAttention2):
    def __init__(self, head_dim, head_num, attention_dropout = 0, dtype = ms.float16):
        super().__init__(head_dim, head_num, attention_dropout, input_layout="BNSD", dtype=dtype)
    
    def construct(self, q, k, v, cu_seqlens=None, max_seq_len=None, mask=None):
        # BSND -> BNSD
        q = q.swapaxes(1, 2)
        k = k.swapaxes(1, 2)
        v = v.swapaxes(1, 2)

        output = super().construct(q, k, v, mask)

        # BNSD -> BSND
        output = output.swapaxes(1, 2)

        return output    
