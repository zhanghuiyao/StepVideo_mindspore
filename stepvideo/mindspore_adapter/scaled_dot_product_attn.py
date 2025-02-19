import numpy as np

import mindspore as ms
from mindspore import nn, ops
from mindspore.ops.operations.nn_ops import FlashAttentionScore as _FlashAttention

DTYPE_FP16_MIN = float(np.finfo(np.float16).min)


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, dtype=None):
    # force dtype(fp16 or bf16) precision calculation
    ori_dtype = query.dtype
    if dtype is not None:
        query, key, value = query.astype(dtype), key.astype(dtype), value.astype(dtype)

    if attn_mask is not None:
        if attn_mask.dtype == ms.bool_:
            attn_mask = attn_mask.to(ms.float32)
            attn_mask = attn_mask.masked_fill((1 - attn_mask).to(ms.bool_), DTYPE_FP16_MIN)
        attn_mask = attn_mask.to(query.dtype)

        attn_weight = ops.softmax(
            ops.matmul(query, key.swapaxes(-2, -1)) / (query.shape[-1] ** 0.5) + attn_mask, axis=-1, dtype=ms.float32
        ).astype(query.dtype)
    else:
        L, S = query.shape[-2], key.shape[-2]
        attn_bias = ops.zeros(L, S, dtype=query.dtype)
        if is_causal:
            # assert attn_mask is None
            temp_mask = ops.ones(L, S, dtype=ms.bool_).tril(diagonal=0)
            attn_bias = ops.masked_fill(attn_bias, ops.logical_not(temp_mask), DTYPE_FP16_MIN)
            attn_bias = attn_bias.to(query.dtype)

        attn_weight = ops.softmax(
            ops.matmul(query, key.swapaxes(-2, -1)) / (query.shape[-1] ** 0.5) + attn_bias, axis=-1, dtype=ms.float32
        ).astype(query.dtype)

    attn_weight = ops.dropout(attn_weight, p=dropout_p)

    out = ops.matmul(attn_weight, value)
    out = out.astype(ori_dtype)

    return out
