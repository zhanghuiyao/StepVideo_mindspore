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

import os
import numpy as np
from typing import Any, Dict, Optional, Union
from stepvideo.parallel import parallel_forward, get_sp_group
from stepvideo.modules.blocks import (
        StepVideoTransformerBlock, 
        PatchEmbed
    )
from stepvideo.modules.normalization import (
        PixArtAlphaTextProjection,
        AdaLayerNormSingle
    )

from mindone.diffusers.configuration_utils import ConfigMixin, register_to_config
from mindone.diffusers.models.modeling_utils import ModelMixin



class StepVideoModel(ModelMixin, ConfigMixin):
    _no_split_modules = ["StepVideoTransformerBlock", "PatchEmbed"]

    # @with_empty_init
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 48,
        attention_head_dim: int = 128,
        in_channels: int = 64,
        out_channels: Optional[int] = 64,
        num_layers: int = 48,
        dropout: float = 0.0,
        patch_size: int = 1,
        norm_type: str = "ada_norm_single",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        use_additional_conditions: Optional[bool] = False,
        caption_channels: Optional[Union[int, list, tuple]] = [6144, 1024],
        attention_type: Optional[str] = "parallel",
    ):
        super().__init__()

        # get sp_group from global var
        sp_group = None
        if attention_type == "parallel":
            sp_group = get_sp_group()

        # Set some common variables used across the board.
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels

        self.use_additional_conditions = use_additional_conditions

        self.pos_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
        )

        self.transformer_blocks = nn.CellList(
            [
                StepVideoTransformerBlock(
                    dim=self.inner_dim,
                    attention_head_dim=self.config.attention_head_dim,
                    attention_type=attention_type,
                    sp_group=sp_group
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # 3. Output blocks.
        self.norm_out = nn.LayerNorm([self.inner_dim], epsilon=norm_eps)
        if not norm_elementwise_affine:
            self.norm_out.gamma.requires_grad = False
            self.norm_out.beta.requires_grad = False
        self.scale_shift_table = Parameter(np.random.randn(2, self.inner_dim) / self.inner_dim**0.5)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels)
        self.patch_size = patch_size

        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim, use_additional_conditions=self.use_additional_conditions
        )

        if isinstance(self.config.caption_channels, int):
            caption_channel = self.config.caption_channels
            self.clip_projection = None
        else:
            caption_channel, clip_channel = self.config.caption_channels
            self.clip_projection = nn.Linear(clip_channel, self.inner_dim) 

        self.caption_norm = nn.LayerNorm([caption_channel], epsilon=norm_eps)
        if not norm_elementwise_affine:
            self.caption_norm.gamma.requires_grad = False
            self.caption_norm.beta.requires_grad = False
        
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channel, hidden_size=self.inner_dim
        )

        self.parallel = attention_type=='parallel'

    def patchfy(self, hidden_states):
        # hidden_states = rearrange(hidden_states, 'b f c h w -> (b f) c h w')
        b, f, c, h, w = hidden_states.shape
        hidden_states = hidden_states.view(b*f, c, h, w)

        hidden_states = self.pos_embed(hidden_states)
        return hidden_states

    # FIXME: encoder_hidden_states shape
    def prepare_attn_mask(self, encoder_attention_mask, encoder_hidden_states, q_seqlen):
        kv_seqlens = encoder_attention_mask.sum(axis=1).int()
        mask = ops.zeros([len(kv_seqlens), q_seqlen, int(kv_seqlens.max())], dtype=ms.bool_)
        encoder_hidden_states = encoder_hidden_states[:,: kv_seqlens.max()]  # FIXME: dynamic shape
        for i, kv_len in enumerate(kv_seqlens):
            mask[i, :, :kv_len] = 1
        return encoder_hidden_states, mask
        
        
    @parallel_forward
    def block_forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        rope_positions=None,
        attn_mask=None,
        parallel=True
    ):

        for i, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep=timestep,
                attn_mask=attn_mask,
                rope_positions=rope_positions
            )

        return hidden_states
        

    # @inference_mode()
    def construct(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_hidden_states_2: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
        added_cond_kwargs: Dict[str, Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        fps: Tensor=None,
        return_dict: bool = True,  # FIXME: return_dict
    ):
        assert hidden_states.ndim==5; "hidden_states's shape should be (bsz, f, ch, h ,w)"

        bsz, frame, _, height, width = hidden_states.shape
        height, width = height // self.patch_size, width // self.patch_size
                
        hidden_states = self.patchfy(hidden_states) 
        len_frame = hidden_states.shape[1]
                
        if self.use_additional_conditions:
            added_cond_kwargs = {
                "resolution": Tensor([(height, width)]*bsz, dtype=hidden_states.dtype),
                "nframe": Tensor([frame]*bsz, dtype=hidden_states.dtype),
                "fps": fps
            }
        else:
            added_cond_kwargs = {}
        
        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs=added_cond_kwargs
        )

        encoder_hidden_states = self.caption_projection(self.caption_norm(encoder_hidden_states))
        
        # if encoder_hidden_states_2 is not None and hasattr(self, 'clip_projection'):
        if encoder_hidden_states_2 is not None and self.clip_projection is not None:
            clip_embedding = self.clip_projection(encoder_hidden_states_2)
            encoder_hidden_states = ops.cat([clip_embedding, encoder_hidden_states], axis=1)

        # hidden_states = rearrange(hidden_states, '(b f) l d->  b (f l) d', b=bsz, f=frame, l=len_frame).contiguous()
        hidden_states = hidden_states.view(bsz, frame * len_frame, -1)

        encoder_hidden_states, attn_mask = self.prepare_attn_mask(encoder_attention_mask, encoder_hidden_states, q_seqlen=frame*len_frame)
        
        hidden_states = self.block_forward(
            hidden_states,
            encoder_hidden_states,
            timestep=timestep,
            rope_positions=[frame, height, width],
            attn_mask=attn_mask,
            parallel=self.parallel
        )
        
        # hidden_states = rearrange(hidden_states, 'b (f l) d -> (b f) l d', b=bsz, f=frame, l=len_frame)
        hidden_states = hidden_states.view(bsz * frame, len_frame, -1)

        # embedded_timestep = repeat(embedded_timestep, 'b d -> (b f) d', f=frame).contiguous()
        b, d = embedded_timestep.shape
        embedded_timestep.view(b, 1, d).broadcast_to((b, frame, d)).view(b*frame, d)

        shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, axis=1)
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)
        
        # unpatchify
        hidden_states = hidden_states.reshape(
            (-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        )

        # hidden_states = rearrange(hidden_states, 'n h w p q c -> n c h p w q')
        hidden_states = hidden_states.transpose(0, 5, 1, 3, 2, 4)

        output = hidden_states.reshape(
            (-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        )

        # output = rearrange(output, '(b f) c h w -> b f c h w', f=frame)
        _, c, h, w = output.shape
        output = output.view(-1, frame, c, h, w)

        # if return_dict:
        #     return {'x': output}
        return output
    
