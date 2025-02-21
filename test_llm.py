import numpy as np

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

from stepvideo.diffusion.video_pipeline import StepVideoPipeline
from stepvideo.config import parse_args
from stepvideo.utils import setup_seed
from stepvideo.parallel import initialize_parall_group


# for test
import os
from api.call_remote_server import CaptionPipeline, StepVaePipeline


if __name__ == "__main__":
    args = parse_args()
    args.llm_dir = "step_llm"
    args.clip_dir = "hunyuan_clip"

    ms.set_context(
        mode=ms.PYNATIVE_MODE,
        jit_config={"jit_level": "O0"},
        deterministic="ON",
        pynative_synchronize=True,
        memory_optimize_level="O1",
        # max_device_memory="59GB",
        # jit_syntax_level=ms.STRICT,
    )
    
    initialize_parall_group(args, ring_degree=args.ring_degree, ulysses_degree=args.ulysses_degree)
    
    setup_seed(args.seed)
        
    # pipeline = StepVideoPipeline.from_pretrained(args.model_dir).to(ms.bfloat16)
    # pipeline.setup_api(
    #     vae_url = args.vae_url,
    #     caption_url = args.caption_url,
    # )

    caption_pipeline = CaptionPipeline(
        llm_dir=os.path.join(args.model_dir, args.llm_dir), 
        clip_dir=os.path.join(args.model_dir, args.clip_dir)
    )

    def encode_prompt(
        prompt: str,
        neg_magic: str = '',
        pos_magic: str = '',
    ):
        prompts = [prompt+pos_magic]
        bs = len(prompts)
        prompts += [neg_magic]*bs

        # data = self.caption(prompts)
        data = caption_pipeline.embedding(prompts)

        prompt_embeds, prompt_attention_mask, clip_embedding = Tensor(data['y']), Tensor(data['y_mask']), Tensor(data['clip_embedding'])

        print(f"encode_prompt output shape:")
        print(f"{prompt_embeds.shape=}, {prompt_embeds.dtype=}")
        print(f"{prompt_attention_mask.shape=}, {prompt_attention_mask.dtype=}")
        print(f"{clip_embedding.shape=}, {clip_embedding.dtype=}")

        return prompt_embeds, clip_embedding, prompt_attention_mask


    # 3. Encode input prompt
    prompt = args.prompt
    prompt_embeds, prompt_embeds_2, prompt_attention_mask = encode_prompt(
        prompt=prompt,
        neg_magic=args.neg_magic,
        pos_magic=args.pos_magic,
    )


    import pdb;pdb.set_trace()

    transformer_dtype = ms.bfloat16
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
    prompt_embeds_2 = prompt_embeds_2.to(transformer_dtype)

    print(f"{prompt_embeds.shape=}")
    print(f"{prompt_attention_mask.shape=}")
    print(f"{prompt_embeds_2.shape=}")
