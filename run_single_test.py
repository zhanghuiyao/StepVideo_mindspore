import numpy as np

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

from stepvideo.diffusion.video_pipeline import StepVideoPipeline
from stepvideo.config import parse_args
from stepvideo.utils import setup_seed
from stepvideo.parallel import initialize_parall_group, get_parallel_group


# for test
import os
from .api.call_remote_server import CaptionPipeline, StepVaePipeline


if __name__ == "__main__":
    args = parse_args()
    
    ms.set_context(
        mode=ms.PYNATIVE_MODE,
        jit_config={"jit_level": "O0"},
        deterministic="ON",
        pynative_synchronize=True,
        memory_optimize_level="O1",
        # max_device_memory="59GB",
        # jit_syntax_level=ms.STRICT,
    )
    
    initialize_parall_group(ring_degree=args.ring_degree, ulysses_degree=args.ulysses_degree)
    
    # local_rank = get_parallel_group().local_rank
    
    setup_seed(args.seed)
        
    pipeline = StepVideoPipeline.from_pretrained(args.model_dir).to(ms.bfloat16)
    pipeline.setup_api(
        vae_url = args.vae_url,
        caption_url = args.caption_url,
    )
    
    
    def decode_vae(self, samples: Tensor):
        # () -> (b, 128, 128, 16)
        # samples = np.random.randn(2, 128, 128, 16)

        samples = self.vae_pipeline.decode(samples)

        return samples

    def encode_prompt(
        self,
        prompt: str,
        neg_magic: str = '',
        pos_magic: str = '',
    ):
        prompts = [prompt+pos_magic]
        bs = len(prompts)
        prompts += [neg_magic]*bs

        # data = self.caption(prompts)
        data = self.caption_pipeline.embedding(prompts)

        prompt_embeds, prompt_attention_mask, clip_embedding = Tensor(data['y']), Tensor(data['y_mask']), Tensor(data['clip_embedding'])

        return prompt_embeds, clip_embedding, prompt_attention_mask

    # replace method for test, 
    # test shape: (128, 128, 16), official shape: 544px992px136f
    pipeline.decode_vae = decode_vae
    pipeline.encode_prompt = encode_prompt

    pipeline.caption = CaptionPipeline(
        llm_dir=os.path.join(args.model_dir, args.llm_dir), 
        clip_dir=os.path.join(args.model_dir, args.clip_dir)
    )
    pipeline.vae = StepVaePipeline(
        vae_dir=os.path.join(args.model_dir, args.vae_dir)
    )


    prompt = args.prompt
    videos = pipeline(
        prompt=prompt, 
        num_frames=args.num_frames, 
        height=args.height, 
        width=args.width,
        num_inference_steps = args.infer_steps,
        guidance_scale=args.cfg_scale,
        time_shift=args.time_shift,
        pos_magic=args.pos_magic,
        neg_magic=args.neg_magic,
        output_file_name=prompt[:50]
    )
