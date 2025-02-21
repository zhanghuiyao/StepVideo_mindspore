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


    vae_pipeline = StepVaePipeline(
        vae_dir=os.path.join(args.model_dir, "vae")
    )

    def decode_vae(samples: Tensor):
        # () -> (b, 128, 128, 16)
        # samples = np.random.randn(2, 128, 128, 16)

        print(f"decode_vae input shape: {samples.shape}")

        samples = vae_pipeline.decode(samples)

        print(f"decode_vae output shape: {samples.shape}")

        return samples

    x = Tensor(np.random.randn(1, 36, 64, 34, 62))
    out = decode_vae(x)

    print(f"{out.shape=}")
    

    # save video
    from stepvideo.utils.video_process import VideoProcessor
    video_processor = VideoProcessor("./results", "")
    video_processor.postprocess_video(out, output_file_name="test_video", output_type="mp4")

    print(f"svae test_video success.")
