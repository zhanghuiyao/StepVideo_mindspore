import numpy as np

from stepvideo.utils.video_process import VideoProcessor


def save():

    vae_out = np.load("./npys/vae_output_numpy.py")

    # save video
    from stepvideo.utils.video_process import VideoProcessor
    video_processor = VideoProcessor("./results", "")
    video_processor.postprocess_video(vae_out, output_file_name="test_video", output_type="mp4")


save()