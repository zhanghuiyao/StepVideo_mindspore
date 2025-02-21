#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import os

def read_video_to_numpy(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="读取视频文件 (mp4) 并将其转换为 numpy 数组")
    parser.add_argument("video_path", help="视频文件路径 (mp4)")
    parser.add_argument("--output", "-o", help="可选的, 保存 numpy 数组的文件路径 (.npy)", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print("视频文件不存在:", args.video_path)
        exit(1)

    video_array = read_video_to_numpy(args.video_path)
    print("视频读取完毕, 数组形状:", video_array.shape)
    if args.output:
        np.save(args.output, video_array)
        print("数组已保存到:", args.output)
