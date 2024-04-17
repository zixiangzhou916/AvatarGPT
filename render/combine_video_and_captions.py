import os, sys, argparse
sys.path.append(os.getcwd())
import numpy as np
import imageio
import json
from tqdm import tqdm

def video2frames(video_name):
    vid = imageio.get_reader(video_name, "ffmpeg")
    images = []
    for num, image in enumerate(vid):
        images.append(image)
    return images

def combine(video_file, caption_file):
    video_frames = video2frames(video_file)
    caption_frames = video2frames(caption_file)
    images = []
    for (vframe, cframe) in tqdm(zip(video_frames, caption_frames)):
        image = np.concatenate([cframe, vframe[200:]], axis=0)
        images.append(image)
    images = np.stack(images, axis=0)
    return images

if __name__ == "__main__":
    video_input_dir = "logs/avatar_gpt/eval/flan_t5_large/exp7/animation/se1_p0"
    caption_input_dir = "logs/avatar_gpt/eval/flan_t5_large/exp7/captions/se1_p0"
    output_dir = "logs/avatar_gpt/eval/flan_t5_large/exp7/captioned/se1_p0/"
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(video_input_dir) if ".mp4" in f]
    # print(files)
    for file in files:
        video_file = os.path.join(video_input_dir, file)
        caption_file = os.path.join(caption_input_dir, file)
        if not os.path.exists(caption_file):
            continue
        if os.path.exists(output_dir+file):
            continue
        images = combine(video_file=video_file, caption_file=caption_file)
        imageio.mimsave(output_dir+file, images, fps=20)