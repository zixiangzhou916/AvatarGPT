import os, sys, argparse
sys.path.append(os.getcwd())
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
from moviepy.editor import VideoFileClip, concatenate_videoclips, clips_array, vfx
import time
from multiprocessing import Process

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

def resample_load_librosa(path, sample_rate=16000, downmix_to_mono=True):
    src, sr = librosa.load(path, sr=sample_rate, mono=downmix_to_mono)
    return src, sr

def plot(time_axis, data, sr, indices, temp_output_path):
    for idx in indices:
        time = idx * 16000 / 30
        fig = plt.figure(figsize=(10, 10), dpi=100)
        plt.plot(time_axis, data)
        plt.axvline(x=time / sr, color='r')
        plt.xlabel("Time (s)")
        plt.ylabel("Audio")
        plt.show()
        plt.savefig(os.path.join(temp_output_path, "{:d}.png".format(idx)))
        plt.close()

def showing_audiotrack(data, sr, output_path="temp", output_name=None, fps=30, num_proc=8):
    temp_output_path = os.path.join(output_path, "audio_temp")
    if not os.path.exists(temp_output_path):
        os.makedirs(temp_output_path)
        
    n = len(data)
    duration = len(data) / sr
    num_frames = int(duration * fps)
    time_axis = np.linspace(0, n / sr, n, endpoint=False)
    
    times = np.arange(0, num_frames)
    num_frames_per_proc = num_frames // num_proc + 1
    indices_list = []
    for n in range(num_proc):
        indices_list.append(times[n*num_frames_per_proc:(n+1)*num_frames_per_proc])
                            
    job_list = []
    for i in range(num_proc):
        job = Process(target=plot, 
                      args=(time_axis, data, sr, indices_list[i], temp_output_path,))
        job.start()
        job_list.append(job)
        
    for job in job_list:
        job.join()    
    
    files = [int(f.split(".")[0]) for f in os.listdir(temp_output_path) if ".png" in f]
    files = ["{:d}.png".format(f) for f in sorted(files)]
    
    with imageio.get_writer(os.path.join(output_path, output_name+".mp4"), fps=fps) as writer:
        for file in tqdm(files, desc="Generating MP4..."):
            img = np.array(imageio.imread(os.path.join(temp_output_path, file)), dtype=np.uint8)
            writer.append_data(img[..., :3])
            os.remove(os.path.join(temp_output_path, file))

def merge_videos(vid_1, vid_2, final_vid):
    print(" --- Stack video clips")
    clip1 = VideoFileClip(vid_1)
    clip2 = VideoFileClip(vid_2)
    h1 = clip1.h
    w1 = clip1.w
    h2 = clip2.h
    w2 = clip2.w
    scale = h1 / h2
    clip2_resize = clip2.resize(scale)
    final_clip = clips_array([[clip1, clip2_resize]])
    final_clip.resize(0.5).write_videofile(final_vid)

def main(audio_file, track_path, track_name, video_path, video_name):
    data, sr = resample_load_librosa(audio_file, sample_rate=16000, downmix_to_mono=True)
    showing_audiotrack(data, sr, track_path, track_name, fps=30, num_proc=8)
    merge_videos(os.path.join(video_path, video_name), 
                 os.path.join(track_path, track_name+".mp4"), 
                 os.path.join(video_path, track_name.replace("_track", "_merge.mp4")))

if __name__ == "__main__":
    audio_name = "2_scott_raw_video_1_004"
    main(audio_file="../dataset/YoYo/audios/2/2_scott_raw_video_1_004.wav", 
         track_path="logs/ude_v2/eval/ude/yoyo/ude_clip_mtr_wav2vec2/animation/s2m", 
         track_name=audio_name+"_track", 
         video_path="logs/ude_v2/eval/ude/yoyo_base_base/ude_clip_mtr_wav2vec2_short_primitive/animation/s2m/", 
         video_name="2_scott_raw_video_1_004_audio.mp4")
    # # wav_file = "../dataset/YoYo/audios/2/2_scott_raw_video_1_003.wav"
    # wav_file = "../dataset/YoYo/audios/2/2_scott_raw_video_1_004.wav"
    # # data, sr = sf.read(wav_file)
    # data, sr = resample_load_librosa(wav_file, sample_rate=16000, downmix_to_mono=True)
    # print('Start to visualize audiotrack...')
    # showing_audiotrack(data, sr, 
    #                    output_path="logs/ude_v2/eval/ude/yoyo/ude_clip_mtr_wav2vec2/animation/s2m", 
    #                    output_name="2_scott_raw_video_1_004_track", fps=30, num_proc=32)
    # merge_videos("logs/ude_v2/eval/ude/yoyo/ude_clip_mtr_wav2vec2/animation/s2m/2_scott_raw_video_1_004_audio.mp4", 
    #              "logs/ude_v2/eval/ude/yoyo/ude_clip_mtr_wav2vec2/animation/s2m/2_scott_raw_video_1_004_track.mp4", 
    #              "logs/ude_v2/eval/ude/yoyo/ude_clip_mtr_wav2vec2/animation/s2m/2_scott_raw_video_1_004_merge.mp4")
