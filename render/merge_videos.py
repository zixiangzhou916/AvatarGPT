import os, sys
sys.path.append(os.getcwd())
from moviepy.editor import VideoFileClip, concatenate_videoclips, clips_array, vfx
from moviepy.video import fx
from tqdm import tqdm

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def get_video_files_to_merge(base_path, folders):
    common_files = None
    for folder in folders:
        folder_path = base_path + folder
        files = [f for f in os.listdir(folder_path) if ".mp4" in f]
        if common_files is None:
            common_files = files
        else:
            common_files = intersection(common_files, files)
    return common_files

def merge_videos(base_path, folders, crops, files, output_path):
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    for file in tqdm(files):
        if os.path.exists(output_path+"/"+file):
            continue
        videos = []
        for (folder, crop) in zip(folders, crops):
            vid_path = base_path + folder + "/" + file
            clip = VideoFileClip(vid_path)
            w = clip.w
            h = clip.h
            if crop:
                clip = fx.all.crop(clip, x1=w//2, x2=w, y1=0, y2=h)
            videos.append(clip)
        final_clip = clips_array([videos])
        try:
            final_clip.write_videofile(output_path + "/" + file)
        except:
            pass
        
def merge_videos_v2(base_path, files, output_path, output_name):
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    videos = []
    for file in files:
        # videos = []
        vid_path = base_path + "/" + file
        clip = VideoFileClip(vid_path)
        w = clip.w
        h = clip.h
        print('---->', h, w)
        # cropped = fx.all.crop(clip)
        videos.append(clip)
    final_clip = clips_array([videos])
    print('----', final_clip.h, final_clip.w)
    final_clip.write_videofile(output_path+"/"+output_name, logger=None)
    
def merge_videos_v3(video_paths, crops, output_path):
    videos = []
    h = 0
    for (vid, crop) in zip(video_paths, crops):
        clip = VideoFileClip(vid)
        
        # Scale the video, we want to align them with same video height
        if h == 0:
            h = clip.h
        else:
            scale = h / clip.h
            clip = clip.resize(scale)
        
        # Crop the video, we only want to merge the right half.
        if crop:
            w = clip.w
            h = clip.h
            clip = fx.all.crop(clip, x1=w//2, x2=w, y1=0, y2=h)
        
        # clip = clip.subclip(0, 50)  # clip between t=0(sec) to t=50(sec)
        videos.append(clip)
    final_clip = clips_array([videos])
    final_clip.resize(0.5).write_videofile(output_path)

if __name__ == "__main__":
    """"""" V1 """""""
    
    base_path = "debug"
    output_path = "debug/comparison"
    # output_path = "logs/ude_v2/eval/vqvae/comparison"
    # output_path = "logs/ude_v2/eval/dmd/comparison/ntok_n512_n512_n512_large"
    folders = [
        # "/lora_llama/exp5/animation/t2m/T0000",
        # "/lora_llama/exp5/animation/t2m/T0001",
        # "/lora_llama/exp5/animation/t2m/T0002",
        "/d263_to_d75",
        "/d75_to_d263",
    ]
    crops = [False, False]
    
    files = get_video_files_to_merge(base_path, folders)
    print(files)
    merge_videos(base_path, folders, crops, files, output_path)
    exit(0)
    
    """"""" V2 """""""
    # files = [
    #     "person_is_walking_straight_backward.mp4", 
    #     "person_is_walking_straight_forward.mp4"
    # ]
    
    # merge_videos_v2("logs/ude_v2/eval/ude/orien_pose/ude_clip_mtr_hubert_debug/animation/t2m", 
    #                 files, 
    #                 "logs/ude_v2/eval/ude/orien_pose/ude_clip_mtr_hubert_debug/animation/t2m", 
    #                 "person_is_walking_straight.mp4")
    
    """"""" v3 """""""
    # video_paths = [
    #     "logs/ude_v2/eval/ude/debug/animation/t2m/T000/gLH_sBM_cAll_d17_mLH4_ch02_0.mp4", 
    #     "logs/ude_v2/eval/ude/debug/animation/t2m/T000/gLH_sBM_cAll_d17_mLH4_ch02_1.mp4",
    #     "logs/ude_v2/eval/ude/debug/animation/t2m/T000/gLH_sBM_cAll_d17_mLH4_ch02_2.mp4",
    #     "logs/ude_v2/eval/ude/debug/animation/t2m/T000/gLH_sBM_cAll_d17_mLH4_ch02_3.mp4",
    # ]
    # crops = [False, False, False, False]
    # merge_videos_v3(video_paths, crops, 
    #                 "logs/ude_v2/eval/ude/debug/animation/t2m/compare/merged.mp4")
    