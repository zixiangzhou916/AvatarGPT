import os, sys, argparse
sys.path.append(os.getcwd())
import numpy as np
import imageio
import json
import cv2
from tqdm import tqdm

def reorg_text(text, max_len=15):
    words = text.split(" ")
    num_words = len(words)
    text_reorg = []
    for i in range(0, num_words, 15):
        text_reorg.append(" ".join(words[i:i+15]))
    return "\n".join(text_reorg)

def crop_motion_segment(labels):
    segments = []
    cur_segment = [0]
    for i in range(1, len(labels), 1):
        if labels[i-1] == 1 and labels[i] == 1:
            cur_segment.append(i)
        if labels[i-1] == 1 and labels[i] == 0:
            segments.append(cur_segment)
        if labels[i-1] == 0 and labels[i] == 1:
            cur_segment = [i]
    if len(cur_segment) != 0:
        segments.append(cur_segment)       
    return segments

def plot_text(text):
    # print(len(text.split(" ")))
    lines = text.split("\n")
    image = 255 * np.ones((200, 960, 3))
    # position = (20, 20)
    font_size = 0.5
    font_color = (0, 0, 0, 255)
    font_stroke = 1
    for id, line in enumerate(lines):
        position = (20, 20+id*20)
        image = cv2.putText(image, line, position, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_stroke)
    # cv2.imwrite("text.png", image)
    return image
    
if __name__ == "__main__":
    files = [f for f in os.listdir("logs/avatar_gpt/eval/flan_t5_large/exp7/meshes/se1_p0") if ".npy" in f]
    # file_name = "B0180_T0000.npy"
    input_dir = "logs/avatar_gpt/eval/flan_t5_large/exp7/output/se1_p0/"
    output_dir = "logs/avatar_gpt/eval/flan_t5_large/exp7/captions/se1_p0/"
    os.makedirs(output_dir, exist_ok=True)
    
    with open("logs/avatar_gpt/eval/flan_t5_large/exp7/output/planning_se1_p0.json", "r") as f:
        json_data = json.load(f)
        
    for file_name in files:
        
        # if "B1320_T0000" not in file_name: continue
        text_data = json_data[file_name]
        scene_description, task_description, step_description = [], [], []
        for scene, tasks in text_data.items():
            scene_description.append(scene)
            for task, steps in tasks.items():
                task_description.append(task)
                step_description += steps

        # print(scene_description)
        # print(task_description)
        # print(step_description)

        data = np.load(input_dir+file_name, allow_pickle=True).item()
        color_labels = data.get("color_labels", None)
        captions = data["caption"][0]
        captions = captions.replace(".", "").replace("/", "_")
        words = captions.split(" ")
        name = "_".join(words[:20])
        segments = crop_motion_segment(labels=color_labels)

        if os.path.exists(output_dir + name+".mp4"):
            continue
        
        all_frames = [None for _ in range(len(color_labels))]
        for i, seg in enumerate(segments):
            task_id = i // 5
            step_id = i
            for j in seg:
                all_frames[j] = {"task_id": task_id, "step_id": step_id}

        images = []
        for frame_info in tqdm(all_frames):
            if frame_info is not None:
                scene_text = reorg_text(text=scene_description[0], max_len=15)
                task_text = reorg_text(text=task_description[frame_info["task_id"]], max_len=15)
                step_text = reorg_text(text=step_description[frame_info["step_id"]], max_len=15)

                final_text = "[Scene] {:s}\n[Task] {:s}\n[Step] {:s}".format(scene_text, task_text, step_text)    
                image = plot_text(final_text)
                images.append(image)
            else:
                image = 255 * np.ones_like(images[-1])
                images.append(image)
        images = np.stack(images)
        images = images.astype(np.uint8)
        imageio.mimsave(output_dir + name+".mp4", images, fps=20)
    
