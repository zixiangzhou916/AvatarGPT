import os
import sys
sys.path.append(os.getcwd())
import glob
import imageio
import numpy as np
from tqdm import tqdm

def img2vid(output_name, files, input_dir):
    print('--->', output_name)
    with imageio.get_writer(output_name, fps=30) as writer:
        for file in tqdm(files, desc="Generating MP4..."):
            img_path = os.path.join(input_dir, file)
            try:
                img = np.array(imageio.imread(img_path), dtype=np.uint8)
                success = True
            except:
                success = False
                pass
            if success:
                writer.append_data(img[..., :3])

if __name__ == "__main__":
    input_dir = "../dataset/EgoBody/"
    output_dir = "../HumanPoseEstimation/VisualLLM/Ask-Anything/example/ego_centric/raw"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    valid_frame_files = glob.glob(os.path.join(input_dir, "egocentric_color/*/*/*.npz"))

    for i, file in enumerate(valid_frame_files):
        data = np.load(file)
        valid_frame = data["imgname"]
        vid_name = valid_frame[0].split("/")[1]
        output_name = os.path.join(output_dir, vid_name+".mp4")
        if os.path.exists(output_name):
            continue
        print("[{:d}/{:d}] processed".format(i+1, len(valid_frame_files)))
        img2vid(output_name=output_name, files=valid_frame, input_dir=input_dir)
        # exit(0)