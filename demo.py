import os
import codecs as cs
import shutil
import tqdm

if __name__ == "__main__":
    lines = []
    with cs.open("/cpfs/user/zhouzixiang/projects/dataset/VideoCaptionData_chatgpt/test.txt", "r") as f:
        for line in f.readlines():
            lines.append(line.strip())
        
    base_dir = "/cpfs/user/zhouzixiang/projects/dataset/VideoCaptionData_chatgpt/processed_llm_context"
    for idx, file in enumerate(tqdm.tqdm(lines)):
        src_file = f"{base_dir}/{file}.json"
        dst_file = f"demo_inputs/{idx}.json"
        shutil.copy(src_file, dst_file)