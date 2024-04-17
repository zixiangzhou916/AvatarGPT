import os, sys, argparse
sys.path.append(os.getcwd())
import json
from tqdm import tqdm
import numpy as np
import torch
from transformers import LlamaTokenizer

class Preprocessor(object):
    def __init__(self, args):
        
        self.specific_token_len = args.specific_token_len
        self.raw_caption_dir = args.raw_caption_dir
        self.processed_caption_dir = args.processed_caption_dir
        self.tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer)
    
    @staticmethod
    def write_splitting_file(file_path, file_list):
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                for file in file_list:
                    f.write("{:s}\n".format(file))
        else:
            with open(file_path, "a") as f:
                for file in file_list:
                    f.write("{:s}\n".format(file))
    
    def __run(self):
        raw_files = [f for f in os.listdir(self.raw_caption_dir) if ".json" in f]
        train_split = []
        val_split = []
        test_split = []
        for file in tqdm(raw_files):
            raw_file_path = os.path.join(self.raw_caption_dir, file)
            proc_file_path = os.path.join(self.processed_caption_dir, file)
            if os.path.exists(proc_file_path): 
                continue
                        
            with open(os.path.join(self.raw_caption_dir, file), "r") as f:
                caption_list = json.load(f)
                
            proc_caption_list = [
                {
                    "video_name": os.path.split(caption_list["video_name"])[-1], 
                    "video_fps": caption_list["video_fps"], 
                }
            ]
            
            if len(caption_list["caption_list"]) < 2: 
                print("Skip the one very short video.")
                continue
            
            # Preprocess the per-segment captionings
            for caption_info in caption_list["caption_list"]:
                captions = caption_info["captions"]
                proc_caption_info = {
                    "start_frame": caption_info["start_frame"] * 2, 
                    "end_frame": caption_info["end_frame"] * 2, 
                    "start_time": caption_info["start_time"] * 2, 
                    "end_time": caption_info["end_time"] * 2, 
                    "captions": []
                }
                # Filter out invalid captions
                for cap in captions:
                    # 1. Filter out the captions containing '?'
                    if "?" in cap: 
                        continue
                    # 2. Filter out the captions longer than specific value (after tokenization)
                    tokenization_output = self.tokenizer(cap)
                    if len(tokenization_output.input_ids) > self.specific_token_len or len(tokenization_output.input_ids) <= 5:
                        continue
                    proc_caption_info["captions"].append(cap)
                
                if len(proc_caption_info["captions"]) == 0: 
                    continue
                
                proc_caption_list.append(proc_caption_info)
            
            # Preprocess the whole scene description
            proc_caption_list[0]["video_scene"] = []
            for desc in caption_list["description"]:
                # 1. Filter out the captions containing '?'
                if "?" in desc: 
                    continue
                # 2. Filter out the captions longer than specific value (after tokenization)
                tokenization_output = self.tokenizer(desc)
                if len(tokenization_output.input_ids) > self.specific_token_len or len(tokenization_output.input_ids) <= 5:
                    continue
                proc_caption_list[0]["video_scene"].append(desc)
                
            with open(os.path.join(self.processed_caption_dir, file), "w") as f:
                json.dump(proc_caption_list, f)
                
            rnd = np.random.random()
            if rnd < 0.9:
                train_split.append(file.replace(".json", ""))
            else:
                val_split.append(file.replace(".json", ""))
                test_split.append(file.replace(".json", ""))
                
        return train_split, val_split, test_split
    
    def __run_decomposed(self):
        
        if not os.path.exists(self.processed_caption_dir):
            os.makedirs(self.processed_caption_dir)
        
        raw_files = [f for f in os.listdir(self.raw_caption_dir) if ".json" in f]
        sum_files = [f for f in os.listdir("../dataset/VideoCaptionData_chatgpt/summarized") if ".json" in f]
        train_split = []
        val_split = []
        test_split = []
        for file in tqdm(raw_files):
            raw_file_path = os.path.join(self.raw_caption_dir, file)
            sum_file_path = os.path.join("../dataset/VideoCaptionData_chatgpt/summarized", file)
            proc_file_path = os.path.join(self.processed_caption_dir, file)
            
            if not os.path.exists(sum_file_path):
                continue
            
            if os.path.exists(proc_file_path):
                continue
            
            try:
                with open(raw_file_path, "r") as f:
                    raw_caption_list = json.load(f)
                with open(sum_file_path, "r") as f:
                    sum_caption_list = json.load(f)
            except:
                print("Unable to load {:s} or {:s}".format(raw_file_path, sum_file_path))
                continue
            
            proc_caption_list = {
                "video_name": os.path.split(raw_caption_list["video_name"])[-1], 
                "video_fps": raw_caption_list["video_fps"], 
            }
            
            if len(raw_caption_list["caption_list"]) < 2:
                print("Skip the one corresponding to a very short video.")
                continue
            
            # Preprocess the per-segment captions
            proc_caption_list["contexts"] = sum_caption_list["description"]
            proc_caption_list["actions"] = []
            for raw_info in raw_caption_list["caption_list"]:
                output_info = {
                    "start_time": raw_info["start_time"], 
                    "end_time": raw_info["end_time"], 
                    "start_frame": raw_info["start_frame"], 
                    "end_frame": raw_info["end_frame"], 
                    "captions": []
                }
                
                for caps in raw_info["captions"]:
                    if isinstance(caps, dict):
                        caps_info = {
                            "detail": caps["detail"], 
                            "executable_steps": caps["executable"], 
                            "executable_simplified": []
                        }
                        for sim in caps["simplified"]:
                            if sim is None: continue
                            caps_info["executable_simplified"].append(sim)
                        if len(caps_info["executable_simplified"]) > 0:
                            output_info["captions"].append(caps_info)
                
                if len(output_info["captions"]) > 0:
                    proc_caption_list["actions"].append(output_info)
            
            # if len(proc_caption_list["actions"]) == len(proc_caption_list["contexts"]):
            with open(proc_file_path, "w") as f:
                json.dump(proc_caption_list, f)

            rnd = np.random.random()
            if rnd < 0.9:
                train_split.append(file.replace(".json", ""))
            else:
                val_split.append(file.replace(".json", ""))
                test_split.append(file.replace(".json", ""))
                
        return train_split, val_split, test_split
    
    
    def run(self):
        
        # train_split, val_split, test_split = self.__run()
        train_split, val_split, test_split = self.__run_decomposed()
        
        # split_base_dir = os.path.split(self.raw_caption_dir)[0]
        # self.write_splitting_file(os.path.join(split_base_dir, "train.txt"), train_split)
        # self.write_splitting_file(os.path.join(split_base_dir, "val.txt"), val_split)
        # self.write_splitting_file(os.path.join(split_base_dir, "test.txt"), test_split)

def append_splits(processed_dir, train_split_dir, val_split_dir):
    
    def read_from_text(input_dir):
        data = []
        with open(input_dir, "r") as f:
            lines = f.readlines()
            for line in lines:
                data.append(line.strip())
        return data
    
    def write_to_text(output_file, output_dir):
        Preprocessor.write_splitting_file(file_path=output_dir, file_list=output_file) 
    
    train_split_data = read_from_text(train_split_dir)
    val_split_data = read_from_text(val_split_dir)
    split_data = train_split_data + val_split_data
    
    files = [f.replace(".json", "") for f in os.listdir(processed_dir) if ".json" in f]
    train_split, val_split, test_split = [], [], []
    for file in tqdm(files):
        if file in split_data:
            continue
        rnd = np.random.random()
        if rnd < 0.9:
            train_split.append(file)
        else:
            val_split.append(file)
            test_split.append(file)
    
    write_to_text(train_split, train_split_dir)
    write_to_text(val_split, val_split_dir)
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, 
                        default="networks/llama/openlm-research/open_llama_13b/pretrained", 
                        help="Directory of tokenizer")
    parser.add_argument("--raw_caption_dir", type=str, 
                        default="../dataset/VideoCaptionData_chatgpt/summarized_decompose", 
                        help="Directory of raw video captioning files")
    parser.add_argument("--processed_caption_dir", type=str, 
                        default="../dataset/VideoCaptionData_chatgpt/processed_decompose", 
                        help="Directory of processed video captioning files")
    parser.add_argument("--specific_token_len", type=int, default=500,  
                        help="Threshold of caption length, we keep the captions with tokenized lengths shorter than this value.")
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    
    args = parse_args()
    processor = Preprocessor(args=args)
    processor.run()
    
    append_splits(
        args.processed_caption_dir, 
        "../dataset/VideoCaptionData_chatgpt/train.txt", 
        "../dataset/VideoCaptionData_chatgpt/val.txt")