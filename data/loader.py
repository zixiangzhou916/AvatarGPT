import os, sys
sys.path.append(os.getcwd())
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import torchaudio
import torchaudio.functional as FF
import numpy as np
import os
import random
from os.path import join as pjoin
import random
import json
import codecs as cs
from tqdm import tqdm
from scipy import ndimage
import importlib

SPEAKER_ID = {
    "wayne": 0, "kieks": 1, "nidal": 2, "zhao": 3, "lu": 4,
    "zhang": 5, "carlos": 6, "jorge": 7, "itoi": 8, "daiki": 9,
    "jaime": 10, "scott": 11, "li": 12, "ayana": 13, "luqi": 14,
    "hailing": 15, "kexin": 16, "goto": 17, "reamey": 18, "yingqing": 19,
    "tiffnay": 20, "hanieh": 21, "solomon": 22, "katya": 23, "lawrence": 24,
    "stewart": 25, "carla": 26, "sophie": 27, "catherine": 28, "miranda": 29, 
    "ChenShuiRuo": 0
}

SEMANTIC_ID = {
    "deictic_l": 0, "metaphoric_m": 1, "iconic_h": 2, "metaphoric_l": 3,
    "beat_align": 4, "metaphoric_h": 5, "deictic_h": 6, "iconic_m": 7,
    "nogesture": 8, "deictic_m": 9, "need_cut": 10, "iconic_l": 11, "habit": 12 
}

EMOTION_ID = [0, 1, 2, 3, 4, 5, 6, 7]

RECORDING_TYPE = {
    0: "English Speech", 
    1: "English Conversation", 
    2: "Chinese Speech", 
    3: "Chinese Conversation", 
    4: "Spanish Speech", 
    5: "Spanish Conversation", 
    6: "Japanese Speech", 
    7: "Japanese Conversation"
}

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def normalize_trans(motion):
    """
    :param motion: [num_frames, num_dims]
    """
    glob_trans = motion[:, :3]  # [num_frames, 3]
    avg_trans = np.mean(glob_trans, axis=0, keepdims=True)  # [1, 3]
    motion[:, :3] -= avg_trans
    return motion

"""Following Dataset objects are designed for GPT."""
class AvatarGPTDataset(data.Dataset):
    def __init__(
        self, opt, 
        t2m_split_file=None, 
        a2m_split_file=None, 
        t2t_split_file=None, 
        meta_dir=None
    ):
        super(AvatarGPTDataset, self).__init__()
        self.opt = opt
        self.times = opt["times"]
        self.variable_lengths = self.opt.get("variable_lengths", True)
        
        id_lists = {}
        # Text-to-Motion
        if "t2m" in self.opt["modality"]:
            id_lists["t2m"] = []
            with cs.open(t2m_split_file, "r") as f:
                for line in f.readlines():
                    id_lists["t2m"].append(line.strip())
                
        # Audio-to-Motion
        if "a2m" in self.opt["modality"]:
            id_lists["a2m"] = []
            with cs.open(a2m_split_file, "r") as f:
                for line in f.readlines():
                    id_lists["a2m"].append(line.strip())
                    
        # Text-to-Text
        if "t2t" in self.opt["modality"]:
            id_lists["t2t"] = []
            with cs.open(t2t_split_file, "r") as f:
                for line in f.readlines():
                    id_lists["t2t"].append(line.strip())
                    
        """ Debug, we only load very tiny part of the dataset. """
        # debug_size = {"t2m": 1000, "a2m": 50, "t2t": 100}
        # # debug_size = {"t2m": 980}
        # for k, item in id_lists.items():
        #     if len(item) == 0: continue
        #     # item_ = item
        #     # random.shuffle(item)    # Random shuffle
        #     id_lists[k] = item[:debug_size[k]]
        """ End of debug """
        
        # Load all dataset
        self.data_dict = {}
        self.name_list = {}
        self.modality = []
        for key, id_list in id_lists.items():
            if "t2m" == key:
                self.data_dict["t2m"], self.name_list["t2m"] = self.read_all_t2m_data(id_list)
            elif "a2m" == key:
                self.data_dict["a2m"], self.name_list["a2m"] = self.read_all_a2m_data(id_list)
            elif "t2t" == key:
                self.data_dict["t2t"], self.name_list["t2t"] = self.read_all_t2t_data(id_list)
            
            self.modality += [key] * len(self.data_dict[key])
        # Make sure they have the same lenght, if not, we duplicate the shorter ones
        max_length = np.max([len(item) for _, item in self.data_dict.items()])    
        self.num_data = max_length

    def __len__(self):
        return self.num_data * self.times
    
    def __getitem__(self, item):
        
        batch = {}
        for key in self.data_dict.keys():
            if "t2m" in key:
                batch[key] = self.read_one_t2m_data(key, item)
            elif "a2m" in key:
                batch[key] = self.read_one_a2m_data(key, item)
            elif "t2t" in key:
                batch[key] = self.read_one_t2t_data(key, item)
        return batch
     
    @staticmethod
    def parse_part_of_speech(t_tokens):
        part_of_speechs = []
        for token in t_tokens:
            pos = token.split("/")[1]
            part_of_speechs.append(pos)
        return part_of_speechs
    
    def read_all_t2m_data(self, id_list):
        new_name_list = []
        data_dict = {}
        self.all_lengths = []    # record the lengths of every HumanML3D data sample
        for name in tqdm(id_list, desc="Reading text-to-motion dataset"):
            try:
                body = np.load(pjoin(self.opt["t2m_motion_dir"], name+".npy"))
                # Skip the motion sequence with very short length
                if body.shape[0] < self.opt["window_size"]["t2m"] // 4:
                    continue
                
                with cs.open(pjoin(self.opt["text_dir"], name+".txt")) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()
                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            part_of_speechs = self.parse_part_of_speech(t_tokens)
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag
                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            text_dict['part_of_speech'] = part_of_speechs
                            # print(text_dict)
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)
                                data_dict[new_name] = {'text':[text_dict], "body": body}
                                new_name_list.append(new_name)
                                self.all_lengths.append(body.shape[0])
                        except:
                            pass
                
                if flag:
                    data_dict[name] = {'text': text_data, 'body': body}
                    new_name_list.append(name)
                    self.all_lengths.append(body.shape[0])
            
            except:
                pass
        
        print('---', len(data_dict), "|", len(new_name_list))
        return data_dict, new_name_list
    
    def read_all_a2m_data(self, id_list):
        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list, desc="Reading audio-to-motion dataset"):
            try:
                data = np.load(pjoin(self.opt["a2m_motion_dir"], name+".npy"), allow_pickle=True).item()
                body = data["motion_smpl"][::2] # downsample to fps=30
                audio = data["audio_sequence"]
                duration = data["duration"]
                data_dict[name] = {"audio": audio, "body": body, "name": name, "duration": duration}
                new_name_list.append(name)
            except:
                pass
        return data_dict, new_name_list
    
    def read_all_t2t_data(self, id_list):
        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list, desc="Reading text-to-tet dataset"):
            try:
                with open(pjoin(self.opt["t2t_text_dir"], name+".json"), "r") as f:
                    json_data = json.load(f)
                    
                    video_scene = json_data[0]["video_scene"]
                    video_fps = json_data[0]["video_fps"]
                    
                    for action_id in range(1, len(json_data)-1):
                        duration = (json_data[action_id]["end_frame"] - json_data[action_id]["start_frame"]) / video_fps
                        cur_actions = json_data[action_id]["captions"]
                        next_actions = json_data[action_id+1]["captions"]
                        next_duration = (json_data[action_id+1]["end_frame"] - json_data[action_id+1]["start_frame"]) / video_fps
                    
                        data_dict["{:s}_{:d}".format(name, action_id)] = {
                            "video_scene": video_scene, 
                            "duration": duration, 
                            "cur_actions": cur_actions, 
                            "next_actions": next_actions, 
                            "next_duration": next_duration
                        }
                        new_name_list.append("{:s}_{:d}".format(name, action_id))
            except:
                pass
        return data_dict, new_name_list
    
    def read_one_t2m_data(self, key, item):
        if item < len(self.name_list[key]):
            index = item
        else:
            index = item % len(self.name_list[key])
        # print('---', item, index, len(self.name_list[key]))
        name = self.name_list[key][index]
        body = self.data_dict[key][name]["body"]
        text_list = self.data_dict[key][name]["text"]
        text_data = random.choice(text_list)    # Random sample one text
        caption = text_data["caption"]
        t_tokens = text_data["tokens"]
        part_of_speech = "_".join(text_data["part_of_speech"])
        mot_len = body.shape[0]
        
        window_size = self.opt["window_size"]["t2m"]
        if mot_len > window_size:
            i = np.random.randint(0, mot_len-window_size)
            j = i + window_size
            body = body[i:j]
            mot_len = body.shape[0]
            
        # window_size = np.max(np.array(self.all_lengths)) + 8    # We would at least apped 2 <EOS> per sequence.
        pad_len = window_size - mot_len
        pad_body = np.zeros((pad_len, body.shape[1]), dtype=body.dtype)
        body = np.concatenate([body, pad_body], axis=0)
        body_length = mot_len
        
        # Get word tokens from Vocab
        if len(t_tokens) < self.opt["t2m_max_text_len"]:
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
            t_tokens += ['unk/OTHER'] * (self.opt["t2m_max_text_len"] + 2 - sent_len)
        else:
            t_tokens = t_tokens[:self.opt["t2m_max_text_len"]]
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
        
        batch = {"body": body, "text": caption, "part_of_speech": "_".join(t_tokens), "length": body_length}
        return batch

    def read_one_a2m_data(self, key, item):
        if item < len(self.name_list[key]):
            index = item
        else:
            index = item % len(self.name_list[key])
        
        name = self.name_list[key][index]
        body = self.data_dict[key][name]["body"] # The motion has already been downsampled to fps=30
        audio = self.data_dict[key][name]["audio"] # The fps of audio is 60 by default, and we don't modify it.
        duration = self.data_dict[key][name]["duration"]
        mot_len = body.shape[0]
        if mot_len > self.opt["window_size"][key]:
            m_start_idx = np.random.randint(0, mot_len - self.opt["window_size"][key])
            m_end_idx = m_start_idx + self.opt["window_size"][key]
            a_start_idx = int(m_start_idx / 30 * 16000)
            a_end_idx = a_start_idx + int(self.opt["window_size"][key] / 30 * 16000)
            # print(m_start_idx, m_end_idx, a_start_idx, a_end_idx)
            body = body[m_start_idx:m_end_idx]
            audio = audio[a_start_idx:a_end_idx]
        else:
            raise ValueError("Sequence length is not long enough!")
        
        batch = {"body": body, "audio": audio, "name": "{:s}_{:d}_{:d}".format(name, a_start_idx, a_end_idx)}
        return batch
    
    def read_one_t2t_data(self, key, item):
        if item < len(self.name_list[key]):
            index = item
        else:
            index = item % len(self.name_list[key])
            
        name = self.name_list[key][index]
        json_data = self.data_dict[key][name]
        scene = random.choice(json_data["video_scene"])
        duration = json_data["duration"]
        next_duration = json_data["next_duration"]
        cur_action = random.choice(json_data["cur_actions"])
        next_action = random.choice(json_data["next_actions"])
        
        batch = {
            "scene": "[scene] {:s}".format(scene), 
            "duration": "this action lasts {:.1f} seconds".format(duration), 
            "cur_action": "[current action]: {:s}".format(cur_action), 
            "next_action": next_action, 
            "next_duration": "this action lasts {:.1f} seconds".format(next_duration)
        }
        return batch

class AvatarGPTHMLDataset(data.Dataset):
    def __init__(
        self, opt, 
        t2m_split_file=None, 
        a2m_split_file=None, 
        t2t_split_file=None, 
        meta_dir=None
    ):
        super(AvatarGPTHMLDataset, self).__init__()
        self.opt = opt
        self.meta_dir = meta_dir
        self.times = opt["times"]
        self.variable_lengths = self.opt.get("variable_lengths", True)
        self.read_mirrors = self.opt.get("read_mirrors", True)
        self.read_augmented_texts = self.opt.get("read_augmented_texts", False)
        if self.read_augmented_texts:
            with open("data/humanml3d_augmented_texts.json", "r") as f:
                self.augmented_texts = json.load(f)
        self.setup_stats()
        
        id_lists = {}
        # Text-to-Motion
        if "t2m" in self.opt["modality"]:
            id_lists["t2m"] = []
            with cs.open(t2m_split_file, "r") as f:
                for line in f.readlines():
                    if not self.read_mirrors and "M" in line.strip():
                        continue
                    id_lists["t2m"].append(line.strip())
                
        # Audio-to-Motion
        if "a2m" in self.opt["modality"]:
            id_lists["a2m"] = []
            with cs.open(a2m_split_file, "r") as f:
                for line in f.readlines():
                    id_lists["a2m"].append(line.strip())
                    
        # Text-to-Text
        if "t2t" in self.opt["modality"]:
            id_lists["t2t"] = []
            with cs.open(t2t_split_file, "r") as f:
                for line in f.readlines():
                    id_lists["t2t"].append(line.strip())
        
        # Load all dataset
        self.data_dict = {}
        self.name_list = {}
        self.modality = []
        for key, id_list in id_lists.items():
            if "t2m" == key:
                self.data_dict["t2m"], self.name_list["t2m"] = self.read_all_t2m_data(id_list)
            elif "a2m" == key:
                self.data_dict["a2m"], self.name_list["a2m"] = self.read_all_a2m_data(id_list)
            elif "t2t" == key:
                self.data_dict["t2t"], self.name_list["t2t"] = self.read_all_t2t_data(id_list)
            
            self.modality += [key] * len(self.data_dict[key])
        # Make sure they have the same lenght, if not, we duplicate the shorter ones
        max_length = np.max([len(item) for _, item in self.data_dict.items()])    
        self.num_data = max_length
        
    def setup_stats(self):
        
        if self.opt.get("meta_mode", "tm2t") == "tm2t":
            std = np.load(self.opt.get("std_dir", "data/Std.npy"))
            mean = np.load(self.opt.get("mean_dir", "data/Mean.npy"))
            feat_bias = self.opt.get("feat_bias", 5)
            joints_num = self.opt.get("joints_num", 22)

            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4:4+(joints_num-1)*3] = std[4:4+(joints_num-1)*3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4+(joints_num-1)*3:4+(joints_num-1)*9] = std[4+(joints_num-1)*3:4+(joints_num-1)*9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4+(joints_num-1)*9:4+(joints_num-1)*9+joints_num*3] = std[4+(joints_num-1)*9:4+(joints_num-1)*9+joints_num*3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4+(joints_num-1)*9+joints_num*3:] = std[4+(joints_num-1)*9+joints_num*3:] / feat_bias
            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
        elif self.opt.get("meta_mode", "tm2t") == "t2m-gpt":
            std = np.load(self.opt.get("std_dir", "pretrained_models/vqvae/std.npy"))
            mean = np.load(self.opt.get("mean_dir", "pretrained_models/vqvae/mean.npy"))
            
        if self.meta_dir is not None:
            np.save(pjoin(self.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(self.meta_dir, 'std.npy'), std)
        
        self.mean = mean
        self.std = std
    
    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def __len__(self):
        return self.num_data * self.times
    
    def __getitem__(self, item):
        
        batch = {}
        for key in self.data_dict.keys():
            if "t2m" in key:
                batch[key] = self.read_one_t2m_data(key, item)
            elif "a2m" in key:
                batch[key] = self.read_one_a2m_data(key, item)
            elif "t2t" in key:
                batch[key] = self.read_one_t2t_data(key, item)
        return batch
    
    @staticmethod
    def parse_part_of_speech(t_tokens):
        part_of_speechs = []
        for token in t_tokens:
            pos = token.split("/")[1]
            part_of_speechs.append(pos)
        return part_of_speechs
    
    def read_all_t2m_data(self, id_list):
        new_name_list = []
        data_dict = {}
        self.all_lengths = []    # record the lengths of every HumanML3D data sample
        for name in tqdm(id_list, desc="Reading text-to-motion dataset"):
            try:
                body = np.load(pjoin(self.opt["t2m_motion_dir"], name+".npy"))
                # Skip the motion sequence with very short length
                if body.shape[0] < self.opt["window_size"]["t2m"] // 4:
                    continue
                
                if self.read_augmented_texts and name+".txt" in self.augmented_texts.keys():
                    augmented_texts = self.augmented_texts[name+".txt"]
                else:
                    augmented_texts = []    # Empty augmented texts list
                
                with cs.open(pjoin(self.opt["text_dir"], name+".txt")) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()
                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            part_of_speechs = self.parse_part_of_speech(t_tokens)
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag
                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            text_dict['part_of_speech'] = part_of_speechs
                            # print(text_dict)
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)
                                data_dict[new_name] = {'text':[text_dict], "body": body, "augmented_text": augmented_texts}
                                new_name_list.append(new_name)
                                self.all_lengths.append(body.shape[0])
                        except:
                            pass
                
                if flag:
                    data_dict[name] = {'text': text_data, 'body': body, "augmented_text": augmented_texts}
                    new_name_list.append(name)
                    self.all_lengths.append(body.shape[0])
            
            except:
                pass
        
        print('---', len(data_dict), "|", len(new_name_list))
        return data_dict, new_name_list
    
    def read_all_a2m_data(self, id_list):
        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list, desc="Reading audio-to-motion dataset"):
            try:
                data = np.load(pjoin(self.opt["a2m_motion_dir"], name+".npy"), allow_pickle=True).item()
                body = data["motion_smpl"][::2] # downsample to fps=30
                audio = data["audio_sequence"]
                duration = data["duration"]
                data_dict[name] = {"audio": audio, "body": body, "name": name, "duration": duration}
                new_name_list.append(name)
            except:
                pass
        return data_dict, new_name_list
    
    def read_all_t2t_data(self, id_list):
        if not self.opt.get("decompose_actions", False):
            return self.__read_all_t2t_data(id_list=id_list)
        else:
            return self.__read_all_t2t_data_decomposed(id_list=id_list)
    
    def __read_all_t2t_data(self, id_list):
        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list, desc="Reading text-to-text dataset"):
            try:
                with open(pjoin(self.opt["t2t_text_dir"], name+".json"), "r") as f:
                    json_data = json.load(f)
                    
                    video_scene = json_data[0]["video_scene"]
                    video_fps = json_data[0]["video_fps"]
                    
                    for action_id in range(1, len(json_data)-1):
                        duration = (json_data[action_id]["end_frame"] - json_data[action_id]["start_frame"]) / video_fps
                        cur_actions = json_data[action_id]["captions"]
                        next_actions = json_data[action_id+1]["captions"]
                        next_duration = (json_data[action_id+1]["end_frame"] - json_data[action_id+1]["start_frame"]) / video_fps
                    
                        data_dict["{:s}_{:d}".format(name, action_id)] = {
                            "video_scene": video_scene, 
                            "duration": duration, 
                            "cur_actions": cur_actions, 
                            "next_actions": next_actions, 
                            "next_duration": next_duration
                        }
                        new_name_list.append("{:s}_{:d}".format(name, action_id))
            except:
                pass
        return data_dict, new_name_list
    
    def __read_all_t2t_data_decomposed(self, id_list):
        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list, desc="Reading text-to-text dataset"):
            try:
                with open(pjoin(self.opt["t2t_text_dir"], name+".json"), "r") as f:
                    json_data = json.load(f)
                
                video_scene = json_data["contexts"]
                video_fps = json_data["video_fps"]
                
                action_data = json_data["actions"]
                for action_id in range(len(action_data) - 1):
                    cur_duration = (action_data[action_id]["end_frame"] - action_data[action_id]["start_frame"]) / video_fps
                    cur_actions = action_data[action_id]["captions"]
                    next_duration = (action_data[action_id+1]["end_frame"] - action_data[action_id+1]["start_frame"]) / video_fps
                    next_actions = action_data[action_id+1]["captions"]
                    
                    data_dict["{:s}_{:d}".format(name, action_id)] = {
                        "video_scene": video_scene, 
                        "duration": cur_duration, 
                        "next_duration": next_duration, 
                        "cur_actions": cur_actions, 
                        "next_actions": next_actions
                    }
                    new_name_list.append("{:s}_{:d}".format(name, action_id))
                    
            except:
                pass
        return data_dict, new_name_list        
        
    def read_one_t2m_data(self, key, item):
        if item < len(self.name_list[key]):
            index = item
        else:
            index = item % len(self.name_list[key])
        # print('---', item, index, len(self.name_list[key]))
        name = self.name_list[key][index]
        body = self.data_dict[key][name]["body"]
        text_list = self.data_dict[key][name]["text"]
        augmented_texts = self.data_dict[key][name]["augmented_text"]
        if len(augmented_texts) == 0:
            aug_caption = random.choice(text_list)["caption"]
        else:
            aug_caption = random.choice(augmented_texts)
        text_data = random.choice(text_list)    # Random sample one text
        caption = text_data["caption"]
        t_tokens = text_data["tokens"]
        part_of_speech = "_".join(text_data["part_of_speech"])
        mot_len = body.shape[0]
        
        window_size = self.opt["window_size"]["t2m"]
        if mot_len > window_size:
            i = np.random.randint(0, mot_len-window_size)
            j = i + window_size
            body = body[i:j]
            mot_len = body.shape[0]
            
        # window_size = np.max(np.array(self.all_lengths)) + 8    # We would at least apped 2 <EOS> per sequence.
        pad_len = window_size - mot_len
        pad_body = np.zeros((pad_len, body.shape[1]), dtype=body.dtype)
        body = np.concatenate([body, pad_body], axis=0)
        body_length = mot_len
        
        # Get word tokens from Vocab
        if len(t_tokens) < self.opt["t2m_max_text_len"]:
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
            t_tokens += ['unk/OTHER'] * (self.opt["t2m_max_text_len"] + 2 - sent_len)
        else:
            t_tokens = t_tokens[:self.opt["t2m_max_text_len"]]
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
        
        body = (body - self.mean) / self.std
        batch = {"body": body, "text": caption, "part_of_speech": "_".join(t_tokens), "length": body_length, "aug_text": aug_caption}
        return batch

    def read_one_a2m_data(self, key, item):
        if item < len(self.name_list[key]):
            index = item
        else:
            index = item % len(self.name_list[key])
        
        name = self.name_list[key][index]
        body = self.data_dict[key][name]["body"] # The motion has already been downsampled to fps=30
        audio = self.data_dict[key][name]["audio"] # The fps of audio is 60 by default, and we don't modify it.
        duration = self.data_dict[key][name]["duration"]
        mot_len = body.shape[0]
        if mot_len > self.opt["window_size"][key]:
            m_start_idx = np.random.randint(0, mot_len - self.opt["window_size"][key])
            m_end_idx = m_start_idx + self.opt["window_size"][key]
            a_start_idx = int(m_start_idx / 30 * 16000)
            a_end_idx = a_start_idx + int(self.opt["window_size"][key] / 30 * 16000)
            # print(m_start_idx, m_end_idx, a_start_idx, a_end_idx)
            body = body[m_start_idx:m_end_idx]
            audio = audio[a_start_idx:a_end_idx]
        else:
            raise ValueError("Sequence length is not long enough!")
        
        batch = {"body": body, "audio": audio, "name": "{:s}_{:d}_{:d}".format(name, a_start_idx, a_end_idx)}
        return batch
    
    def read_one_t2t_data(self, key, item):
        if not self.opt.get("decompose_actions", False):
            return self.__read_one_t2t_data(key=key, item=item)
        else:
            return self.__read_one_t2t_data_decomposed(key=key, item=item)
    
    def __read_one_t2t_data(self, key, item):
        if item < len(self.name_list[key]):
            index = item
        else:
            index = item % len(self.name_list[key])
            
        name = self.name_list[key][index]
        json_data = self.data_dict[key][name]
        scene = random.choice(json_data["video_scene"])
        duration = json_data["duration"]
        next_duration = json_data["next_duration"]
        cur_action = random.choice(json_data["cur_actions"])
        next_action = random.choice(json_data["next_actions"])
        
        batch = {
            "scene": "[scene] {:s}".format(scene), 
            "duration": "this action lasts {:.1f} seconds".format(duration), 
            "cur_action": "[current action]: {:s}".format(cur_action), 
            "next_action": next_action, 
            "next_duration": "this action lasts {:.1f} seconds".format(next_duration)
        }
        return batch
    
    def __read_one_t2t_data_decomposed(self, key, item):
        if item < len(self.name_list[key]):
            index = item
        else:
            index = item % len(self.name_list[key])
        
        name = self.name_list[key][index]
        json_data = self.data_dict[key][name]
        scene = random.choice(json_data["video_scene"])
        cur_duration = json_data["duration"]
        next_duration = json_data["next_duration"]
        cur_action_info = random.choice(json_data["cur_actions"])
        cur_executable_steps = cur_action_info["executable_steps"]
        cur_executable_simplified = random.choice(cur_action_info["executable_simplified"])
        next_action_info = random.choice(json_data["next_actions"])
        next_executable_steps = next_action_info["executable_steps"]    # Bug fix!
        next_executable_simplified = random.choice(next_action_info["executable_simplified"])
        
        batch = {
            "scene": "[scene] {:s}".format(scene), 
            "duration": "this action lasts {:.1f} seconds".format(cur_duration), 
            "next_duration": "this action lasts {:.1f} seconds".format(next_duration), 
            "cur_action": "[current action]: {:s}".format(cur_executable_simplified), 
            "next_action": next_executable_simplified, 
            "cur_steps": cur_executable_steps, 
            "next_steps": next_executable_steps
        }
        return batch
    
class AvatarGPTEvalDataset(data.Dataset):
    def __init__(
        self, opt, 
        t2m_split_file=None, 
        a2m_split_file=None, 
        t2t_split_file=None, 
        meta_dir=None
    ):
        super(AvatarGPTEvalDataset, self).__init__()
        
        self.opt = opt
        self.times = opt["times"]
        
        id_lists = {}
        # Text-to-Motion
        """ Don't load HumanML3D dataset """
        if "t2m" in self.opt["modality"]:
            id_lists["t2m"] = []
            with cs.open(t2m_split_file, "r") as f:
                for line in f.readlines():
                    id_lists["t2m"].append(line.strip())
                
        # Audio-to-Motion
        if "a2m" in self.opt["modality"]:
            id_lists["a2m"] = []
            with cs.open(a2m_split_file, "r") as f:
                for line in f.readlines():
                    id_lists["a2m"].append(line.strip())
                    
        # Text-to-Text
        if "t2t" in self.opt["modality"]:
            id_lists["t2t"] = []
            with cs.open(t2t_split_file, "r") as f:
                for line in f.readlines():
                    id_lists["t2t"].append(line.strip())

        # Load all dataset
        self.data = []
        self.name_list = []
        self.modality = []
        for key, id_list in id_lists.items():
            if "t2m" == key:
                data, name_list = self.read_all_t2m_data(id_list)
                # print('---->', len(data), "|", len(name_list))
            elif "a2m" == key:
                data, name_list = self.read_all_a2m_data(id_list)
                # print('----', len(data), "|", len(name_list))
            elif "t2t" == key:
                data, name_list = self.read_all_t2t_data(id_list)
                # print('----', len(data), "|", len(name_list))
                
            self.data += data
            self.name_list += name_list
            self.modality += [key] * len(data)
        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        valid_test_set = json.load(
            open(os.path.join(dir_path, "ValidTestset_HumanML3D.json"), "r"))
        self.valid_amass_test_set = [t["name"] for t in valid_test_set]
        self.num_data = len(self.data)
        
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, item):
        modality = self.modality[item]
        if modality == "t2m":
            batch = self.read_one_t2m_data(item)
        elif modality == "a2m":
            batch = self.read_one_a2m_data(item)
        elif modality == "t2t":
            batch = self.read_one_t2t_data(item)
        batch.update({"modality": modality})
        return batch
    
    def read_all_t2m_data(self, id_list):
        new_name_list = []
        data_dict = []
        self.all_lengths = []    # record the lengths of every HumanML3D data sample
        for name in tqdm(id_list, desc="Reading text-to-motion dataset"):
            try:
                body = np.load(pjoin(self.opt["t2m_motion_dir"], name+".npy"))
                with cs.open(pjoin(self.opt["text_dir"], name+".txt")) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()
                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            # part_of_speechs = self.parse_part_of_speech(t_tokens)
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag
                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            # text_dict['part_of_speech'] = part_of_speechs
                            
                            # print(text_dict)
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)
                                data_dict.append({'text':[text_dict], "body": body, "name": new_name})
                                new_name_list.append(new_name)
                                self.all_lengths.append(body.shape[0])
                        except:
                            pass
                
                if flag:
                    data_dict.append({'text': text_data, 'body': body, "name": name})
                    new_name_list.append(name)
                    self.all_lengths.append(body.shape[0])
            
            except:
                pass
        
        print('---', len(data_dict), "|", len(new_name_list))
        return data_dict, new_name_list
    
    def read_all_a2m_data(self, id_list):
        new_name_list = []
        data_dict = []
        for name in tqdm(id_list, desc="Reading audio-to-motion dataset"):
            try:
                data = np.load(pjoin(self.opt["a2m_motion_dir"], name+".npy"), allow_pickle=True).item()
                body = data["motion_smpl"][::2] # downsample to fps=30
                audio = data["audio_sequence"]
                duration = data["duration"]
                data_dict.append({"audio": audio, "body": body, "name": name, "duration": duration, "name": name})
                new_name_list.append(name)
            except:
                pass
        return data_dict, new_name_list
    
    def read_all_t2t_data(self, id_list):
        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list, desc="Reading text-to-tet dataset"):
            try:
                with open(pjoin(self.opt["t2t_text_dir"], name+".json"), "r") as f:
                    json_data = json.load(f)
                    
                    video_scene = json_data[0]["video_scene"]
                    video_fps = json_data[0]["video_fps"]
                    
                    for action_id in range(1, len(json_data)-1):
                        duration = (json_data[action_id]["end_frame"] - json_data[action_id]["start_frame"]) / video_fps
                        cur_actions = json_data[action_id]["captions"]
                        next_actions = json_data[action_id+1]["captions"]
                        next_duration = (json_data[action_id+1]["end_frame"] - json_data[action_id+1]["start_frame"]) / video_fps
                    
                        data_dict["{:s}_{:d}".format(name, action_id)] = {
                            "video_scene": video_scene, 
                            "duration": duration, 
                            "cur_actions": cur_actions, 
                            "next_actions": next_actions, 
                            "next_duration": next_duration
                        }
                        new_name_list.append("{:s}_{:d}".format(name, action_id))
            except:
                pass
        return data_dict, new_name_list
    
    def read_one_t2m_data(self, item):
        body = self.data[item]["body"]
        text_list = self.data[item]["text"]
        text_data = random.choice(text_list)    # Random sample one text
        caption = text_data["caption"]
        t_tokens = text_data["tokens"]
        mot_len = body.shape[0]
        window_size = self.opt["window_size"]["t2m"]
        
        if mot_len > window_size:
            i = np.random.randint(0, mot_len-window_size)
            j = i + window_size
            body = body[i:j]
            mot_len = body.shape[0]
            
        # window_size = np.max(np.array(self.all_lengths)) + 8    # We would at least apped 2 <EOS> per sequence.
        pad_len = window_size - mot_len
        pad_body = np.zeros((pad_len, body.shape[1]), dtype=body.dtype)
        body = np.concatenate([body, pad_body], axis=0)
        body_length = mot_len
        
        # Get word tokens from Vocab
        if len(t_tokens) < self.opt["t2m_max_text_len"]:
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
            t_tokens += ['unk/OTHER'] * (self.opt["t2m_max_text_len"] + 2 - sent_len)
        else:
            t_tokens = t_tokens[:self.opt["t2m_max_text_len"]]
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
        
        batch = {"body": body, "text": caption, "length": body_length, "text_list": [t["caption"] for t in text_list]}
        return batch
    
    def read_one_a2m_data(self, item):
        name = self.name_list[item]
        body = self.data[item]["body"] # The motion has already been downsampled to fps=30
        audio = self.data[item]["audio"] # The fps of audio is 60 by default, and we don't modify it.
        duration = self.data[item]["duration"]
        batch = {"body": body, "audio": audio, "name": name, "duration": duration}
        return batch
    
    def read_one_t2t_data(self, item):
        name = self.name_list[item]
        json_data = self.data[item]
        scene = random.choice(json_data["video_scene"])
        duration = json_data["duration"]
        next_duration = json_data["next_duration"]
        cur_action = random.choice(json_data["cur_actions"])
        next_action = random.choice(json_data["next_actions"])
        name = json_data["name"]
        
        batch = {
            "scene": "[scene] {:s}".format(scene), 
            "duration": "this action lasts {:.1f} seconds".format(duration), 
            "cur_action": "[current action]: {:s}".format(cur_action), 
            "next_action": next_action, 
            "next_duration": "this action lasts {:.1f} seconds".format(next_duration), 
            "name": name, 
            "scene_list": json_data["video_scene"], 
            "cur_action_list": json_data["cur_actions"], 
            "next_action_list": json_data["next_actions"]
        }
        return batch
    
class AvatarGPTEvalHMLDataset(data.Dataset):
    def __init__(
        self, opt, 
        t2m_split_file=None, 
        a2m_split_file=None, 
        t2t_split_file=None, 
        meta_dir=None
    ):
        super(AvatarGPTEvalHMLDataset, self).__init__()
        
        self.opt = opt
        self.meta_dir = meta_dir
        self.times = opt["times"]
        self.read_mirrors = self.opt.get("read_mirrors", True)
        self.setup_stats()
        
        id_lists = {}
        
        # Text-to-Motion
        if "t2m" in self.opt["modality"]:
            id_lists["t2m"] = []
            with cs.open(t2m_split_file, "r") as f:
                for line in f.readlines():
                    id_lists["t2m"].append(line.strip())
                
        # Audio-to-Motion
        if "a2m" in self.opt["modality"]:
            id_lists["a2m"] = []
            with cs.open(a2m_split_file, "r") as f:
                for line in f.readlines():
                    id_lists["a2m"].append(line.strip())
                    
        # Text-to-Text
        if "t2t" in self.opt["modality"]:
            id_lists["t2t"] = []
            with cs.open(t2t_split_file, "r") as f:
                for line in f.readlines():
                    id_lists["t2t"].append(line.strip())
                    
        # Load all dataset
        self.data = []
        self.name_list = []
        self.modality = []
        for key, id_list in id_lists.items():
            if "t2m" == key:
                data, name_list = self.read_all_t2m_data(id_list)
                # print('---->', len(data), "|", len(name_list))
            elif "a2m" == key:
                data, name_list = self.read_all_a2m_data(id_list)
                # print('----', len(data), "|", len(name_list))
            elif "t2t" == key:
                data, name_list = self.read_all_t2t_data(id_list)
                # print('----', len(data), "|", len(name_list))
                
            self.data += data
            self.name_list += name_list
            self.modality += [key] * len(data)
        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        valid_test_set = json.load(
            open(os.path.join(dir_path, "ValidTestset_HumanML3D.json"), "r"))
        self.valid_amass_test_set = [t["name"] for t in valid_test_set]
        self.num_data = len(self.data)
    
    def setup_stats(self):
        
        if self.opt.get("meta_mode", "tm2t") == "tm2t":
            std = np.load(self.opt.get("std_dir", "data/Std.npy"))
            mean = np.load(self.opt.get("mean_dir", "data/Mean.npy"))
            feat_bias = self.opt.get("feat_bias", 5)
            joints_num = self.opt.get("joints_num", 22)

            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4:4+(joints_num-1)*3] = std[4:4+(joints_num-1)*3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4+(joints_num-1)*3:4+(joints_num-1)*9] = std[4+(joints_num-1)*3:4+(joints_num-1)*9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4+(joints_num-1)*9:4+(joints_num-1)*9+joints_num*3] = std[4+(joints_num-1)*9:4+(joints_num-1)*9+joints_num*3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4+(joints_num-1)*9+joints_num*3:] = std[4+(joints_num-1)*9+joints_num*3:] / feat_bias
            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
        elif self.opt.get("meta_mode", "tm2t") == "t2m-gpt":
            std = np.load(self.opt.get("std_dir", "pretrained_models/vqvae/std.npy"))
            mean = np.load(self.opt.get("mean_dir", "pretrained_models/vqvae/mean.npy"))
            
        if self.meta_dir is not None:
            np.save(pjoin(self.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(self.meta_dir, 'std.npy'), std)
        
        self.mean = mean
        self.std = std
    
    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, item):
        modality = self.modality[item]
        if modality == "t2m":
            batch = self.read_one_t2m_data(item)
        elif modality == "a2m":
            batch = self.read_one_a2m_data(item)
        elif modality == "t2t":
            batch = self.read_one_t2t_data(item)
        batch.update({"modality": modality})
        return batch
    
    def read_all_t2m_data(self, id_list):
        new_name_list = []
        data_dict = []
        self.all_lengths = []    # record the lengths of every HumanML3D data sample
        for name in tqdm(id_list, desc="Reading text-to-motion dataset"):
            try:
                body = np.load(pjoin(self.opt["t2m_motion_dir"], name+".npy"))
                with cs.open(pjoin(self.opt["text_dir"], name+".txt")) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()
                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            # part_of_speechs = self.parse_part_of_speech(t_tokens)
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag
                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            # text_dict['part_of_speech'] = part_of_speechs
                            
                            # print(text_dict)
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)
                                data_dict.append({'text':[text_dict], "body": body, "name": new_name})
                                new_name_list.append(new_name)
                                self.all_lengths.append(body.shape[0])
                        except:
                            pass
                
                if flag:
                    data_dict.append({'text': text_data, 'body': body, "name": name})
                    new_name_list.append(name)
                    self.all_lengths.append(body.shape[0])
            
            except:
                pass
        
        print('---', len(data_dict), "|", len(new_name_list))
        return data_dict, new_name_list
    
    def read_all_a2m_data(self, id_list):
        new_name_list = []
        data_dict = []
        for name in tqdm(id_list, desc="Reading audio-to-motion dataset"):
            try:
                data = np.load(pjoin(self.opt["a2m_motion_dir"], name+".npy"), allow_pickle=True).item()
                body = data["motion_smpl"][::2] # downsample to fps=30
                audio = data["audio_sequence"]
                duration = data["duration"]
                data_dict.append({"audio": audio, "body": body, "name": name, "duration": duration, "name": name})
                new_name_list.append(name)
            except:
                pass
        return data_dict, new_name_list
    
    def read_all_t2t_data(self, id_list):
        if not self.opt.get("decompose_actions", False):
            return self.__read_all_t2t_data(id_list=id_list)
        else:
            return self.__read_all_t2t_data_decomposed(id_list=id_list)
    
    def __read_all_t2t_data(self, id_list):
        new_name_list = []
        data_dict = []
        for name in tqdm(id_list, desc="Reading text-to-text dataset"):
            try:
                with open(pjoin(self.opt["t2t_text_dir"], name+".json"), "r") as f:
                    json_data = json.load(f)
                    
                    video_scene = json_data[0]["video_scene"]
                    video_fps = json_data[0]["video_fps"]
                    
                    for action_id in range(1, len(json_data)-1):
                        duration = (json_data[action_id]["end_frame"] - json_data[action_id]["start_frame"]) / video_fps
                        cur_actions = json_data[action_id]["captions"]
                        next_actions = json_data[action_id+1]["captions"]
                        next_duration = (json_data[action_id+1]["end_frame"] - json_data[action_id+1]["start_frame"]) / video_fps
                    
                        data_dict.append(
                            {
                                "video_scene": video_scene, 
                                "duration": duration, 
                                "cur_actions": cur_actions, 
                                "next_actions": next_actions, 
                                "next_duration": next_duration, 
                                "name": "{:s}_{:d}".format(name, action_id)
                            }
                        )
                        new_name_list.append("{:s}_{:d}".format(name, action_id))
            except:
                pass
        return data_dict, new_name_list
    
    def __read_all_t2t_data_decomposed(self, id_list):
        new_name_list = []
        data_dict = []
        for name in tqdm(id_list, desc="Reading text-to-text dataset"):
            try:
                with open(pjoin(self.opt["t2t_text_dir"], name+".json"), "r") as f:
                    json_data = json.load(f)
                
                video_scene = json_data["contexts"]
                video_fps = json_data["video_fps"]
                
                action_data = json_data["actions"]
                for action_id in range(len(action_data) - 1):
                    cur_duration = (action_data[action_id]["end_frame"] - action_data[action_id]["start_frame"]) / video_fps
                    cur_actions = action_data[action_id]["captions"]
                    next_duration = (action_data[action_id+1]["end_frame"] - action_data[action_id+1]["start_frame"]) / video_fps
                    next_actions = action_data[action_id+1]["captions"]
                    
                    data_dict.append(
                        {
                        "video_scene": video_scene, 
                            "duration": cur_duration, 
                            "next_duration": next_duration, 
                            "cur_actions": cur_actions, 
                            "next_actions": next_actions
                        }
                    )
                    new_name_list.append("{:s}_{:d}".format(name, action_id))
            except:
                pass
        return data_dict, new_name_list
    
    def read_one_t2m_data(self, item):
        body = self.data[item]["body"]
        text_list = self.data[item]["text"]
        text_data = random.choice(text_list)    # Random sample one text
        caption = text_data["caption"]
        t_tokens = text_data["tokens"]
        mot_len = body.shape[0]
        window_size = self.opt["window_size"]["t2m"]
        
        if mot_len > window_size:
            i = np.random.randint(0, mot_len-window_size)
            j = i + window_size
            body = body[i:j]
            mot_len = body.shape[0]
            
        # window_size = np.max(np.array(self.all_lengths)) + 8    # We would at least apped 2 <EOS> per sequence.
        pad_len = window_size - mot_len
        pad_body = np.zeros((pad_len, body.shape[1]), dtype=body.dtype)
        body = np.concatenate([body, pad_body], axis=0)
        body_length = mot_len
        
        # Get word tokens from Vocab
        if len(t_tokens) < self.opt["t2m_max_text_len"]:
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
            t_tokens += ['unk/OTHER'] * (self.opt["t2m_max_text_len"] + 2 - sent_len)
        else:
            t_tokens = t_tokens[:self.opt["t2m_max_text_len"]]
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
        
        body = (body - self.mean) / self.std
        batch = {"body": body, "text": caption, "length": body_length, "text_list": [t["caption"] for t in text_list]}
        return batch
    
    def read_one_a2m_data(self, item):
        name = self.name_list[item]
        body = self.data[item]["body"] # The motion has already been downsampled to fps=30
        audio = self.data[item]["audio"] # The fps of audio is 60 by default, and we don't modify it.
        duration = self.data[item]["duration"]
        batch = {"body": body, "audio": audio, "name": name, "duration": duration}
        return batch
    
    def read_one_t2t_data(self, item):
        if not self.opt.get("decompose_actions", False):
            return self.__read_one_t2t_data(item=item)
        else:
            return self.__read_one_t2t_data_decomposed(item=item)
    
    def __read_one_t2t_data(self, item):
        name = self.name_list[item]
        json_data = self.data[item]
        scene = random.choice(json_data["video_scene"])
        duration = json_data["duration"]
        next_duration = json_data["next_duration"]
        cur_action = random.choice(json_data["cur_actions"])
        next_action = random.choice(json_data["next_actions"])
        name = json_data["name"]
        
        batch = {
            "scene": "[scene] {:s}".format(scene), 
            "duration": "this action lasts {:.1f} seconds".format(duration), 
            "cur_action": "[current action]: {:s}".format(cur_action), 
            "next_action": next_action, 
            "next_duration": "this action lasts {:.1f} seconds".format(next_duration), 
            "name": name, 
            "scene_list": json_data["video_scene"], 
            "cur_action_list": json_data["cur_actions"], 
            "next_action_list": json_data["next_actions"]
        }
        return batch
    
    def __read_one_t2t_data_decomposed(self, item):
        name = self.name_list[item]
        json_data = self.data[item]
        scene = random.choice(json_data["video_scene"])
        cur_duration = json_data["duration"]
        next_duration = json_data["next_duration"]
        cur_action_info = random.choice(json_data["cur_actions"])
        cur_executable_steps = cur_action_info["executable_steps"]
        cur_executable_simplified = random.choice(cur_action_info["executable_simplified"])
        next_action_info = random.choice(json_data["next_actions"])
        next_executable_steps = cur_action_info["executable_steps"]
        next_executable_simplified = random.choice(next_action_info["executable_simplified"])
        
        cur_action_list, next_action_list = [], []
        for (t1, t2) in zip(json_data["cur_actions"], json_data["next_actions"]):
            cur_action_list += t1["executable_simplified"]
            next_action_list += t2["executable_simplified"]
            
        
        batch = {
            "scene": "[scene] {:s}".format(scene), 
            "duration": "this action lasts {:.1f} seconds".format(cur_duration), 
            "next_duration": "this action lasts {:.1f} seconds".format(next_duration), 
            "cur_action": "[current action]: {:s}".format(cur_executable_simplified), 
            "next_action": next_executable_simplified, 
            "cur_steps": cur_executable_steps, 
            "next_steps": next_executable_steps, 
            "name": name, 
            "scene_list": json_data["video_scene"], 
            "cur_action_list": cur_action_list, 
            "next_action_list": next_action_list, 
            "cur_steps_list": [t["executable_steps"] for t in json_data["cur_actions"]], 
            "next_steps_list": [t["executable_steps"] for t in json_data["next_actions"]]
        }
        return batch
    
DATASET_MAP = {
    "AvatarGPTDataset": AvatarGPTDataset, 
    "AvatarGPTHMLDataset": AvatarGPTHMLDataset, 
    "AvatarGPTEvalDataset": AvatarGPTEvalDataset, 
    "AvatarGPTEvalHMLDataset": AvatarGPTEvalHMLDataset, 
}
    
def get_dataloader(data_conf, loader_conf, meta_dir=None):
    
    split_files = {}
    for key, item in loader_conf["split_path"].items():
        if isinstance(loader_conf["split"][key], str):
            split_files["{:s}_split_file".format(key)] = pjoin(item, loader_conf["split"][key]+".txt")
        elif isinstance(loader_conf["split"][key], list):
            split_files["{:s}_split_file".format(key)] = [pjoin(item, split+".txt") for split in loader_conf["split"][key]]
    
    dataset = DATASET_MAP[loader_conf["dataset"]](data_conf, **split_files, meta_dir=meta_dir)
    loader = DataLoader(dataset, 
                        batch_size=loader_conf["batch_size"], 
                        drop_last=True, 
                        num_workers=loader_conf["workers"], 
                        shuffle=loader_conf["shuffle"], 
                        pin_memory=True)
    return loader, dataset
    
if __name__ == "__main__":
    import yaml
    with open("configs/llm_t5/flan_t5_large/config_flan_t5_large_exp7.yaml", "r") as f:
    # with open("configs/denoiser/config_denoiser_exp13_half.yaml", "r") as f:
        conf = yaml.safe_load(f)
    
    loader, _ = get_dataloader(
        data_conf=conf["data"]["dataset"], 
        loader_conf=conf["data"]["loader"]["train"], 
        meta_dir=None)
    for bid, batch in enumerate(tqdm(loader)):
        # pass
        modality = batch["modality"]
        if modality[0] == "t2t":
            print(batch["scene"][0], "|", batch["cur_action"][0], "|", batch["next_action"][0])
        # t2t_batch = batch["t2t"]
        # scene = t2t_batch["scene"]                  # List of strings
        # duration = t2t_batch["duration"]            # List of strings
        # next_duration = t2t_batch["next_duration"]  # List of strings
        # cur_action = t2t_batch["cur_action"]        # List of strings
        # next_action = t2t_batch["next_action"]      # List of strings
        # input_text = [s + " " + c + " " + d + " [next action] " for (s, c, d) in zip(scene, cur_action, duration)]
        # targ_text = [n + " " + d for (n, d) in zip(next_action, next_duration)]
        # for (i, t) in zip(input_text, targ_text):
        #     print(i, "|", t)
