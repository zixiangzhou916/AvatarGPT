import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import torchaudio

@torch.no_grad()
def update_predicted_tokens(inp_tokens, special_token):
    """
    :param inp_tokens: [T, 1]
    """
    mask = inp_tokens.gt(special_token).squeeze(dim=-1) # [T]
    out_tokens = inp_tokens[mask]   # [T, 1]
    return out_tokens

@torch.no_grad()
def prepare_generation_tokens(inp_tokens, sos, eos, pad, targ_length):
    """
    :param inp_tokens: [T, 1]
    """
    device = inp_tokens.device
    sos_token = torch.tensor(sos).long().to(device).view(1, 1)
    eos_token = torch.tensor(eos).long().to(device).view(1, 1)
    pad_token = torch.tensor(pad).long().to(device).view(1, 1)
    
    out_tokens = torch.cat([inp_tokens, eos_token], dim=0)
    mot_len = out_tokens.size(0)
    pad_len = targ_length - mot_len
    if pad_len > 0:
        pad_token = pad_token.repeat(pad_len, 1)
        out_tokens = torch.cat([sos_token, out_tokens, pad_token], dim=0)
    else:
        out_tokens = torch.cat([sos_token, out_tokens], dim=0)
    return out_tokens

@torch.no_grad()
def filter_special_tokens(inp_tokens, sos, eos, pad):
    """
    :param inp_tokens: [T, 1]
    """
    mask = inp_tokens.gt(pad).squeeze(dim=-1)   # [T]
    out_tokens = inp_tokens[mask]
    if out_tokens.dim() == 1:
        out_tokens = out_tokens.unsqueeze(dim=-1)   # [T, 1]
    return out_tokens

def apply_inverse_transform(inp_motion, data_obj):
    """
    :param inp_motion: [1, T, 263] torch tensor
    """
    if hasattr(data_obj, "mean") and hasattr(data_obj, "std"):
        out_motion = []
        for data in inp_motion:
            out = data_obj.inv_transform(data.data.cpu().numpy())
            out = torch.from_numpy(out).float()
            out_motion.append(out)
        return torch.stack(out_motion, dim=0)
    else:
        return inp_motion
    
def apply_transform(inp_motion, data_obj):
    if hasattr(data_obj, "mean") and hasattr(data_obj, "std"):
        out_motion = []
        for data in inp_motion:
            out = (data - data_obj.mean) / data_obj.std
            out = torch.from_numpy(out).float()
            out_motion.append(out)
        return torch.stack(out_motion, dim=0)
    else:
        raise ValueError

def save_generation_results(results, output_path, modality="t2m", batch_id=0):
    final_output_path = os.path.join(output_path, modality)
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)
    # Save results
    for tid, out in enumerate(results):
        np.save(os.path.join(final_output_path, "B{:004}_T{:004d}.npy".format(batch_id, tid)), out)
        
        try:
            gt_audio = torch.from_numpy(out["audio"])
            pred_audio = torch.from_numpy(out["pred_audio"])
            torchaudio.save(os.path.join(final_output_path, "B{:004}_T{:004}_gt.wav".format(batch_id, tid)), gt_audio, 16000)
            torchaudio.save(os.path.join(final_output_path, "B{:004}_T{:004}_pred.wav".format(batch_id, tid)), pred_audio, 16000)
        except:
            pass

def save_audio(output_path, wav, sr=16000):
    torchaudio.save(output_path, wav, sr)
    
def load_scene_information(input_path, split_file):
    import codecs as cs
    import json
    import random
    from tqdm import tqdm
    
    id_lists = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_lists.append(line.strip())
    
    scene_list = []
    for name in tqdm(id_lists, desc="Load scene descriptions"):
        with open(os.path.join(input_path, name+".json"), "r") as f:
            json_data = json.load(f)
        # Get all scene descriptions
        scenes = json_data[0]["video_scene"]
        actions = json_data[1]["captions"]
        # Random sample one scene
        scene = random.choice(scenes)
        action = random.choice(actions)
        scene_list.append({"scene": scene, "action": action})
    
    return scene_list

def load_scene_information_decompose(input_path, split_file):
    import codecs as cs
    import json
    import random
    from tqdm import tqdm
    
    id_lists = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_lists.append(line.strip())
    
    scene_list = []
    for name in tqdm(id_lists, desc="Load scene descriptions"):
        try:
            with open(os.path.join(input_path, name+".json"), "r") as f:
                json_data = json.load(f)
            scenes = json_data["contexts"]
            actions = json_data["actions"]
            for action in actions:
                scene_list.append({"scene": scenes, "action": action["captions"]})
        except:
            pass
    return scene_list

def load_text_descriptions(input_path, split_file):
    import codecs as cs
    import json
    import random
    from tqdm import tqdm
    
    id_lists = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_lists.append(line.strip())

    text_list = []
    for name in tqdm(id_lists, desc="Load text descriptions"):
        try:
            with open(os.path.join(input_path, name+".json"), "r") as f:
                tests = json.load(f)
            # Random sample one text descriptions
            text = random.choice(tests)
            text_list.append(text)
        except:
            pass
    
    return text_list

def load_task_planning_results(input_path):
    import json
    with open(input_path, "r") as f:
        input_data = json.load(f)
    
    output = {}
    for filename, file_item in input_data.items():
        tasks, steps, step_cnt = [], [], []
        scene = list(file_item.keys())[0]
        id = 0
        for task, step in file_item[scene].items():
            tasks.append(task)
            step_cnt.append([id+k for k in range(len(step))])
            id += len(step)
            steps += step
        output[filename] = {
            "scene": scene, 
            "tasks": tasks, 
            "steps": steps, 
            "step_cnt": step_cnt
        }
    return output

class PseudoDataset(object):
    def __init__(self) -> None:
        self.mean = np.load("networks/vqvae/pretrained/mean.npy")
        self.std = np.load("networks/vqvae/pretrained/std.npy")
    
    def inv_transform(self, data):
        return data * self.std + self.mean
    
def print_generation_info(inp_batch, predicted, task):
    assert task in ["ct2t", "cs2s", "ct2s", "cs2t", "t2c", "s2c", "t2s", "s2t"]
    batch = {}
    for key, val in inp_batch.items():
        if isinstance(val, list): batch[key] = val[0]
        else: batch[key] = val
        
    if task == "ct2t":
        task_name = "[Planning Scene-Task-to-Task]"
        input_text = "[Scene] {:s} | [Historical Task] {:s} | [Planned Task] {:s}".format(
            batch.get("scene", "No scene info"), batch.get("cur_task", "Idel"), predicted)
    elif task == "cs2s":
        task_name = "[Planning Scene-Steps-to-Steps]"
        input_text = "[Scene] {:s} | [Historical Steps] {:s} | [Planned Steps] {:s}".format(
            batch.get("scene", "No scene info"), batch.get("cur_steps", "Idel"), predicted)
    elif task == "ct2s":
        task_name = "[Planning Scene-Task-to-Steps]"
        input_text = "[Scene] {:s} | [Historical Task] {:s} | [Planned Steps] {:s}".format(
            batch.get("scene", "No scene info"), batch.get("cur_task", "Idel"), predicted)
    elif task == "cs2t":
        task_name = "[Planning Scene-Steps-to-Task]"
        input_text = "[Scene] {:s} | [Historical Steps] {:s} | [Planned Task] {:s}".format(
            batch.get("scene", "No scene info"), batch.get("cur_steps", "Idel"), predicted)
    elif task == "t2c":
        task_name = "[Planning Task-to-Scene]"
        input_text = "[Task] {:s} | [Estimated Scene] {:s}".format(batch.get("cur_task", "Idle"), predicted)
    elif task == "s2c":
        task_name = "[Planning Steps-to-Scene]"
        input_text = "[Steps] {:s} | [Estimated Scene] {:s}".format(batch.get("cur_steps", "Idle"), predicted)
    elif task == "t2s":
        task_name = "[Planning Task-to-Steps]"
        input_text = "[Task] {:s} | [Planned Steps] {:s}".format(batch.get("cur_task", "Idle"), predicted)
    elif task == "s2t":
        task_name = "[Planning Steps-to-Task]"
        input_text = "[Steps] {:s} | [Planned Task] {:s}".format(batch.get("cur_steps", "Idle"), predicted)
    
    return "{:s} | {:s}".format(task_name, input_text)
    
if __name__ == "__main__":
    load_task_planning_results("logs/avatar_gpt/eval/flan_t5_large/exp7/output/planning_se1_p0.json")