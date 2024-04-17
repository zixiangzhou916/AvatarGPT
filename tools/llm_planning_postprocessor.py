import os, sys
sys.path.append(os.getcwd())
import argparse
import numpy as np
import json
from tqdm import tqdm

def postprocess_ct2t(data):
    scene = data["scene"][0]
    cur_task = data["cur_task"][0]
    # cur_steps = data["cur_steps"]
    # next_task = data["next_task"]
    # next_steps = data["next_steps"]
    pred_task = data["pred"][0]
    
    return {"scene": scene, "historical_task": cur_task, "target_task": pred_task}

def postprocess_cs2s(data):
    scene = data["scene"][0]
    # cur_task = data["cur_task"][0]
    cur_steps = data["cur_steps"]
    # next_task = data["next_task"]
    # next_steps = data["next_steps"]
    pred_steps = data["pred"][0]
    
    return {"scene": scene, "historical_steps": cur_steps, "target_steps": pred_steps}

def postprocess_ct2s(data):
    scene = data["scene"][0]
    cur_task = data["cur_task"][0]
    # cur_steps = data["cur_steps"]
    # next_task = data["next_task"]
    # next_steps = data["next_steps"]
    pred_steps = data["pred"][0]
    
    return {"scene": scene, "historical_task": cur_task, "target_steps": pred_steps}

def postprocess_cs2t(data):
    scene = data["scene"][0]
    # cur_task = data["cur_task"][0]
    cur_steps = data["cur_steps"]
    # next_task = data["next_task"]
    # next_steps = data["next_steps"]
    pred_task = data["pred"][0]
    
    return {"scene": scene, "historical_steps": cur_steps, "target_task": pred_task}

def postprocess_t2c(data):
    scene = data["scene"][0]
    cur_task = data["cur_task"][0]
    # cur_steps = data["cur_steps"]
    # next_task = data["next_task"]
    # next_steps = data["next_steps"]
    pred_scene = data["pred"][0]
    
    return {"scene": scene, "historical_task": cur_task, "target_scene": pred_scene}

def postprocess_s2c(data):
    scene = data["scene"][0]
    # cur_task = data["cur_task"][0]
    cur_steps = data["cur_steps"]
    # next_task = data["next_task"]
    # next_steps = data["next_steps"]
    pred_scene = data["pred"][0]
    
    return {"scene": scene, "historical_steps": cur_steps, "target_scene": pred_scene}

def postprocess_t2s(data):
    scene = data["scene"][0]
    cur_task = data["cur_task"][0]
    # cur_steps = data["cur_steps"]
    # next_task = data["next_task"]
    # next_steps = data["next_steps"]
    pred_steps = data["pred"][0]
    
    return {"scene": scene, "historical_task": cur_task, "target_steps": pred_steps}

def postprocess_s2t(data):
    scene = data["scene"][0]
    # cur_task = data["cur_task"][0]
    cur_steps = data["cur_steps"]
    # next_task = data["next_task"]
    # next_steps = data["next_steps"]
    pred_task = data["pred"][0]
    
    return {"scene": scene, "historical_steps": cur_steps, "target_task": pred_task}

TASK_MAP = {
    "ct2t": postprocess_ct2t, 
    "cs2s": postprocess_cs2s, 
    "ct2s": postprocess_ct2s, 
    "cs2t": postprocess_cs2t, 
    "t2c": postprocess_t2c, 
    "s2c": postprocess_s2c, 
    "t2s": postprocess_t2s, 
    "s2t": postprocess_s2t
}
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_base_dir', type=str, 
                        default='logs/avatar_gpt/eval/flan_t5_large/exp9/output', 
                        # default='logs/avatar_gpt/eval/gpt_large/exp9/output', 
                        help='')
    parser.add_argument('--output_base_dir', type=str, 
                        default='logs/avatar_gpt/eval/flan_t5_large/exp9/postprocessed', 
                        # default='logs/avatar_gpt/eval/gpt_large/exp9/postprocessed', 
                        help='')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    input_base_dir = args.input_base_dir
    output_base_dir = args.output_base_dir
    tasks = ["ct2t", "cs2s", "ct2s", "cs2t", "t2c", "s2c", "t2s", "s2t"]
    for task in tasks:
        input_task_dir = os.path.join(input_base_dir, task)
        if not os.path.exists(input_task_dir):
            continue
        
        output_task_dir = os.path.join(output_base_dir, task)
        if not os.path.exists(output_task_dir):
            os.makedirs(output_task_dir)
            
        files = [f for f in os.listdir(input_task_dir) if ".npy" in f]
        output = []
        for file in tqdm(files, desc="Task {:s}".format(task)):
            try:
                data = np.load(os.path.join(input_task_dir, file), allow_pickle=True).item()
                out = TASK_MAP[task](data=data)
                output.append(
                    {
                        "filename": file.replace(".npy", ""), 
                        "data": out
                    }
                )
            except:
                pass
        with open(os.path.join(output_task_dir, "result.json"), "w") as f:
            json.dump(output, f)