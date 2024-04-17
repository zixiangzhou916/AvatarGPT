import os, sys, argparse, json
sys.path.append(os.getcwd())
import numpy as np
from tqdm import tqdm

def main(input_data):
    scene = input_data["scene"]
    gt_tasks = input_data["tasks"]
    pred_tasks = input_data["pred_task"]
    gt_steps = input_data["steps"]
    pred_steps = input_data["pred_steps"]
    step_cnt = input_data["step_cnt"]
    
    task_pairs = []
    for (gt_task, pred_task) in zip(gt_tasks, pred_tasks):
        task_pairs.append(
            {
                "gt": gt_task, 
                "pred": pred_task
            }
        )
    
    step_pairs = []
    for index in step_cnt:
        gt_steps_ = [gt_steps[i] for i in index]
        pred_steps_ = [pred_steps[i] for i in index]
        for (gt_step, pred_step) in zip(gt_steps_, pred_steps_):
            step_pairs.append(
                {
                    "gt": gt_step, 
                    "pred": pred_step
                }
            )
    
    output_data = {
        "scene": scene, 
        "tasks": task_pairs, 
        "steps": step_pairs
    }
    return output_data

if __name__ == "__main__":
    input_dir = "logs/avatar_gpt/eval/gpt_large/exp3/output/se2_p0"
    output_dir = "logs/avatar_gpt/eval/gpt_large/exp3/output/planning_se2_p0.json"
    
    files = [f for f in os.listdir(input_dir) if ".json" in f]
    results = {}
    for file in tqdm(files):
        with open(os.path.join(input_dir, file), "r") as f:
            input_data = json.load(f)
        output_data = main(input_data=input_data)
        results[file.replace(".json", ".npy")] = output_data
    
    with open(output_dir, "w") as f:
        json.dump(results, f)