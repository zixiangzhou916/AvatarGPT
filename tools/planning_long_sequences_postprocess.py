import os, sys, argparse, json
sys.path.append(os.getcwd())
import numpy as np
from tqdm import tqdm

def print_se1_p0(data):
    scene = data["caption"][0]
    actions = data["actions"]
    result = {}
    for action in actions:
        if scene in result.keys():
            if action["task"] in result[scene].keys():
                result[scene][action["task"]].append(action["step"])
            else:
                result[scene][action["task"]] = [action["step"]]
        else:
            result[scene] = {action["task"]: [action["step"]]}
        # result.append({"Scene": scene, "Task": action["task"], "Step": action["step"]})
    return result

def print_se1_p1(data):
    scene = data["caption"][0]
    actions = data["actions"]
    result = []
    for action in actions:
        result.append({"Scene": scene, "Task": action["task"]})
        # result.append("[Scene] {:s} | [Task] {:s}".format(scene, action["task"]))
    return result

def print_se1_p2(data):
    scene = data["caption"][0]
    actions = data["actions"]
    result = []
    for action in actions:
        result.append({"Scene": scene, "Steps": action["steps"]})
        # result.append("[Scene] {:s} | [Steps] {:s}".format(scene, action["steps"]))
    return result

def print_se1_p3(data):
    scene = data["caption"][0]
    actions = data["actions"]
    result = []
    for action in actions:
        result.append({"Scene": scene, "Steps": action["steps"], "Task": action["task"]})
        # result.append("[Scene] {:s} | [Steps] {:s} | [Task] {:s}".format(scene, action["steps"], action["task"]))
    return result

def print_se1_p4(data):
    scene = data["caption"][0]
    actions = data["actions"]
    result = []
    for action in actions:
        result.append({"Scene": scene, "Task": action["task"], "Steps": action["steps"]})
        # result.append("[Scene] {:s} | [Task] {:s} | [Steps] {:s}".format(scene, action["task"], action["steps"]))
    return result

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    # import torch
    
    # data = torch.randn(10)
    # d1 = torch.nn.functional.softmax(data, dim=-1)
    # r = torch.multinomial(d1, num_samples=20, replacement=True)
    # d2 = torch.nn.functional.softmax(data / 0.1, dim=-1)
    # d3 = torch.nn.functional.softmax(data / 10, dim=-1)
    
    # print(d1.data.cpu().numpy())
    # print(r)
    # print(d2.data.cpu().numpy())
    # print(d3.data.cpu().numpy())
    
    
    args = parse_args()
    
    input_dir = "logs/avatar_gpt/eval/gpt_large/exp3/output/se1_p0"
    output_dir = "logs/avatar_gpt/eval/gpt_large/exp3/output/planning_se1_p0.json"
    
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    
    files = [f for f in os.listdir(input_dir) if ".npy" in f]
    results = {}
    for file in tqdm(files):
        data = np.load(os.path.join(input_dir, file), allow_pickle=True).item()
        if "se1_p0" in input_dir or "se2_p0" in input_dir:
            result = print_se1_p0(data=data)
        elif "se1_p1" in input_dir or "se2_p1" in input_dir:
            result = print_se1_p1(data=data)
        elif "se1_p2" in input_dir or "se2_p2" in input_dir:
            result = print_se1_p2(data=data)
        elif "se1_p3" in input_dir or "se2_p3" in input_dir:
            result = print_se1_p3(data=data)
        elif "se1_p4" in input_dir or "se2_p4" in input_dir:
            result = print_se1_p4(data=data)
        results[file] = result
    
    with open(output_dir, "w") as f:
        json.dump(results, f)