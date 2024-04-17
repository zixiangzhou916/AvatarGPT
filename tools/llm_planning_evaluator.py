import os, sys
sys.path.append(os.getcwd())
import json
import openai
import random
import numpy as np
from tqdm import tqdm, trange
openai.api_type = "azure"
openai.api_base = "https://llm.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = '5d93a0feea2a46ddac086d7d4d37d6ea'

def run_chatgpt(content):
    cnt = 0
    result = None
    while True and cnt < 100:
        cnt += 1
        try:
            response = openai.ChatCompletion.create(
                engine="gpt-35-turbo",
                messages = [
                    {
                        "role": "user", 
                        "content": content}],
                temperature=0.7,
                max_tokens=800,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["<|im_end|>"])
            result = response['choices'][0]['message']['content']
            flag = True
            break
        except:
            flag = False
    return result, flag

def read_annotated_data(file):
    with open(file, "r") as f:
        annotated_data = json.load(f)
    contexts = annotated_data["contexts"]
    actions = []
    for action in annotated_data["actions"]:
        captions = action["captions"]
        pair_data = []
        for caption in captions:
            step = caption["executable_steps"]
            task = caption["executable_simplified"][0]
            pair_data.append({"step": step, "task": task})
        actions.append(pair_data)
    
    return contexts, actions
    
def get_task_from_scene_and_task(scene, task):
    content = "[Instruction] Predict the next action task given scene information and historical action tasks.\n[Scene] {:s}\n[Historical task(s)] {:s}\n[Response] ".format(scene, task)
    result, flag = run_chatgpt(content=content)
    print(content)
    print("-" * 50)
    print(result)

""" CT2T """
def check_task_from_scene_and_task(scene, historical_task, target_task, **kwargs):
    # content = "I will give you some scene information and historical task(s) description, tell me whether the target task is logically possible and consistent with the scene and historical task. Answer me with Yes or No.\n\n[Scene]:\n{:s}\n\n[Historical task(s)]:\n{:s}\n\n[Target task]:\n{:s}".format(scene, historical_task, target_task)
    content = "Suppose it is under specific scenario:\n\n[Scene]:\n{:s}\n\nThere are two descriptions that describe a person is doing something.\n\n[Task 1]:\n{:s}\n\n[Task 2]:\n{:s}\n\nPlease tell me whether it is logically possible that this person can execute these two tasks sequentially under the specific scenario with word 'Yes' or 'No'.\n\n[Result]: ".format(scene, historical_task, target_task)
    # content = "Suppose it is under specific scenario:\n\n[Scene]:\n{:s}\n\nThere are two descriptions that describe a person is doing something.\n\n[Task 1]:\n{:s}\n\n[Task 2]:\n{:s}\n\nPlease tell me whether it is 100 percent accurate that this person execute these two tasks sequentially under the specific scenario with word 'Yes' or 'No'.\n\n[Result]: ".format(scene, historical_task, target_task)
    result, flag = run_chatgpt(content=content)
    if flag:
        if "Yes" in result:
            return 1
        elif "No" in result:
            return 0
        else:
            return -1
    else:
        return -1

""" CS2S """
def check_steps_from_scene_and_steps(scene, historical_steps, target_steps, **kwargs):
    content = "Suppose it is under specific scenario:\n\n[Scene]:\n{:s}\n\nThere are two descriptions that describe a person is doing something by executing the descriptions step by step.\n\n[Steps 1]:\n{:s}\n\n[Steps 2]:\n{:s}\n\nPlease tell me whether it is logically plausible that this person can execute these step descriptions sequentially under the provided scenario with word 'Yes' or 'No'.\n\n[Result]: ".format(scene, historical_steps, target_steps)
    # content = "Suppose it is under specific scenario:\n\n[Scene]:\n{:s}\n\nThere are two descriptions that describe a person is doing something by executing the descriptions step by step.\n\n[Steps 1]:\n{:s}\n\n[Steps 2]:\n{:s}\n\nPlease tell me whether it is 100 percent correct that this person can execute these step descriptions sequentially under the provided scenario with word 'Yes' or 'No'.\n\n[Result]: ".format(scene, historical_steps, target_steps)
    # content = "I will give you some scene information and historical executable step(s), tell me whether the target executable step(s) is logically possible and consistent with the scene and historical executable steps. Answer me with Yes or No.\n\n[Scene]:\n{:s}\n\n[Historical step(s)]:\n{:s}\n\n[Target step]:\n{:s}".format(scene, historical_steps, target_steps)
    result, flag = run_chatgpt(content=content)
    if flag:
        if "Yes" in result:
            return 1
        elif "No" in result:
            return 0
        else:
            return -1
    else:
        return -1
    
""" CT2S """
def check_steps_from_scene_and_task(scene, historical_task, target_steps, **kwargs):
    content = "Suppose it is under specific scenario:\n\n[Scene]:\n{:s}\n\nThere are two descriptions, the first one describes somebody is conducting some task, and the second one is a list of axecutable action steps.\n\n[Task]:\n{:s}\n\n[Step]:\n{:s}\n\nPlease tell me whether it is logically plausible that this person can execute these action steps sequentially after he(she) finished the first task under the provided scenario with word 'Yes' or 'No'.\nn[Result]: ".format(scene, historical_task, target_steps)
    # content = "Suppose it is under specific scenario:\n\n[Scene]:\n{:s}\n\nThere are two descriptions, the first one describes somebody is conducting some task, and the second one is a list of axecutable action steps.\n\n[Task]:\n{:s}\n\n[Step]:\n{:s}\n\nPlease tell me whether it is correct that this person execute these action steps sequentially after he(she) finished the first task under the provided scenario with word 'Yes' or 'No'.\nn[Result]: ".format(scene, historical_task, target_steps)
    # content = "I will give you some scene information and historical executable step(s), tell me whether the target executable step(s) is logically possible and consistent with the scene and historical executable steps. Answer me with Yes or No.\n\n[Scene]:\n{:s}\n\n[Historical step(s)]:\n{:s}\n\n[Target step]:\n{:s}".format(scene, historical_steps, target_steps)
    result, flag = run_chatgpt(content=content)
        
    if flag:
        if "Yes" in result:
            return 1
        elif "No" in result:
            return 0
        else:
            return -1
    else:
        return -1
    
""" CS2T """
def check_task_from_scene_and_steps(scene, historical_steps, target_task, **kwargs):
    content = "Suppose it is under specific scenario:\n\n[Scene]:\n{:s}\n\nThere are two descriptions, the first one are list of executable action steps, and the second one is a sentence that describes somebody is conducting some task.\n\n[Step]:\n{:s}\n\n[Task]:\n{:s}\n\nPlease tell me whether it is logically plausible that this person can conduct this task after he(she) executes those action steps under the provided scenario with word 'Yes' or 'No'.\n[Result]: ".format(scene, historical_steps, target_task)
    # content = "Suppose it is under specific scenario:\n\n[Scene]:\n{:s}\n\nThere are two descriptions, the first one are list of executable action steps, and the second one is a sentence that describes somebody is conducting some task.\n\n[Step]:\n{:s}\n\n[Task]:\n{:s}\n\nPlease tell me whether it is correct that this person must conduct this task after he(she) executes those action steps under the provided scenario with word 'Yes' or 'No'.\n[Result]: ".format(scene, historical_steps, target_task)
    # content = "I will give you some scene information and historical executable step(s), tell me whether the target executable step(s) is logically possible and consistent with the scene and historical executable steps. Answer me with Yes or No.\n\n[Scene]:\n{:s}\n\n[Historical step(s)]:\n{:s}\n\n[Target step]:\n{:s}".format(scene, historical_steps, target_steps)
    result, flag = run_chatgpt(content=content)        
    if flag:
        if "Yes" in result:
            return 1
        elif "No" in result:
            return 0
        else:
            return -1
    else:
        return -1
    
""" T2C """
def check_scene_from_task(historical_task, scene, **kwargs):
    content = "I will give you a sentence that describes somebody is conducting some task.\n\n[Task]:\n{:s}\n\nI will also give you a sentence that describes a specific scene.\n\n[Scene]:\n{:s}\n\nPlease tell me whether this task is logically consistent with the given scenario with word 'Yes' or 'No'.\n\n[Result]: ".format(historical_task, scene)
    result, flag = run_chatgpt(content=content)   
        
    if flag:
        if "Yes" in result:
            return 1
        elif "No" in result:
            return 0
        else:
            return -1
    else:
        return -1
    
""" S2C """
def check_scene_from_steps(historical_steps, scene, **kwargs):
    content = "I will give you a list of action steps descriptions.\n\n[Steps]:\n{:s}\n\nI will also give you a sentence that describes a specific scene.\n\n[Scene]:\n{:s}\n\nPlease tell me whether these action steps are logically consistent with the given scenario with word 'Yes' or 'No'.\n\n[Result]: ".format(historical_steps, scene)
    result, flag = run_chatgpt(content=content)   
        
    # print(content)
    # print("="*50)
    # print(result)
    # exit(0)
        
    if flag:
        if "Yes" in result:
            return 1
        elif "No" in result:
            return 0
        else:
            return -1
    else:
        return -1
    
""" T2S """
def check_steps_from_task(historical_task, target_steps, **kwargs):
    content = "I will give you a task description that describes somebody is conducting some task.\n\n[Task]:\n{:s}\n\nI will also give you a list of action step descriptions.\n\n[Steps]:\n{:s}\n\nPlease tell me whether a person can conduct this specific task by executing these action steps with word 'Yes' or 'No'.\n\n[Result]: ".format(historical_task, target_steps)
    result, flag = run_chatgpt(content=content)   
                
    if flag:
        if "Yes" in result:
            return 1
        elif "No" in result:
            return 0
        else:
            return -1
    else:
        return -1

""" S2T """
def check_task_from_steps(historical_steps, target_task, **kwargs):
    content = "I will give you a list of action step descriptions.\n\n[Steps]:\n{:s}\n\nI will also give you a task description that describes somebody is conducting some task.\n\n[Task]:\n{:s}\n\nPlease tell me whether a person can conduct this specific task by executing these action steps with word 'Yes' or 'No'.\n\n[Result]: ".format(historical_steps, target_task)
    result, flag = run_chatgpt(content=content)   
        
    # print(content)
    # print("="*50)
    # print(result)
    # exit(0)
    if flag:
        if "Yes" in result:
            return 1
        elif "No" in result:
            return 0
        else:
            return -1
    else:
        return -1

    
FUNC_MAP = {
    "ct2t": check_task_from_scene_and_task, 
    "cs2s": check_steps_from_scene_and_steps, 
    "ct2s": check_steps_from_scene_and_task, 
    "cs2t": check_task_from_scene_and_steps, 
    "t2c": check_scene_from_task, 
    "s2c": check_scene_from_steps, 
    "t2s": check_steps_from_task, 
    "s2t": check_task_from_steps
}
    
def main(file, targ_tasks=["ct2t"]):
    total_score = {key: 0.0 for key in ["ct2t", "cs2s", "ct2s", "cs2t", "t2c", "s2c", "t2s", "s2t"]}
    total_cnt = {key: 0 for key in ["ct2t", "cs2s", "ct2s", "cs2t", "t2c", "s2c", "t2s", "s2t"]}
    contexts, actions = read_annotated_data(file=file)
    for i in trange(0, len(actions)-1, 1):
        for _ in range(3):
            scene = random.choice(contexts)
            inp_task = random.choice(actions[i])["task"]
            inp_steps = random.choice(actions[i])["step"]
            targ_task = random.choice(actions[i+1])["task"]
            targ_steps = random.choice(actions[i+1])["step"]
            
            inp_batch = {"scene": scene, "historical_task": inp_task, "target_task": targ_task, "historical_steps": inp_steps, "target_steps": targ_steps}
            
            for task in targ_tasks:
                score = FUNC_MAP[task](**inp_batch)
                if score != -1:
                    total_score[task] += score
                    total_cnt[task] += 1
    return total_score, total_cnt

if __name__ == "__main__":
    input_dir = "../dataset/VideoCaptionData_chatgpt/processed_llm_context"
    files = [f for f in os.listdir(input_dir) if ".json" in f]
    
    # file = "../dataset/VideoCaptionData_chatgpt/processed_llm_context/Evaluation0016.json"
    # score, cnt = main(file=file, targ_tasks=["s2t"])
    # for task in score.keys():
    #     if cnt[task] != 0:
    #         print("Task {:s} | score: {:.5f}".format(task, score[task]/cnt[task]))
    # exit(0)
    
    # total_score = 0
    # total_cnt = 0
    
    random.shuffle(files)
    files = files[::10]
    total_score = {key: 0.0 for key in ["ct2t", "cs2s", "ct2s", "cs2t", "t2c", "s2c", "t2s", "s2t"]}
    total_cnt = {key: 0 for key in ["ct2t", "cs2s", "ct2s", "cs2t", "t2c", "s2c", "t2s", "s2t"]}
    for index, file in enumerate(files):
        print("Evaluating [{:d}/{:d}]".format(index+1, len(files)))
        score, cnt = main(file=os.path.join(input_dir, file), targ_tasks=["s2t"])
        for task in score.keys():
            total_score[task] += score[task]
            total_cnt[task] += cnt[task]
        # total_score += score
        # total_cnt += cnt
    
    for task in total_score.keys():
        if total_cnt[task] != 0:
            print("Task {:s} | score: {:.5f}".format(task, total_score[task]/total_cnt[task]))

