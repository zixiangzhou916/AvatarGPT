import os, sys
sys.path.append(os.getcwd())
import json
import openai
import random
import numpy as np
from tqdm import tqdm, trange
import pandas as pd
import torch
from packaging import version

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

def check_steps_consistency(input_data):
    gt_text = input_data["gt"]
    pred_text = input_data["pred"]
    content = "There are two sentences: [Sentence 1]:\n{:s}\n\n[Sentence 2]:\n{:s}\n\nPlease tell me whether these two sentences describe similary torso(body) movements. Please answer me with 'Yes' or 'No'.\n\n[Response]: ".format(gt_text, pred_text)
    result, flag = run_chatgpt(content=content)
    print(content)
    print("-" * 50)
    print(result)
    

    
if __name__ == "__main__":
    input_dir = "logs/avatar_gpt/eval/flan_t5_large/exp7/output/planning_se2_p0.json"
    with open(input_dir, "r") as f:
        input_data = json.load(f)
    
    for data in input_data.values():
        for step in data["steps"]:
            print('-' * 100)
            check_steps_consistency(input_data=step)
