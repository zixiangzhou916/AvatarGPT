import os, sys
sys.path.append(os.getcwd())
import openai 
import json
import codecs as cs
from tqdm import tqdm
openai.api_type = "azure"
openai.api_base = "https://llm.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = '5d93a0feea2a46ddac086d7d4d37d6ea'

# example 1
# response = openai.ChatCompletion.create(
#     engine="gpt-35-turbo",
#     messages = [
#         {
#             "role": "user", 
#             "content": "I'll give you a sentence, please give me another sentence with the same meaning.\n[input] Generate a sequence of motion tokens matching the following natural language description.\n[output]"}],
#     temperature=0.7,
#     max_tokens=800,
#     top_p=0.95,
#     frequency_penalty=0,
#     presence_penalty=0,
#     stop=["<|im_end|>"])
# result = response['choices'][0]['message']['content']
# print(result)

def read_text_file(path):
    with cs.open(path) as f:
        text_data = []
        flag = False
        lines = f.readlines()
        for line in lines:
            try:
                line_split = line.strip().split('#')
                caption = line_split[0]
                text_data.append(caption)
            except:
                pass
    return text_data

def augment_text_annotations(text_annotations, repeat_times=1):
    augmented_texts = []
    for text in text_annotations:
        content = "I'll give you a sentence, please give me another {:d} sentences with the same meaning.\n[Input] {:s}\n[Output]".format(
            repeat_times, text)
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
        res_texts = result.split("\n")
        for i, res in enumerate(res_texts):
            res = res.replace("{:d}. ".format(i+1), "")
            augmented_texts.append(res)
    return augmented_texts

def main(input_dir, output_path):
    
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            augmented_text_prompts = json.load(f)
    else:
        augmented_text_prompts = {}
    
    files = [f for f in os.listdir(input_dir) if ".txt" in f]
    for file in tqdm(files):
        if file in augmented_text_prompts.keys():
            continue
        text_data = read_text_file(os.path.join(input_dir, file))
        if len(text_data) == 0:
            continue
        try:
            # Connection to OpenAI may fail
            aug_text = augment_text_annotations(text_annotations=text_data, repeat_times=10)
            augmented_text_prompts[file] = aug_text
        except:
            pass
        
    with open(output_path, "w") as f:
        json.dump(augmented_text_prompts, f)
        
def main_v2(input_dir, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    files = [f for f in os.listdir(input_dir) if ".txt" in f]
    for file in tqdm(files):
        text_data = read_text_file(os.path.join(input_dir, file))
        if len(text_data) == 0:
            continue
        try:
            # Connection to OpenAI may fail
            aug_text = augment_text_annotations(text_annotations=text_data, repeat_times=10)
            with open(os.path.join(output_path, file.replace(".txt", ".json")), "w") as f:
                json.dump(aug_text, f)
        except:
            pass

if __name__ == "__main__":
    input_dir = "../dataset/HumanML3D/texts/"
    # output_path = "data/humanml3d_augmented_texts.json"
    # main(input_dir=input_dir, output_path=output_path)
    output_path = "data/augmented_prompts"
    main_v2(input_dir=input_dir, output_path=output_path)