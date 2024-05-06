import os, sys
sys.path.append(os.getcwd())
import argparse
import openai 
import json
import codecs as cs
from tqdm import tqdm
import numpy as np
import random

openai.api_type = "azure"
openai.api_base = "https://llm.openai.azure.com/"
openai.api_version = "yyyy-mm-dd"                   # Enter your API version here
openai.api_key = 'xxxxx'                            # Enter your API key here

def run(text, task="content", example="action", check=False, num_repeats=3):
    
    action_example = {
        "input": "The video shows a man with his back facing the camera. He is wearing a blue sweater and holding a microphone in his hand. He has a white button on his chest with white text and is speaking in chinese. There is a red background behind him. He is not smiling and has a serious expression. The video does not show any movement or activity other than the man's speaking and holding the microphone.", 
        "output": "A man is holding a microphone on a stage and giving a speech."
    }
    
    context_example = {
        "input": "In the video, a young blonde woman is seen lying on the ground, performing a variety of fitness exercises, such as push-ups and squats, on a mat next to a pool. She is wearing a white sports bra and shorts and a pair of white sneakers. She also performs some jump squats and lunges. The video includes a caption that reads \"girl in white.\" In some frames, the woman wears headphones and listens to music. There is a sign that says \"diet meal\" and some furniture in the background, including a white couch. The water in the pool looks clear and blue, and there is a hose in the water. The woman's face is not visible, but she appears to be smiling in some frames. The background is mostly white and green, with a few brown plants and a pool towel hanging from a tree. The video ends with the woman standing up from her final exercise and reaching out for her water bottle.", 
        "output": "A woman is doing yoga exercises in a gym."
    }
    
    action_unit_example = {
        "input": "In the video, a young blonde woman is seen lying on the ground, performing a variety of fitness exercises, such as push-ups and squats, on a mat next to a pool. She is wearing a white sports bra and shorts and a pair of white sneakers. She also performs some jump squats and lunges. The video includes a caption that reads \"girl in white.\" In some frames, the woman wears headphones and listens to music. There is a sign that says \"diet meal\" and some furniture in the background, including a white couch. The water in the pool looks clear and blue, and there is a hose in the water. The woman's face is not visible, but she appears to be smiling in some frames. The background is mostly white and green, with a few brown plants and a pool towel hanging from a tree. The video ends with the woman standing up from her final exercise and reaching out for her water bottle.", 
        "output": "1. lying on the ground.\n2. performing push-ups.\n3. performing squats.\n4. performing jump squats.\n5. performing lunges."
    }
    
    action_concat_example = {
        "input": "1. lying on the ground.\n2. performing push-ups.\n3. performing squats.\n4. performing jump squats.\n5. performing lunges.", 
        "output": "A person is first lying on the ground, then he does various exercises such as push-ups, squats, jumps and lunges"
    }
    
    action_decompose_example = {
        "input": "", 
        "output": ""
    }
        
    EXAMPLES = {
        "action": action_example, 
        "context": context_example, 
        "action_unit": action_unit_example, 
        "concat": action_concat_example, 
        "decompose": action_decompose_example
    }
    
    # content_list = [
    #     "I'll give you a long text description, the description is given following '[Input]'. The text description describes the content of a video in detail. Please summarize the activity performed by the person in the video, and answer me in a consice but semantically accurate sentence.\n[Input] {:s}", 
    #     "I'll give you a long text description, the description is given following '[Input]'. The text description describes the content of a video in detail. Please descibe what is the person doing, and answer me in a short sentence. The short sentence should contain semantically rich and accurate information. \n{:s}", 
    #     "I'll give you a long text description, the description is given following '[Input]'. Please summarize the actions the person does. You should give me a short sentence as answer, the anwser should contain only the action information. \n[Input] {:s}"
    # ]
    
    few_shot_content_list = [
        "I'll give you a long text description, please summarize the actions the person does and answer me with a very short sentence which contains semantically rich accurate information. \nFor example, when given text description: ({:s}), the answer could be: ({:s}). \nNow please summarize the actions in this text description: ({:s})"
    ]
    
    few_shot_context_list = [
        "I'll give you a long text description, please summarize the context information described by this text description. The summariation should be a very short sentence which contains accurate context information. \nFor example, when given the text description: ({:s}), the answer could be: ({:s}). \nNow please summarize the context of this text description: ({:s})"
    ]
    
    few_shot_unit_list = [
        "I'll give you a long text description, please summarize the actions the person does and list all the action the person does in sequence. The answer should only contain very basic action description. \nFor example, when given text description: ({:s}), the answer could be: ({:s}). \nNow please summarize the actions in this text description: ({:s})"
    ]
    
    few_shot_concat_list = [
        "I'll give you a list of action decriptions, please write a short sentence which contains same semantic meaning.\nFor example, when given action description list: ({:s}), the answer could be: ({:s}).\nNow please write a sentence based on these action descriptions: ({:s})"
        # "I'll give you a list of action unit descriptions, please write a short sentence that summarizes all the action unit descriptions.\nFor example, when given action unit descriptions: ({:s}), the answer could be: ({:s}).\nNow please summarize the action unit descriptions: ({:s})"
    ]
    
    few_shot_decompose_list = [
        "I'll give you a long text description, it describes a person is conducting some activity, as well as the enviroment. Please list all movements that this person conducted. For each movement, you should answer me with no more than 5 words.\nThe long text descriptions is: <{:s}{:s}{:s}>"
    ]
        
    TASKS = {
        "content": few_shot_content_list, 
        "context": few_shot_context_list, 
        "unit": few_shot_unit_list, 
        "concat": few_shot_concat_list, 
        "decompose": few_shot_decompose_list
    }
    
    def parse_results(text):
        splits = text.split("\n")
        words = []
        for i, w in enumerate(splits):
            words.append(w.replace("{:d}. ".format(i+1), ""))
        return words
    
    def check_results(text):
        check_prompt = "I'll give you a short sentence, please determine whether it describes basic human action. Please answer me in 'Yes' or 'No'. The input sentence is: <{:s}>".format(text)
        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo", 
            messages = [
                {
                    "role": "user", 
                    "content": check_prompt}
            ],
            temperature=0.7,
            max_tokens=100,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["<|im_end|>"]
        )
        results = response['choices'][0]['message']['content']
        if "Yes" in results or "yes" in results:
            return True
        else:
            return False
        
    results = []
    for content in TASKS[task]:
        for _ in range(num_repeats):
            try:
                response = openai.ChatCompletion.create(
                    engine="gpt-35-turbo",
                    messages = [
                        {
                            "role": "user", 
                            "content": content.format(EXAMPLES[example]["input"], EXAMPLES[example]["output"], text)}],
                    temperature=0.7,
                    max_tokens=100,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=["<|im_end|>"])
                result = response['choices'][0]['message']['content']
                if check:
                    result = parse_results(text=result)
                    valid_result = []
                    for res in result:
                        flag = check_results(text=res)
                        if flag:
                            valid_result.append(res)
                    result = valid_result
                if len(result) == 0:
                    continue
                results.append(result)
            except:
                pass
            
    return results

def run_abstract(text):
    
    # prompt = "I'll give you a set of textual descriptions, they all describe a person is performing some actions.\nThese textual descriptions are:\n{:s}\nNow according to these descriptions, please summarize and give me a list of short action descriptions that desribes what this person does. The answer should be very clear and concrete action descriptions.".format(text)
    prompt = "I'll give you a set of textual descriptions, they all describe a person is performing some actions.\nThese textual descriptions are:\n{:s}\nNow according to these descriptions, please summarize and give me a very clear and concrete sentence that desribes what this person is doing. Please answer me with no more then 50 words".format(text)
    print(prompt)
    results = []
    for _ in range(3):
        try:
            response = openai.ChatCompletion.create(
                engine="gpt-35-turbo",
                messages = [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=100,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["<|im_end|>"])
            result = response['choices'][0]['message']['content']
            results.append(result)
        except:
            pass
    
    return results

def main_description(annotation_json):
    refined_annotation_json = {}
    refined_annotation_json["video_name"] = annotation_json["video_name"]
    refined_annotation_json["video_fps"] = annotation_json["video_fps"]
    
    refined_caption_list = []
    for annotation_item in tqdm(annotation_json["caption_list"], desc="Action descriptions"):
        inp_caption_list = annotation_item["captions"]
        refined_annotation_item = {}
        out_caption_list = []
        for text in inp_caption_list:
            results = run(text=text, task="content", example="action")
            out_caption_list += results
        refined_annotation_item.update(annotation_item)
        refined_annotation_item["captions"] = out_caption_list
        refined_caption_list.append(refined_annotation_item)
    
    refined_annotation_json["caption_list"] = refined_caption_list
    
    refined_description_list = []
    for text in tqdm(annotation_json["description"], desc="Context descriptions"):
        results = run(text=text, task="context", example="context")
        refined_description_list += results
    
    refined_annotation_json["description"] = refined_description_list
    
    return refined_annotation_json

def main_unit(annotation_json):
    refined_annotation_json = {}
    refined_annotation_json["video_name"] = annotation_json["video_name"]
    refined_annotation_json["video_fps"] = annotation_json["video_fps"]
    
    refined_caption_list = []
    for annotation_item in tqdm(annotation_json["caption_list"], desc="Action descriptions"):
        inp_caption_list = annotation_item["captions"]
        refined_annotation_item = {}
        out_caption_list = []
        for text in inp_caption_list:
            results = run(text=text, task="unit", example="action_unit", check=True, num_repeats=1)
            out_caption_list += results
        refined_annotation_item.update(annotation_item)
        refined_annotation_item["captions"] = out_caption_list
        refined_caption_list.append(refined_annotation_item)
    
    refined_annotation_json["caption_list"] = refined_caption_list
    
    refined_description_list = []
    for text in tqdm(annotation_json["description"], desc="Context descriptions"):
        results = run(text=text, task="context", example="context")
        refined_description_list += results
    
    refined_annotation_json["description"] = refined_description_list
    
    return refined_annotation_json

def main_abstract(annotation_json):
    refined_annotation_json = {}
    refined_annotation_json["video_name"] = annotation_json["video_name"]
    refined_annotation_json["video_fps"] = annotation_json["video_fps"]
    
    raw_caption_list = annotation_json.get("caption_list", None)
    if raw_caption_list is not None:
        for raw_captions in raw_caption_list:
            texts = raw_captions["captions"]
            # Re-write the texts
            texts_reorg = ["{:d}. {:s}".format(i, t) for (i, t) in enumerate(texts)]
            texts_reorg = "\n".join(texts_reorg)
            
            results = run_abstract(text=texts_reorg)

def main_concat(annotation_json):
    refined_annotation_json = {}
    refined_annotation_json["video_name"] = annotation_json["video_name"]
    refined_annotation_json["video_fps"] = annotation_json["video_fps"]
    
    def random_select(text_list):
        num_text = len(text_list)
        if num_text == 1:
            num_select = 1
        else:
            num_select = np.random.randint(1, num_text)
        texts_selected = random.sample(text_list, num_select)
        texts_selected = ["{:d}. {:s}".format(i+1, t) for (i, t) in enumerate(texts_selected)]
        return "\n".join(texts_selected)
    
    refined_caption_list = []
    for annotation_item in tqdm(annotation_json["caption_list"], desc="Action descriptions"):
        inp_caption_list = annotation_item["captions"]
        refined_annotation_item = {}
        out_caption_list = []
        for text in inp_caption_list:
            text_select = random_select(text_list=text)
            results = run(text=text_select, task="concat", example="concat", num_repeats=1)
            out_caption_list += results
        if len(out_caption_list) == 0:
            print('=' * 10, "Input: ", inp_caption_list, "=" * 10)
            continue
        refined_annotation_item.update(annotation_item)
        refined_annotation_item["captions"] = out_caption_list
        refined_caption_list.append(refined_annotation_item)
    
    refined_annotation_json["caption_list"] = refined_caption_list
    
    # We use the summarized context description
    refined_annotation_json["description"] = annotation_json["description"]
    
    return refined_annotation_json

def main_decompose(annotation_json):
    
    def run_chatgpt(prompt):
        try:
            response = openai.ChatCompletion.create(
                engine="gpt-35-turbo",
                messages = [
                    {
                        "role": "user", 
                        "content": input_prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1000,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["<|im_end|>"])
            result = response['choices'][0]['message']['content']
            return result
        except:
            return None
    
    refined_annotation_json = {}
    refined_annotation_json["video_name"] = annotation_json["video_name"]
    refined_annotation_json["video_fps"] = annotation_json["video_fps"]
    
    caption_lists = annotation_json["caption_list"]
    refined_caption_list = []
    for cap_list in tqdm(caption_lists, desc="Action descriptions: "):
        
        refined_annotation_item = {}
        out_caption_list = []
        for text in cap_list["captions"]:
            # 1. Generate executable textual descriptions
            prompts = [
                "I will provide you some very detail textual descriptions which describe a person is doing some activity.", 
                "\nYou need to help me to convert these descriptions to a script.", 
                "\nThis script should decompose the person's activity into executable steps, each step should be a very short sentence but clearly describe the body action/movement.", 
                "\nThe output MUST NOT contain any enviromental descriptions, background descriptions or appearance descriptions."
                "\n\nThe input textual descriptions are:\n", 
                "{:s}", 
                "\n\nThe output executable steps are:\n"
            ]
            input_prompt = "".join(prompts)
            input_prompt = input_prompt.format(text)
            execuable_text = run_chatgpt(prompt=input_prompt)
            if execuable_text is None:
                print("executable texts are NONEs")
                continue
                        
            # Generate simplified textual descriptions
            simplified_texts = []
            for _ in range(3):
                prompts = [
                    "I will provide you a set of executable descriptions, they descibe a person is doing some activity.", 
                    "\nPlease simplify these descriptions and write a short sentence.", 
                    "\nThis sentence should be no longer than 30 words.", 
                    "\n\nThe input descriptions are:\n", 
                    "{:s}", 
                    "\n\nThe output sentence is:\n"
                ]
                input_prompt = "".join(prompts)
                input_prompt = input_prompt.format(execuable_text)
                # print(input_prompt)

                try_cnt = 0
                while True and try_cnt < 10:
                    try_cnt += 1
                    simplifed_text = run_chatgpt(prompt=input_prompt)
                    if simplifed_text is None:
                        print("simplified text is NONE")
                        continue
                    if "1." not in simplifed_text:
                        break
                print(simplifed_text)
                simplified_texts.append(simplifed_text)

                output = {
                    "detail": text, 
                    "executable": execuable_text, 
                    "simplified": simplified_texts
                }
                out_caption_list.append(output)
        refined_annotation_item.update(cap_list)
        refined_annotation_item["captions"] = out_caption_list
        refined_caption_list.append(refined_annotation_item)
    
    refined_annotation_json["caption_list"] = refined_caption_list
    
    # We use the summarized context description
    refined_annotation_json["description"] = annotation_json["description"]
    
    return refined_annotation_json
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, 
                        default="summarize_concat", 
                        help="1. summarize_description, 2. summarize_unit, 3. summarize_concat, 4. summarize_abstract, 5. summarize_decompose")
    parser.add_argument('--input_dir', type=str, 
                        # default="example/motion_x/raw", 
                        default="../../../dataset/VideoCaptionData_chatgpt/summarized_unit", 
                        help='')
    parser.add_argument('--output_dir', type=str, 
                        default="../../../dataset/VideoCaptionData_chatgpt/summarized_concat", 
                        help='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    files = [f for f in os.listdir(args.input_dir) if ".json" in f]
    random.shuffle(files)
    for file in files:
        inp_file = os.path.join(args.input_dir, file)
        out_file = os.path.join(args.output_dir, file)
        if os.path.exists(out_file):
            print("{:s} already processed".format(out_file))
            continue
        
        with open(inp_file, "r") as f:
            annotation_json = json.load(f)

        # if args.task == "summarize_description":
        #     refined_annotation_json = main_description(annotation_json=annotation_json)
        # elif args.task == "summarize_unit":
        #     refined_annotation_json = main_unit(annotation_json=annotation_json)
        # elif args.task == "summarize_concat":
        #     refined_annotation_json = main_concat(annotation_json=annotation_json)
        # elif args.task == "summarize_abstract":
        #     refined_annotation_json = main_abstract(annotation_json=annotation_json)
        # else:
        #     refined_annotation_json = main_decompose(annotation_json=annotation_json)
        refined_annotation_json = main_decompose(annotation_json=annotation_json)
        # exit(0)
        with open(out_file, "w") as f:
            json.dump(refined_annotation_json, f)

