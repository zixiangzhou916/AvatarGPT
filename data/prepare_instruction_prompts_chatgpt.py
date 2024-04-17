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

def main(content):
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
    return result
    
if __name__ == "__main__":
    input = "Estimate the scene given set of executable steps."
    num_repeat = 30
    prompt = "I'll give you a sentence, please give me another {:d} sentences with the same meaning.\n[Input] {:s}\n[Output]".format(num_repeat, input)
    results =  main(content=prompt)
    print(results)