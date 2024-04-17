import os
import json
from tqdm import tqdm

if __name__ == "__main__":
    
    input_path = "data/augmented_prompts"
    input_files = [f for f in os.listdir(input_path) if ".json" in f]
    output_file = "data/humanml3d_augmented_texts.json"
    with open(output_file, "r") as f:
        augmented_prompts = json.load(f)
    
    for file in tqdm(input_files):
        with open(os.path.join(input_path, file), "r") as f:
            input_prompts = json.load(f)
        base_name = file.split(".")[0]
        key_name = base_name + ".txt"
        if key_name in augmented_prompts.keys():
            augmented_prompts[key_name] += input_prompts
        else:
            augmented_prompts[key_name] = input_prompts
    
    with open(output_file, "w") as f:
        json.dump(augmented_prompts, f)