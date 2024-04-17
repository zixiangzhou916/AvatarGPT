import os
import glob
import random

if __name__ == "__main__":
    base_path = "../dataset/BEAT_v0.2.1/beat_english_v0.2.1"
    files = glob.glob(base_path+"/**/*.wav", recursive=True)
    num_files = len(files)
    num_training = int(num_files * 0.9)
    random.shuffle(files)
    
    training_files = []
    validation_files = []
    for idx, file in enumerate(files):
        base_len = len(base_path) + 1
        if idx <= num_training:
            training_files.append(file[base_len:])
        else:
            validation_files.append(file[base_len:])
    
    with open("../dataset/BEAT_v0.2.1/train_wav.txt", "w") as f:
        for file in training_files:
            f.write(file+"\n")
    with open("../dataset/BEAT_v0.2.1/vald_wav.txt", "w") as f:
        for file in validation_files:
            f.write(file+"\n")
    
    