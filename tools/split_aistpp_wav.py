import os
import random

if __name__ == "__main__":
    files = [f for f in os.listdir("../dataset/AIST++/wav") if ".wav" in f]
    num_files = len(files)
    
    files_reorganized = {}
    for file in files:
        genre = file[1:3]
        if genre not in files_reorganized.keys():
            files_reorganized[genre] = [file]
        else:
            files_reorganized[genre].append(file)
    
    training_files = []
    validation_files = []
    for key, vals in files_reorganized.items():
        random.shuffle(vals)
        training_files += vals[:-1]
        validation_files += vals[-1:]
    
    with open("../dataset/AIST++/train_wav.txt", "w") as f:
        for file in training_files:
            f.write(file+"\n")
    with open("../dataset/AIST++/vald_wav.txt", "w") as f:
        for file in validation_files:
            f.write(file+"\n")