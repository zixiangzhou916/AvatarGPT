import wave
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import librosa

import matplotlib
matplotlib.use("Agg")

def run(file, out_file):
    # wav_obj = torchaudio.load(file)[0][0].data.cpu().numpy()
    y, sr = librosa.load(file, duration=10)
    fig, ax = plt.subplots(nrows=3, sharex=True)
    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    ax[0].set(title='Envelope view, mono')
    ax[0].label_outer()
    
    y, sr = librosa.load(file, mono=False, duration=10)
    librosa.display.waveshow(y, sr=sr, ax=ax[1])
    ax[1].set(title='Envelope view, stereo')
    ax[1].label_outer()
    
    y, sr = librosa.load(file, duration=10)
    y_harm, y_perc = librosa.effects.hpss(y)
    librosa.display.waveshow(y_harm, sr=sr, alpha=0.5, ax=ax[2], label='Harmonic')
    librosa.display.waveshow(y_perc, sr=sr, color='r', alpha=0.5, ax=ax[2], label='Percussive')
    ax[2].set(title='Multiple waveforms')
    ax[2].legend()

    plt.savefig(out_file)
    plt.close()
    
if __name__ == "__main__":
    import os
    from tqdm import tqdm
    base_path = "logs/avatar_gpt/eval/lora_llama/exp3/output/a2m"
    output_path = "logs/avatar_gpt/eval/lora_llama/exp3/animation/a2m/wav"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    files = [f for f in os.listdir(base_path) if ".wav" in f]
    for file in tqdm(files):
        out_file = output_path+"/"+file.replace(".wav", ".png")
        run(base_path+"/"+file, out_file)
    
    