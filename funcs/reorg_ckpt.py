import numpy as np
import torch
import torch.nn as nn

def print_ckpt(ckpt_path):

    ckpt_param = torch.load(ckpt_path, map_location=torch.device("cpu"))["param"]
    print('-' * 50)
    for name, val in ckpt_param.items():
        val_shape = ', '.join(map(str, val.size()))
        print(name, val_shape)

def combine_ckpt_for_ganimator_clip(src_ckpt_path, tgt_ckpt_path):

    src_ckpt_param = torch.load(src_ckpt_path, map_location=torch.device("cpu"))["param"]

    tgt_ckpt_param = dict()
    for name, val in src_ckpt_param.items():
        if "generator.skel_embedding" in name:
            new_name = name.replace("generator", "generator.motionencoder")
            tgt_ckpt_param[new_name] = val
        elif "generator.sequence_pos_encoding" in name:
            new_name = name.replace("generator", "generator.motionencoder")
            tgt_ckpt_param[new_name] = val
        elif "generator.latents" in name:
            new_name = name.replace("generator", "generator.motionencoder")
            tgt_ckpt_param[new_name] = val
        elif "generator.tokens" in name:
            new_name = name.replace("generator", "generator.motionencoder")
            tgt_ckpt_param[new_name] = val
        elif "generator.encoder" in name:
            new_name = name.replace("generator", "generator.motionencoder")
            tgt_ckpt_param[new_name] = val
        elif "generator.decoder" in name:
            new_name = name.replace("generator", "generator.motiondecoder")
            tgt_ckpt_param[new_name] = val
        elif "generator.linear" in name:
            new_name = name.replace("generator", "generator.motiondecoder")
            tgt_ckpt_param[new_name] = val
        elif "discriminator" in name:
            tgt_ckpt_param[name] = val

    # Print tgt_ckpt_param
    for name, val in tgt_ckpt_param.items():
        val_shape = ', '.join(map(str, val.size()))
        print(name, val_shape)

if __name__ == '__main__':
    import os
    os.environ["NODE_RANK"] = "0"
    cwd = os.getcwd()
    import sys
    sys.path.append(cwd)

    ckpt_path = "./logs/ganimator_alpha_multi_res/22-05-31-14-43-45/checkpoints/0/ganimator_10.pth"
    # print_ckpt(ckpt_path)
    combine_ckpt_for_ganimator_clip(ckpt_path, None)

    