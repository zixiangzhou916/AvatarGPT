import os, sys, argparse
sys.path.append(os.getcwd())
import torch
import numpy as np
from scipy import interpolate

from funcs.hml3d.convert_d263_to_d75 import *
from funcs.hml3d.skeleton import Skeleton
from funcs.hml3d.quaternion import *
from funcs.hml3d.param_util import *
from funcs.hml3d.conversion import *

import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Process
import imageio
matplotlib.use("Agg")

SMPL_GRAPH = [
    (15, 12), (12, 9), (9, 13), (13, 16), 
    (16, 18), (18, 20), (9, 14), (14, 17), 
    (17, 19), (19, 21), (9, 6), (6, 3), 
    (3, 0), (0, 2), (0, 1), (2, 5), (5, 8), 
    (8, 11), (1, 4), (4, 7), (7, 10)
]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def plot_smpl_3d(joints, output_name, orient=None, ctr_xyz=None, hrange=None, color='k-', ego_centric=False):
    
    if ctr_xyz is None or hrange is None:
        min_xyz = np.min(joints, axis=0)
        max_xyz = np.max(joints, axis=0)
        ctr_xyz = 0.5 * (min_xyz + max_xyz)
        hrange = 0.5 * np.max(max_xyz - min_xyz)
        
    if ego_centric:
        min_xyz = np.min(joints, axis=0)
        max_xyz = np.max(joints, axis=0)
        ctr_xyz = 0.5 * (min_xyz + max_xyz)
        hrange = 2.0
    
    pxs = np.arange(ctr_xyz[0] - hrange, ctr_xyz[0] + hrange + 0.01, 2.0 * hrange)       # global plane
    pys = np.arange(ctr_xyz[1] - hrange, ctr_xyz[1] + hrange + 0.01, 2.0 * hrange)
    pxs, pys = np.meshgrid(pxs, pys)
    pzs = np.zeros_like(pxs)
    
    fig = plt.figure(figsize=(10, 10), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=15)
    
    ax.set_xlim(ctr_xyz[0] - hrange, ctr_xyz[0] + hrange)
    ax.set_ylim(ctr_xyz[1] - hrange, ctr_xyz[1] + hrange)
    ax.set_zlim(ctr_xyz[2] - hrange, ctr_xyz[2] + hrange)
    
    ax.plot_surface(pxs, pys, pzs, rstride=1, cstride=1, facecolors="green", alpha=0.5)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], marker='.', c='red', s=50, alpha=1)
    
    for (i, j) in SMPL_GRAPH:
        if i >= joints.shape[0] or j >= joints.shape[0]: continue
        ax.plot([joints[i, 0], joints[j, 0]], 
                [joints[i, 1], joints[j, 1]], 
                [joints[i, 2], joints[j, 2]], color, lw=5.0)
            
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(output_name)
    plt.close()
        
def plot_multiple_3d(joints, colors, indices, output_path, ctr_xyz, hrange, prefix, ego_centric, plot_type="smplx"):
    assert plot_type in ["smpl", "smplx", "flame"]
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for idx in indices:
        if idx >= joints.shape[0]: 
            continue
        joint = joints[idx] # [J, 3]
        output_name = os.path.join(output_path, "{:s}_{:d}.png".format(prefix, idx))
        if plot_type == "smpl":
            plot_smpl_3d(joints=joint, output_name=output_name, orient=None, ctr_xyz=ctr_xyz, hrange=hrange, color=colors[idx], ego_centric=ego_centric)
        else:
            raise ValueError("plot type {:s} not recognized!".format(plot_type))    
    
def animate_multiple(input_path, output_name, prefixs, fmt="gif", fps=30):
    assert fmt in ["gif", "mp4"]
    
    files = [int(f.split(".")[0].split("_")[-1]) for f in os.listdir(input_path) if ".png" in f]
    files = sorted(list(set(files)))
        
    if fmt == "gif":
        # Generate GIF from images
        with imageio.get_writer(output_name, mode="I") as writer:
            for file in tqdm(files, desc="Generating GIF..."):
                image = []
                for pref in prefixs:
                    # print(os.path.join(input_path, "{:s}_{:d}.png".format(pref, file)))
                    img = np.array(imageio.imread(os.path.join(input_path, "{:s}_{:d}.png".format(pref, file))), dtype=np.uint8)
                    image.append(img)
                    os.remove(os.path.join(input_path, "{:s}_{:d}.png".format(pref, file)))
                image = np.concatenate(image, axis=1)
                writer.append_data(image[..., :3])
                
    elif fmt == "mp4":
        # Generate MP4 from images
        with imageio.get_writer(output_name, fps=fps) as writer:
            for file in tqdm(files, desc="Generating MP4..."):
                image = []
                for pref in prefixs:
                    img = np.array(imageio.imread(os.path.join(input_path, "{:s}_{:d}.png".format(pref, file))), dtype=np.uint8)
                    image.append(img)
                    os.remove(os.path.join(input_path, "{:s}_{:d}.png".format(pref, file)))
                image = np.concatenate(image, axis=1)
                writer.append_data(image[..., :3])
    else:
        raise ValueError("format {:s} not recognized!".format(fmt))

def render_animation(joints_list, output_path, output_name, num_proc=1, plot_types="smpl", prefixs=[], video_type="gif", ego_centric=[], fps=30, colors=None):
    """Render videos of input joints_list.
    :param joints_list: list of joints [N, J, 3]
    :param output_path: directory of output path
    :param output_name: name of output video
    """
    if os.path.exists(os.path.join(output_path, output_name+"."+video_type)):
        print("{:s} already animated".format(os.path.join(output_path, output_name+"."+video_type)))
        return
    
    # assert plot_type in ["smpl", "smplx"]
    assert video_type in ["gif", "mp4"]
    assert len(joints_list) == len(prefixs)
    
    # Calculate xyz limits and the range
    # joints_cat = np.concatenate(joints_list, axis=0)
    # min_xyz = np.min(joints_cat.reshape((-1, 3)), axis=0)
    # max_xyz = np.max(joints_cat.reshape((-1, 3)), axis=0)
    # ctr_xyz = 0.5 * (min_xyz + max_xyz)
    # hrange = 0.5 * np.max(max_xyz - min_xyz)
    
    if colors is None:
        colors = ['k-'] * len(joints_list[0])
    
    ctr_xyz_list = []
    hrange_list = []
    for joints in joints_list:
        # joints is a [N, J, 3] numpy array
        min_xyz = np.min(joints.reshape((-1, 3)), axis=0)
        max_xyz = np.max(joints.reshape((-1, 3)), axis=0)
        ctr_xyz = 0.5 * (min_xyz + max_xyz)
        hrange = 0.5 * np.max(max_xyz - min_xyz)
        ctr_xyz_list.append(ctr_xyz)
        hrange_list.append(hrange)
        
    # Multithreading plotting
    indices_list = []
    step_size = joints_list[0].shape[0] // num_proc \
        if joints_list[0].shape[0] % num_proc == 0 \
            else joints_list[0].shape[0] // num_proc + 1
    
    for i in range(num_proc):
        indices_list.append(np.arange(i * step_size, (i + 1) * step_size).tolist())
            
    for k in range(len(joints_list)):
        job_list = []
        for i in range(num_proc):
            job = Process(target=plot_multiple_3d, 
                          args=(joints_list[k], 
                                colors, 
                                indices_list[i], 
                                os.path.join(output_path, "temp"), 
                                ctr_xyz_list[k], hrange_list[k], prefixs[k], ego_centric[k], plot_types[k]))
            job.start()
            job_list.append(job)
        
        for job in job_list:
            job.join()
        
    # Generate video
    animate_multiple(os.path.join(output_path, "temp"), 
                     os.path.join(output_path, output_name+"."+video_type), 
                     prefixs, fmt=video_type, fps=fps)
    
def resize(inp_motion, target_length):
    """
    :inp_motion: [T, J, 3]
    """
    input_length = inp_motion.shape[0]
    y = np.reshape(inp_motion, newshape=(input_length, -1)) # [T, J*3]
    x = np.arange(0, target_length)
    x = np.random.choice(target_length, size=(input_length,), replace=False)
    x = np.sort(x)
    if x[0] != 0: x[0] = 0
    if x[-1] != target_length-1: x[-1] = target_length-1
    new_x = np.arange(0, target_length)
    new_y = []
    for i in range(y.shape[-1]):
        f = interpolate.interp1d(x, y[:, i])
        new_y_ = f(new_x)
        new_y.append(new_y_)
    new_y = np.stack(new_y, axis=-1)
    return np.reshape(new_y, newshape=(target_length, -1, 3))
    
def main(args):
    
    input_path = os.path.join(args.base_dir, args.input_dir)
    output_path = os.path.join(args.base_dir, args.output_dir)
    joints_num = 22
    
    mean = np.load(args.mean_dir)
    std = np.load(args.std_dir)
    
    files = os.listdir(input_path)
    for idx, file in enumerate(files):
        
        try:
            tid = file.split(".")[0].split("_")[1]
        except:
            tid = "T000"
            
        data = np.load(os.path.join(input_path, file), allow_pickle=True).item()
        name = data["caption"][0]
        name = name.replace(",", "").replace(".", "").replace("/", "")
        words = name.split(" ")
        name = "_".join(words[:20])
        # name = name.replace(" ", "_").replace(",", "").replace(".", "").replace("/", "")    # Remove specific char

        gt_motion = data["gt"]["body"][0].transpose(1, 0)
        rc_motion = data["pred"]["body"][0].transpose(1, 0)
        
        # gt_motion = gt_motion * std + mean
        # rc_motion = rc_motion * std + mean
        
        gt_ric_data = recover_from_ric(
            torch.from_numpy(gt_motion).float(), joints_num).numpy()
        gt_ric_data = motion_temporal_filter(gt_ric_data)
        rc_ric_data = recover_from_ric(
            torch.from_numpy(rc_motion).float(), joints_num).numpy()
        rc_ric_data = motion_temporal_filter(rc_ric_data)
        
        # Rotate the joint
        rot = R.from_euler("x", 90, degrees=True)
        gt_ric_data = rotate_joints(joints=gt_ric_data, rot=rot)
        rc_ric_data = rotate_joints(joints=rc_ric_data, rot=rot)
        
        # Prepare the starting and endding primitives
        gt_len = gt_ric_data.shape[0]
        rc_len = rc_ric_data.shape[0]
        start_len = (gt_len * 2) // 5
        end_len = (gt_len * 2) // 5
        if args.visualize_gt:
            start_ric_data = gt_ric_data[:start_len]
            start_ric_data = np.concatenate([start_ric_data, np.repeat(start_ric_data[-1:], rc_len-start_len, axis=0)], axis=0)
            end_ric_data = gt_ric_data[-end_len:]
            end_ric_data = np.concatenate([np.repeat(end_ric_data[:1], rc_len-end_len, axis=0), end_ric_data], axis=0)
        if args.visualize_gt:
            colors = ['b-'] * start_len + ['k-'] * (rc_len-start_len-end_len) + ['g-'] * end_len
        else:
            colors = ['k-'] * rc_len
        
        # Resize the motions
        if args.visualize_gt:
            gt_length = gt_ric_data.shape[0]
            rc_length = rc_ric_data.shape[0]
            if gt_length < rc_length:
                gt_ric_data = resize(inp_motion=gt_ric_data, target_length=rc_length)
            elif rc_length < gt_length:
                rc_ric_data = resize(inp_motion=rc_ric_data, target_length=gt_length)
        
        if args.visualize_gt:
            render_animation(
                joints_list=[start_ric_data, rc_ric_data, end_ric_data], 
                output_path=os.path.join(output_path, tid), 
                output_name=name,  
                num_proc=args.num_proc, 
                plot_types=["smpl", "smpl", "smpl"], 
                prefixs=["start", "pred", "end"], 
                ego_centric=[args.ego_centric, args.ego_centric, args.ego_centric], 
                video_type=args.video_fmt, fps=args.fps, colors=colors)
        else:
            render_animation(
                joints_list=[rc_ric_data], 
                output_path=os.path.join(output_path, tid), 
                output_name=name,  
                num_proc=args.num_proc, 
                plot_types=["smpl"], 
                prefixs=["pred"], 
                ego_centric=[args.ego_centric], 
                video_type=args.video_fmt, fps=args.fps, colors=colors)
        
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, default="../dataset/BEAT_v0.2.1/beat_english_v0.2.1", help="")
    parser.add_argument('--base_dir', type=str, default='logs/avatar_gpt/eval/mlora_llama_13b/exp11/', help='')
    parser.add_argument('--input_dir', type=str, default="output/t2m", help="")
    parser.add_argument('--output_dir', type=str, default="animation/t2m", help="")
    parser.add_argument('--fps', type=int, default=30, help='path of training folder')
    parser.add_argument('--add_audio', type=str2bool, default=False, help='animate or not')
    parser.add_argument('--plot_type', type=str, default="2d", help='animate or not')
    parser.add_argument('--video_fmt', type=str, default="mp4", help='animate or not')
    parser.add_argument('--sample_rate', type=int, default=1, help='sampling rate')
    parser.add_argument('--num_proc', type=int, default=16, help='sampling rate')
    parser.add_argument('--max_length', type=int, default=10000, help='maximum length to animate')
    parser.add_argument('--visualize_gt', type=str2bool, default=True, help='visualize the GT joints or not')
    parser.add_argument('--post_process', type=str2bool, default=True, help='conduct post-process or not')
    parser.add_argument('--mean_dir', type=str, default="logs/avatar_gpt/eval/mlora_llama_13b/exp11/output/mean.npy", help='conduct post-process or not')
    parser.add_argument('--std_dir', type=str, default="logs/avatar_gpt/eval/mlora_llama_13b/exp11/output/std.npy", help='conduct post-process or not')
    parser.add_argument('--ego_centric', type=str2bool, default=False, help='conduct post-process or not')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    main(args)
    
    
    
    