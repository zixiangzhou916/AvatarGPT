import os, sys, argparse
sys.path.append(os.getcwd())
import glob
try:
    from omegaconf import OmegaConf
except:
    pass
import yaml
import random
import pickle
from tqdm import tqdm
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F
from networks import smplx_code
# from networks.flame.DecaFLAME import FLAME
from tools.postprocess import (smooth_joints, 
                               correct_sudden_jittering, 
                               adaptive_correct_sudden_jittering)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
# from PIL import Image, ImageTk, ImageSequence
from multiprocessing import Process
import imageio
# from mesh_renderer import MeshRenderer

# import pyrender
import trimesh
# from pyrender.constants import RenderFlags

try:
    from moviepy.editor import *
    from moviepy.video.fx.all import crop
    from pydub import AudioSegment
except:
    print("Unable to import moviepy and pydub")

SMPL_GRAPH = [(15, 12), (12, 9), (9, 13), (13, 16), (16, 18), (18, 20), 
        (20, 22), (9, 14), (14, 17), (17, 19), (19, 21), 
        (21, 23), (9, 6), (6, 3), (3, 0), (0, 2), (0, 1), 
        (2, 5), (5, 8), (8, 11), (1, 4), (4, 7), (7, 10)]

SMPLX_GRAPH = [(15, 12), (12, 9), (9, 13), (13, 16), (16, 18), (18, 20), 
        (9, 14), (14, 17), (17, 19), (19, 21), 
        (9, 6), (6, 3), (3, 0), (0, 2), (0, 1), 
        (2, 5), (5, 8), (8, 11), (1, 4), (4, 7), (7, 10), 
        # Left hand
        (20, 25), (67, 27), (25, 26), (26, 27), # left index
	    (20, 28), (68, 30), (28, 29), (29, 30), # left middel
	    (20, 31), (70, 33), (31, 32), (32, 33), # left pinky
	    (20, 34), (69, 36), (34, 35), (35, 36), # left ring
	    (20, 37), (66, 39), (37, 38), (38, 39),	# left thumb
        # Right hand
        (21, 40), (72, 42), (40, 41), (41, 42), # right index
        (21, 43), (73, 45), (43, 44), (44, 45), # right middle
        (21, 46), (75, 48), (46, 47), (47, 48), # right pinky
        (21, 49), (74, 51), (49, 50), (50, 51), # right ring
        (21, 52), (71, 54), (52, 53), (53, 54), # right thumb
        ]

FLAME_GRAPH = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), 
    (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), 
    (12, 13), (13, 14), (14, 15), (15, 16), 
    (17, 18), (18, 19), (19, 20), (20, 21), 
    (22, 23), (23, 24), (24, 25), (25, 26), 
    (36, 37), (37, 38), (38, 39), (36, 41), (40, 41), (39, 40), 
    (42, 43), (43, 44), (44, 45), (42, 47), (47, 46), (45, 46), 
    (27, 28), (28, 29), (29, 30), 
    (31, 32), (32, 33), (33, 34), (34, 35), 
    (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), 
    (60, 61), (61, 62), (62, 63), (63, 64), 
    (48, 59), (59, 58), (58, 57), (57, 56), (56, 55), (55, 54), 
    (60, 67), (67, 66), (66, 65), (65, 64), 
    (48, 60), (64, 54)
]

VALID_JOINTS_INDICES = [k[0] for k in SMPLX_GRAPH] + [k[1] for k in SMPLX_GRAPH]
VALID_JOINTS_INDICES = list(set(VALID_JOINTS_INDICES))


smplx_cfg = dict(
    model_path="networks/smpl-x/SMPLX_NEUTRAL_2020.npz", 
    model_type="smplx", 
    gender="neutral", 
    use_face_contour=True, 
    use_pca=True, 
    flat_hand_mean=False, 
    use_hands=True, 
    use_face=True, 
    num_pca_comps=12, 
    num_betas=300, 
    num_expression_coeffs=100,
)

smpl_cfg = dict(
    model_path="networks/smpl/SMPL_NEUTRAL.pkl", 
    model_type="smpl", 
    gender="neutral", 
    batch_size=1,
)

try:
    with open("networks/flame/cfg.yaml", "r") as f:
        flame_cfg = OmegaConf.load(f)
    flame_cfg = flame_cfg.coarse.model
except:
    pass

def convert_smpl2joints(
    smpl_model, body_pose, **kwargs
):
    """
    :param smplx_model: 
    :param body_pose: [batch_size, nframes, 75]
    """
    B, T = body_pose.shape[:2]
    device = body_pose.device
    
    transl = body_pose[..., :3]
    global_orient = body_pose[..., 3:6]
    body_pose = body_pose[..., 6:]
    
    output = smpl_model(
        global_orient=global_orient.reshape(B*T, 1, -1), 
        body_pose=body_pose.reshape(B*T, -1, 3), 
        transl=transl.reshape(B*T, -1)
    )
    
    joints = output.joints.reshape(B, T, -1, 3)
    vertices3d = output.vertices.reshape(B, T, -1, 3)
    
    return {"joints": joints[:, :, :24], "vertices": vertices3d}

def convert_smplx2joints(
    smplx_model, body_pose, left_pose, right_pose, **kwargs
):
    """
    :param smplx_model: 
    :param body_pose: [batch_size, nframes, 69]
    :param left_pose: [batch_size, nframes, 12]
    :param right_pose: [batch_size, nframes, 12]
    """
    B, T = body_pose.shape[:2]
    device = body_pose.device
    
    transl = body_pose[..., :3]
    global_orient = body_pose[..., 3:6]
    body_pose = body_pose[..., 6:]
    if body_pose.shape[-1] == 69:
        body_pose = body_pose[..., :-6]
    elif body_pose.shape[-1] == 75:
        body_pose = body_pose[..., :-12]
    
    print(body_pose.shape, left_pose.shape, right_pose.shape)
    # exit(0)
    betas = torch.zeros(B*T, 300).float().to(device)
    expression = torch.zeros(B*T, 100).float().to(device)
    jaw_pose = torch.zeros(B*T, 3).float().to(device)
    leye_pose = torch.zeros(B*T, 3).float().to(device)
    reye_pose = torch.zeros(B*T, 3).float().to(device)
    
    output = smplx_model(
        betas=betas, 
        global_orient=global_orient.reshape(B*T, 1, -1), 
        body_pose=body_pose.reshape(B*T, -1), 
        left_hand_pose=left_pose.reshape(B*T, -1), 
        right_hand_pose=right_pose.reshape(B*T, -1), 
        expression=expression, 
        jaw_pose=jaw_pose, 
        leye_pose=leye_pose, 
        reye_pose=reye_pose,
        transl=transl.reshape(B*T, -1)
    )
    
    
    joints = output.joints.reshape(B, T, -1, 3)
    vertices3d = output.vertices.reshape(B, T, -1, 3)
    # print(joints.shape)
    # exit(0)
    
    return {"joints": joints, "vertices": vertices3d}

def convert_flame2joints(
    flame_model, expr_pose=None, neck_pose=None, jaw_pose=None, leye_pose=None, reye_pose=None, **kwargs
):
    """
    :param expression: [batch_size, seq_len, 100]
    :param neck_pose: [batch_size, seq_len, 3]
    :param jaw_pose: [batch_size, seq_len, 3]
    :param leye_pose: [batch_size, seq_len, 3]
    :param reye_pose: [batch_size, seq_len, 3]
    """
    B, T, _ = expr_pose.shape
    shapes = torch.zeros(B*T, 300).float().to(expr_pose.device)
    # pose = torch.zeros(B*T, 3).float().to(expr_pose.device)
    pose = neck_pose.reshape(B*T, -1) if neck_pose is not None else torch.zeros(B*T, 3).float().to(expr_pose.device)
    pose = torch.cat([pose, jaw_pose.reshape(B*T, -1)], dim=-1)
    eye_pose = torch.cat([leye_pose, reye_pose], dim=-1)
    
    results = flame_model(shapes, 
                          expr_pose.reshape(B*T, -1), 
                          pose, 
                          eye_pose.reshape(B*T, -1))
    vertices = results[0].reshape(B, T, -1, 3)
    lmk_2d = results[1].reshape(B, T, -1, 3)
    lmk_3d = results[2].reshape(B, T, -1, 3)
    
    return {"vertices": vertices, "joints": lmk_3d}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def plot_smplx_3d(joints, output_name, orient=None, ctr_xyz=None, hrange=None):

    if ctr_xyz is None or hrange is None:
        min_xyz = np.min(joints, axis=0)
        max_xyz = np.max(joints, axis=0)
        ctr_xyz = 0.5 * (min_xyz + max_xyz)
        hrange = 0.5 * np.max(max_xyz - min_xyz)
    
    pxs = np.arange(ctr_xyz[0] - hrange, ctr_xyz[0] + hrange + 0.01, 2.0 * hrange)       # global plane
    pys = np.arange(ctr_xyz[1] - hrange, ctr_xyz[1] + hrange + 0.01, 2.0 * hrange)
    pxs, pys = np.meshgrid(pxs, pys)
    pzs = np.zeros_like(pxs)
    
    fig = plt.figure(figsize=(10, 10), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(ctr_xyz[0] - hrange, ctr_xyz[0] + hrange)
    ax.set_ylim(ctr_xyz[1] - hrange, ctr_xyz[1] + hrange)
    ax.set_zlim(ctr_xyz[2] - hrange, ctr_xyz[2] + hrange)
    
    ax.plot_surface(pxs, pys, pzs, rstride=1, cstride=1, facecolors="green", alpha=0.5)
    ax.scatter(joints[VALID_JOINTS_INDICES, 0], joints[VALID_JOINTS_INDICES, 1], joints[VALID_JOINTS_INDICES, 2], marker='.', c='red', s=10, alpha=1)
    
    for (i, j) in SMPLX_GRAPH:
        if i >= joints.shape[0] or j >= joints.shape[0]: continue
        ax.plot([joints[i, 0], joints[j, 0]], 
                [joints[i, 1], joints[j, 1]], 
                [joints[i, 2], joints[j, 2]], 'k-', lw=1.0)
            
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(output_name)
    plt.close()
    
def plot_smpl_3d(joints, output_name, orient=None, ctr_xyz=None, hrange=None):
    
    if ctr_xyz is None or hrange is None:
        min_xyz = np.min(joints, axis=0)
        max_xyz = np.max(joints, axis=0)
        ctr_xyz = 0.5 * (min_xyz + max_xyz)
        hrange = 0.5 * np.max(max_xyz - min_xyz)
    
    pxs = np.arange(ctr_xyz[0] - hrange, ctr_xyz[0] + hrange + 0.01, 2.0 * hrange)       # global plane
    pys = np.arange(ctr_xyz[1] - hrange, ctr_xyz[1] + hrange + 0.01, 2.0 * hrange)
    pxs, pys = np.meshgrid(pxs, pys)
    pzs = np.zeros_like(pxs)
    
    fig = plt.figure(figsize=(10, 10), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(ctr_xyz[0] - hrange, ctr_xyz[0] + hrange)
    ax.set_ylim(ctr_xyz[1] - hrange, ctr_xyz[1] + hrange)
    ax.set_zlim(ctr_xyz[2] - hrange, ctr_xyz[2] + hrange)
    
    ax.plot_surface(pxs, pys, pzs, rstride=1, cstride=1, facecolors="green", alpha=0.5)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], marker='.', c='red', s=10, alpha=1)
    
    for (i, j) in SMPL_GRAPH:
        if i >= joints.shape[0] or j >= joints.shape[0]: continue
        ax.plot([joints[i, 0], joints[j, 0]], 
                [joints[i, 1], joints[j, 1]], 
                [joints[i, 2], joints[j, 2]], 'k-', lw=1.0)
            
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(output_name)
    plt.close()
    
def plot_flame_3d(joints, output_name, orient=None, ctr_xyz=None, hrange=None):
    
    if ctr_xyz is None or hrange is None:
        min_xyz = np.min(joints, axis=0)
        max_xyz = np.max(joints, axis=0)
        ctr_xyz = 0.5 * (min_xyz + max_xyz)
        hrange = 0.5 * np.max(max_xyz - min_xyz)
    
    pxs = np.arange(ctr_xyz[0] - hrange, ctr_xyz[0] + hrange + 0.01, 2.0 * hrange)       # global plane
    pys = np.arange(ctr_xyz[1] - hrange, ctr_xyz[1] + hrange + 0.01, 2.0 * hrange)
    pxs, pys = np.meshgrid(pxs, pys)
    pzs = np.zeros_like(pxs)
    
    fig = plt.figure(figsize=(10, 10), dpi=100)
    # ax = fig.add_subplot(111, projection="3d")
    ax = fig.add_subplot(111)
    
    ax.set_xlim(ctr_xyz[0] - hrange - 0.1, ctr_xyz[0] + hrange + 0.1)
    ax.set_ylim(ctr_xyz[1] - hrange - 0.1, ctr_xyz[1] + hrange + 0.1)
    # ax.set_zlim(ctr_xyz[2] - hrange, ctr_xyz[2] + hrange)
    
    # ax.plot_surface(pxs, pys, rstride=1, cstride=1, facecolors="green", alpha=0.5)
    ax.scatter(joints[:, 0], joints[:, 1], marker='.', c='red', s=10, alpha=1)
    
    for (i, j) in FLAME_GRAPH:
        if i >= joints.shape[0] or j >= joints.shape[0]: continue
        ax.plot([joints[i, 0], joints[j, 0]], 
                [joints[i, 1], joints[j, 1]], 
                # [joints[i, 2], joints[j, 2]], 
                'k-', lw=1.0)
            
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(output_name)
    plt.close()

def plot_multiple_3d(joints, indices, output_path, ctr_xyz, hrange, prefix, plot_type="smplx"):
    assert plot_type in ["smpl", "smplx", "flame"]
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for idx in indices:
        if idx >= joints.shape[0]: 
            continue
        joint = joints[idx] # [J, 3]
        output_name = os.path.join(output_path, "{:s}_{:d}.png".format(prefix, idx))
        if plot_type == "smpl":
            plot_smpl_3d(joints=joint, output_name=output_name, orient=None, ctr_xyz=ctr_xyz, hrange=hrange)
        elif plot_type == "smplx":
            plot_smplx_3d(joints=joint, output_name=output_name, orient=None, ctr_xyz=ctr_xyz, hrange=hrange)
        elif plot_type == "flame":
            plot_flame_3d(joints=joint, output_name=output_name, orient=None, ctr_xyz=ctr_xyz, hrange=hrange)
        else:
            raise ValueError("plot type {:s} not recognized!".format(plot_type))    

def animate(input_path, output_name, fmt="gif", fps=30):
    assert fmt in ["gif", "mp4"]
    
    files = [int(f.split(".")[0]) for f in os.listdir(input_path) if ".png" in f]
    files = ["{:d}.png".format(f) for f in sorted(files)]
    
    if fmt == "gif":
        # Generate GIF from images
        with imageio.get_writer(output_name, mode="I") as writer:
            for file in tqdm(files, desc="Generating GIF..."):
                img = np.array(imageio.imread(os.path.join(input_path, file)), dtype=np.uint8)
                writer.append_data(img[..., :3])
                os.remove(os.path.join(input_path, file))
    elif fmt == "mp4":
        # Generate MP4 from images
        with imageio.get_writer(output_name, fps=fps) as writer:
            for file in tqdm(files, desc="Generating MP4..."):
                img = np.array(imageio.imread(os.path.join(input_path, file)), dtype=np.uint8)
                writer.append_data(img[..., :3])
                os.remove(os.path.join(input_path, file))
    else:
        raise ValueError("format {:s} not recognized!".format(fmt))
    
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

def render_animation(joints_list, output_path, output_name, num_proc=1, plot_types="smpl", prefixs=[], video_type="gif", fps=30):
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
                                indices_list[i], 
                                os.path.join(output_path, "temp"), 
                                ctr_xyz_list[k], hrange_list[k], prefixs[k], plot_types[k]))
            job.start()
            job_list.append(job)
        
        for job in job_list:
            job.join()
        
    # Generate video
    animate_multiple(os.path.join(output_path, "temp"), 
                     os.path.join(output_path, output_name+"."+video_type), 
                     prefixs, fmt=video_type, fps=fps)

def render_mesh_animation(joints_list, vertices_list, faces, 
                          output_path, output_name, 
                          num_proc=1, 
                          plot_type="smpl", 
                          prefixs=[], 
                          video_type="gif", 
                          fps=30):
    """Render videos of input joints_list.
    :param joints_list: list of joints [N, J, 3]
    :param output_path: directory of output path
    :param output_name: name of output video
    """
    if os.path.exists(os.path.join(output_path, output_name+"."+video_type)):
        print("{:s} already animated".format(os.path.join(output_path, output_name+"."+video_type)))
        return
    
    assert plot_type in ["smpl", "smplx"]
    assert video_type in ["gif", "mp4"]
    assert len(joints_list) == len(prefixs)
    
    # Calculate xyz limits and the range
    Render = MeshRenderer(faces=faces, img_h=1024, img_w=1024, yfov=5.0, x_angle=60)
    for (vertices, prefix) in zip(vertices_list, prefixs):
        Render.render(vertices=vertices, output_path=output_path, output_name=output_name, prefix=prefix)
    
    Render.animate(output_path=output_path, output_name=output_name, prefixs=prefixs, fps=fps)
   
def add_audio(input_video_file, input_audio_file, output_video_file):
    print('-' * 10, input_video_file)
    video_clip = VideoFileClip(input_video_file)
    video_duration = video_clip.duration
    
    audio_clip = AudioFileClip(input_audio_file)
    audio_duration = audio_clip.duration
    
    print(video_duration, "|", audio_duration)
    end_frame = max(video_duration, audio_duration)
    start_frame = end_frame - min(video_duration, audio_duration)
    
    if video_duration < audio_duration:
        audio_clip = audio_clip.subclip(start_frame, end_frame)
    elif audio_duration < video_duration:
        video_clip = video_clip.subclip(start_frame, end_frame)
    
    final_clip = video_clip.set_audio(audio_clip)
    # final_clip.write_videofile(
    #     output_video_file, codec='mpeg4', audio_codec='pcm_s16le', bitrate='50000k')
    final_clip.write_videofile(output_video_file)

def add_s2m_audio(base_video_path, base_audio_path, folder_id, base_name):
    """Add speech audio to video (BEAT dataset).
    """
    try:
        input_video_path = os.path.join(base_video_path, base_name+".mp4")
        # output_video_path = os.path.join(base_video_path, base_name+"_audio.avi")
        output_video_path = os.path.join(base_video_path, base_name+"_audio.mp4")
        input_audio_path = os.path.join(base_audio_path, folder_id, base_name+".wav")
        print(input_video_path, input_audio_path, base_audio_path, folder_id)
        add_audio(input_video_path, input_audio_path, output_video_path)
    except:
        print("speech audio is not added to the vidoe!")

def add_flame_audio(base_video_path, base_audio_path, folder_id, base_name):
    """Add speech audio to video (BEAT dataset).
    """
    try:
        input_video_path = os.path.join(base_video_path, base_name+".mp4")
        # output_video_path = os.path.join(base_video_path, base_name+"_audio.avi")
        output_video_path = os.path.join(base_video_path, base_name+"_audio.mp4")
        input_audio_path = os.path.join(base_audio_path, folder_id, base_name+".mp3")
        print(input_video_path, input_audio_path, base_audio_path, folder_id)
        add_audio(input_video_path, input_audio_path, output_video_path)
    except:
        print("speech audio is not added to the vidoe!")

def add_a2m_audio(base_video_path, base_audio_path, base_name):
    """Add music to video (AIST++ dataset).
    """
    try:
        audio_name = base_name.split("_")[4]
        input_video_path = os.path.join(base_video_path, base_name+".mp4")
        # output_video_path = os.path.join(base_video_path, base_name+"_audio.avi")
        output_video_path = os.path.join(base_video_path, base_name+"_audio.mp4")
        input_audio_path = os.path.join(base_audio_path, audio_name+".wav")
        add_audio(input_video_path, input_audio_path, output_video_path)
    except:
        print("music audio is not added to the video!")
        # raise ValueError
 
def plot_tokens_full_body(data, output_name):
    
    try:
        gt_body = data["gt_tokens"]["body"]
        gt_left = data["gt_tokens"]["left"]
        gt_right = data["gt_tokens"]["right"]

        pred_body = data["pred_tokens"]["body"]
        pred_left = data["pred_tokens"]["left"]
        pred_right = data["pred_tokens"]["right"]
        
        word_tokens = data["word_tokens"][0, ::4]
        onset_env = np.transpose(data["audio"][0, 52, ::4])
        onset_beat = np.transpose(data["audio"][0, 53, ::4])
        length = min(gt_body.shape[1], pred_body.shape[1])
        
        for i in range(0, length, 150):
            j = min(i+150, length)
            x = np.arange(0, j-i)
            
            fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(50, 15))
            f1, = axs[0, 0].plot(x, gt_body[0, i:j], linestyle='solid', linewidth=0.5, color="r")
            f2, = axs[0, 0].plot(x, pred_body[0, i:j], linestyle='solid', linewidth=0.5, color="b")
            axs[0, 0].set_title("body tokens", fontsize=40)
            axs[0, 0].legend([f1, f2], ["gt", "pred"], fontsize=40, loc='upper right')
            
            f1, = axs[1, 0].plot(x, gt_left[0, i:j], linestyle='solid', linewidth=0.5, color="r")
            f2, = axs[1, 0].plot(x, pred_left[0, i:j], linestyle='solid', linewidth=0.5, color="b")
            axs[1, 0].set_title("left hand tokens", fontsize=40)
            axs[1, 0].legend([f1, f2], ["gt", "pred"], fontsize=40, loc='upper right')

            f1, = axs[2, 0].plot(x, gt_right[0, i:j], linestyle='solid', linewidth=0.5, color="r")
            f2, = axs[2, 0].plot(x, pred_right[0, i:j], linestyle='solid', linewidth=0.5, color="b")
            axs[2, 0].set_title("right hand tokens", fontsize=40)
            axs[2, 0].legend([f1, f2], ["gt", "pred"], fontsize=40, loc='upper right')

            axs[0, 1].plot(x, word_tokens[i:j], linestyle='solid', linewidth=0.5, color='g')
            axs[0, 1].set_title("word tokens", fontsize=40)
            axs[1, 1].plot(x, onset_env[i:j], linestyle='solid', linewidth=0.5, color='black')
            axs[1, 1].set_title("audio onset env", fontsize=40)
            axs[2, 1].plot(x, onset_beat[i:j], linestyle='solid', linewidth=0.5, color='black')
            axs[2, 1].set_title("audio onset beat", fontsize=40)
            
            plt.tight_layout()
            plt.show()
            plt.savefig(output_name+"_{:d}_{:d}.png".format(i, j))
            plt.close()
            print('---', output_name+"_{:d}_{:d}.png".format(i, j))
    
    except:
        print("Unable to draw")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=3, help='')
    parser.add_argument('--vis_num_samples', type=int, default=3, help='')

    parser.add_argument('--audio_path', type=str, default="../dataset/BEAT_v0.2.1/beat_english_v0.2.1", help="")
    parser.add_argument('--base_folder', type=str, default='ude', help='')
    parser.add_argument('--sub_base_folder', type=str, default='debug', help='')
    parser.add_argument('--input_folder', type=str, default='t2m', help='path of training folder')
    parser.add_argument('--output_folder', type=str, default='t2m', help='path of training folder')
    parser.add_argument('--cmp_folder', type=str, default='cmp_t2m', help='path of training folder')
    parser.add_argument('--fps', type=int, default=30, help='path of training folder')
    parser.add_argument('--animate', type=str2bool, default=True, help='animate or not')
    parser.add_argument('--draw_tokens', type=str2bool, default=False, help='animate or not')
    parser.add_argument('--add_audio', type=str2bool, default=False, help='animate or not')
    parser.add_argument('--plot_type', type=str, default="2d", help='animate or not')
    parser.add_argument('--video_fmt', type=str, default="mp4", help='animate or not')
    parser.add_argument('--sample_rate', type=int, default=1, help='sampling rate')
    parser.add_argument('--num_proc', type=int, default=16, help='sampling rate')
    parser.add_argument('--max_length', type=int, default=10000, help='maximum length to animate')
    parser.add_argument('--visualize_gt', type=str2bool, default=False, help='visualize the GT joints or not')
    parser.add_argument('--post_process', type=str2bool, default=True, help='conduct post-process or not')
    
    args = parser.parse_args()
    return args

def main(args):
    
    # Build SMPL-X model
    smplx_model = smplx_code.create(**smplx_cfg)
    smpl_model = smplx_code.create(**smpl_cfg)
    # flame_model = FLAME(flame_cfg, flame_full=True)
    
    num_samples = args.num_samples
    base_folder = args.base_folder
    sub_base_folder = args.sub_base_folder
    input_folder = args.input_folder
    output_folder = args.output_folder
    cmp_folder = args.cmp_folder

    FPS = args.fps
    
    input_path = "logs/avatar_gpt/eval/{:s}/{:s}/{:s}".format(base_folder, sub_base_folder, input_folder)
    audio_path = args.audio_path
    output_path = "logs/avatar_gpt/eval/{:s}/animation/{:s}".format(base_folder, output_folder)
    image_path = "logs/avatar_gpt/eval/{:s}/image/{:s}".format(base_folder, output_folder)

    files = os.listdir(input_path)

    for idx, file in enumerate(files):
        
        try:
            tid = file.split(".")[0].split("_")[1]
        except:
            tid = "T000"
        
        data = np.load(os.path.join(input_path, file), allow_pickle=True).item()
        name = data["caption"][0]
        name = name.replace(" ", "_").replace(",", "").replace(".", "").replace("/", "")    # Remove specific char
        
        if len(name) > 150: name = name[:150]
        try:
            audio = torch.from_numpy(data["audio"]).permute(0, 2, 1)
        except:
            keys = list(data["pred"].keys())
            audio = torch.zeros(1, data["pred"][keys[0]].shape[2], 438)
        gt_poses = {key+"_pose": torch.from_numpy(val[..., :args.max_length]).permute(0, 2, 1).float() for key, val in data["gt"].items()}
        rc_poses = {key+"_pose": torch.from_numpy(val[..., :args.max_length]).permute(0, 2, 1).float() for key, val in data["pred"].items()}
        # rc_poses = {"body_pose": torch.from_numpy(data["pred"]["body"][..., :args.max_length]).permute(0, 2, 1).float()}
        print(gt_poses["body_pose"].shape, "|", rc_poses["body_pose"].shape)

        if "s2m" in input_path:
            folder_id = name.split("_")[0]
        else:
            folder_id = ""
            
        # max_len = audio.shape[1]
        # if "a2m" not in input_path: # We treat a2m specially because the length of GT of AIST++ dataset does align with audio.
        #     for key, val in gt_poses.items(): max_len = min(max_len, val.shape[1])
        # for key, val in rc_poses.items(): max_len = min(max_len, val.shape[1])
        # # max_len = min(min(gt_poses["body_pose"].shape[1], rc_poses["body_pose"].shape[1]), audio.shape[1])
        
        # for key, val in gt_poses.items():
        #     if "a2m" in input_path: # We treat a2m specially because the length of GT of AIST++ dataset does align with audio.
        #         val = F.interpolate(val.transpose(2, 1), size=(max_len), mode='linear', align_corners=True).transpose(2, 1)
        #         gt_poses[key] = val[:, ::args.sample_rate]
        #     else:
        #         gt_poses[key] = val[:, :max_len][:, ::args.sample_rate]
        # for key, val in rc_poses.items():
        #     rc_poses[key] = val[:, :max_len][:, ::args.sample_rate]
        
        # Plot predicted tokens
        if args.draw_tokens:
            print("Plotting tokens")
            plot_tokens_full_body(data, os.path.join(image_path, name))
        
        # Convert smplx parameters to joints
        if len(gt_poses) == 3:
            gt_output = convert_smplx2joints(smplx_model, **gt_poses)
            gt_repre = "smplx"
            gt_faces = smplx_model.faces
        elif len(gt_poses) == 1:
            gt_output = convert_smpl2joints(smpl_model, **gt_poses)
            gt_faces = smpl_model.faces
            gt_repre = "smpl"
        elif len(gt_poses) == 5 or len(gt_poses) == 4:
            gt_output = convert_flame2joints(flame_model, **gt_poses)
            gt_faces = flame_model.faces_tensor.data.cpu().numpy()
            gt_repre = "flame"
            
        gt_joints = gt_output["joints"].data.cpu().numpy()
        gt_vertices = gt_output["vertices"].data.cpu().numpy()
        if len(rc_poses) == 3:
            rc_output = convert_smplx2joints(smplx_model, **rc_poses)
            rc_repre = "smplx"
            rc_faces = smplx_model.faces
        elif len(rc_poses) == 1:
            rc_output = convert_smpl2joints(smpl_model, **rc_poses)
            rc_repre = "smpl"
            rc_faces = smpl_model.faces
        elif len(rc_poses) == 5 or len(gt_poses) == 4:
            rc_output = convert_flame2joints(flame_model, **rc_poses)
            rc_faces = flame_model.faces_tensor.data.cpu().numpy()
            rc_repre = "flame"
            
        rc_joints = rc_output["joints"].data.cpu().numpy()
        rc_vertices = rc_output["vertices"].data.cpu().numpy()

        # mesh = trimesh.Trimesh(vertices=rc_vertices[0, 0], faces=rc_faces)
        # mesh.export("test.obj")
        # print(mesh)
        # exit(0)
        
        if args.post_process:
            rc_joints = adaptive_correct_sudden_jittering(rc_joints[0], threshold=0.1)[np.newaxis]
        
        faces = smplx_model.faces
        print('[{:d}/{:d}]'.format(idx, len(files)), gt_joints.shape, rc_joints.shape, audio.shape)
        
        if args.animate:
            # # Render the mesh animation
            # print("Start to render mesh enimation")
            # render_mesh_animation(joints_list=[gt_joints[0], rc_joints[0]], 
            #                       vertices_list=[gt_vertices[0], rc_vertices[0]], 
            #                       output_path=output_path, 
            #                       output_name=file.replace(".npy", ""), 
            #                       faces=faces, 
            #                       num_proc=args.num_proc, 
            #                       plot_type="smplx", 
            #                       prefixs=["gt", "pred"], 
            #                       video_type=args.video_fmt, fps=FPS)

            # Render the animation
            if not os.path.exists(os.path.join(output_path, tid, file.replace("npy", args.video_fmt))):
                if args.visualize_gt:
                    render_animation(joints_list=[gt_joints[0], rc_joints[0]], 
                                     output_path=os.path.join(output_path, tid), 
                                    #  output_name=file.replace(".npy", ""), 
                                     output_name=name,  
                                     num_proc=args.num_proc, 
                                     plot_types=[gt_repre, rc_repre], 
                                     prefixs=["gt", "pred"], 
                                     video_type=args.video_fmt, fps=FPS)
                else:
                    render_animation(joints_list=[rc_joints[0]], 
                                     output_path=os.path.join(output_path, tid), 
                                     output_name=name,  
                                     num_proc=args.num_proc, 
                                     plot_types=[rc_repre], 
                                     prefixs=["pred"], 
                                     video_type=args.video_fmt, fps=FPS)
        
            # Add audio to animation
            if "a2m" in args.input_folder and args.add_audio:
                add_a2m_audio(base_video_path=os.path.join(output_path, tid), 
                              base_audio_path=audio_path, 
                              base_name=name)
            elif "s2m" in args.input_folder and args.add_audio:
                add_s2m_audio(base_video_path=os.path.join(output_path, tid), 
                              base_audio_path=audio_path, 
                              folder_id=folder_id, 
                              base_name=name)
            elif "flame" in args.input_folder and args.add_audio:
                add_flame_audio(base_video_path=os.path.join(output_path, tid), 
                              base_audio_path=audio_path, 
                              folder_id=folder_id, 
                              base_name=name)
            
            # add_a2m_audio(base_video_path=os.path.join(output_path, tid), 
            #               base_audio_path=audio_path, 
            #             #   folder_id=folder_id, 
            #               base_name=name)
            # input_video_file = os.path.join(output_path, name+".mp4")
            # input_audio_file = os.path.join(audio_path, folder_id, name+".wav")
            # output_video_file = os.path.join(output_path, name+"_audio.avi")
            # if not os.path.exists(output_video_file):
            #     try:
            #         add_audio(input_video_file, input_audio_file, output_video_file)
            #     except:
            #         print("audio is not added to the video!")
        
        
if __name__ == "__main__":
    
    args = parse_args()
    main(args)
