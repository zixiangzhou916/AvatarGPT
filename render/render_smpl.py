import os
cwd = os.getcwd()
import sys
import argparse
sys.path.append(cwd)
import glob
import random
import pickle
from tqdm import tqdm
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F
from smplx import SMPL

import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageSequence

matplotlib.use('Agg')

from multiprocessing import  Process
import imageio

try:
    from moviepy.editor import *
    from moviepy.video.fx.all import crop
    from pydub import AudioSegment
except:
    print("Unable to import moviepy and pydub")

AIST_POSE_EDGE = [(15, 12), (12, 9), (9, 13), (13, 16), (16, 18), (18, 20), 
        (20, 22), (9, 14), (14, 17), (17, 19), (19, 21), 
        (21, 23), (9, 6), (6, 3), (3, 0), (0, 2), (0, 1), 
        (2, 5), (5, 8), (8, 11), (1, 4), (4, 7), (7, 10)]

HAND_EDGE = [(0, 1), (2, 1), (3, 2), (4, 3), 
             (5, 0), (6, 5), (7, 6), (8, 7), (9, 8), 
             (10, 5), (11, 10), (12, 11), (13, 12), (14, 13), 
             (15, 0), (16, 15), (17, 16), (18, 17), (19, 18), 
             (20, 15), (21, 20), (22, 21), (23, 22)]

SMPLModel = SMPL(model_path="./networks/smpl", gender="NEUTRAL", batch_size=1)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def unpack_gif(src):
    image = Image.open(src)
    frames = []
    for frame in ImageSequence.Iterator(image):
        frm = np.asarray(frame.convert('RGBA'))[..., :3]
        h, w, _ = frm.shape
        frames.append(frm[100:401, 100:401, :])
    return frames
        
def pack_gif(frames, name):
    with imageio.get_writer(name, mode='I') as writer:
        for frame in frames:
            writer.append_data(frame)
        
def plot_smpl_poses(motion, indices, center, half_range, output_path, prefix, title):
    if title is not None:
        title_sp = title.split(' ')
        if len(title_sp) > 20:
            title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
        elif len(title_sp) > 10:
            title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    x_center, y_center, z_center = center
    for index in indices:
        if index >= motion.shape[0]: continue
        points = motion[index, ...]

        max_xyz = np.max(points, axis=0)
        min_xyz = np.min(points, axis=0)
        ctr_xyz = 0.5 * (min_xyz + max_xyz)

        # pxs = np.arange(ctr_xyz[0] - 1.5, ctr_xyz[0] + 1.51, 3.0)   # local plane
        # pys = np.arange(ctr_xyz[1] - 1.5, ctr_xyz[1] + 1.51, 3.0)

        pxs = np.arange(x_center - 1.5, x_center + 1.51, 3.0)       # global plane
        pys = np.arange(y_center - 1.5, y_center + 1.51, 3.0)
        
        pxs, pys = np.meshgrid(pxs, pys)
        pzs = np.zeros_like(pxs)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(x_center - half_range - 1.0, x_center + half_range + 1.0)
        ax.set_ylim(y_center - half_range - 1.0, y_center + half_range + 1.0)
        # ax.set_zlim(z_center - half_range - 0.1, z_center + half_range + 0.1)

        # ax.set_xlim(ctr_xyz[0] - 1.0, ctr_xyz[0] + 1.0)
        # ax.set_ylim(ctr_xyz[1] - 1.0, ctr_xyz[1] + 1.0)
        # ax.set_zlim(ctr_xyz[2] - 1.0, ctr_xyz[2] + 1.0)
        
        ax.set_zlim(-0.2, 1.8)
        ax.plot_surface(pxs, pys, pzs, rstride=1, cstride=1, facecolors="green", alpha=0.5)
        ax.scatter(points[..., 0], points[..., 1], points[..., 2], marker='o', c='b', s=10, alpha=1)
        
        # ax.scatter([0], [0], [0], marker='o', c='r', s=20, alpha=1)
        # right_foot = points[11]
        # ax.text(right_foot[0], right_foot[1], right_foot[2], "z={:.5f}".format(right_foot[2]))

        plt.xlabel('x')
        plt.ylabel('y')
        for (i, j) in AIST_POSE_EDGE:
            ax.plot([points[i, 0], points[j, 0]], 
                    [points[i, 1], points[j, 1]], 
                    [points[i, 2], points[j, 2]], 'k-', lw=2)
        if title is not None:
            # ax.set_title(title)
            fig.suptitle(title, fontsize=10)
        plt.axis('off')
        plt.savefig(os.path.join(output_path, '{:s}_{:d}.png'.format(prefix, index)))
        plt.close()

def plot_smpl_and_bvh_poses(smpl_motion, left_motion, right_motion, indices, center, half_range, output_path, prefix, title):
    if title is not None:
        title_sp = title.split(' ')
        if len(title_sp) > 20:
            title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
        elif len(title_sp) > 10:
            title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    x_center, y_center, z_center = center
    for index in indices:
        if index >= smpl_motion.shape[0]: continue
        points = smpl_motion[index, ...]
        l_points = left_motion[index, ...]
        r_points = right_motion[index, ...]

        r_wrist = points[21]
        l_wrist = points[20]
        r_points += r_wrist[None]
        l_points += l_wrist[None]

        max_xyz = np.max(points, axis=0)
        min_xyz = np.min(points, axis=0)
        ctr_xyz = 0.5 * (min_xyz + max_xyz)

        # pxs = np.arange(ctr_xyz[0] - 1.5, ctr_xyz[0] + 1.51, 3.0)   # local plane
        # pys = np.arange(ctr_xyz[1] - 1.5, ctr_xyz[1] + 1.51, 3.0)

        pxs = np.arange(x_center - 1.5, x_center + 1.51, 3.0)       # global plane
        pys = np.arange(y_center - 1.5, y_center + 1.51, 3.0)
        
        pxs, pys = np.meshgrid(pxs, pys)
        pzs = np.zeros_like(pxs)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(x_center - half_range - 1.0, x_center + half_range + 1.0)
        ax.set_ylim(y_center - half_range - 1.0, y_center + half_range + 1.0)
        
        ax.set_zlim(-0.2, 1.8)
        ax.plot_surface(pxs, pys, pzs, rstride=1, cstride=1, facecolors="green", alpha=0.5)
        ax.scatter(points[..., 0], points[..., 1], points[..., 2], marker='o', c='b', s=5, alpha=1)
        ax.scatter(l_points[..., 0], l_points[..., 1], l_points[..., 2], marker='o', c='g', s=1, alpha=1)
        ax.scatter(r_points[..., 0], r_points[..., 1], r_points[..., 2], marker='o', c='r', s=1, alpha=1)
        
        plt.xlabel('x')
        plt.ylabel('y')
        for (i, j) in AIST_POSE_EDGE:
            if i >= points.shape[0] or j >= points.shape[0]: continue
            ax.plot([points[i, 0], points[j, 0]], 
                    [points[i, 1], points[j, 1]], 
                    [points[i, 2], points[j, 2]], 'k-', lw=1)
        for (i, j) in HAND_EDGE:
            ax.plot([l_points[i, 0], l_points[j, 0]], 
                    [l_points[i, 1], l_points[j, 1]], 
                    [l_points[i, 2], l_points[j, 2]], 'k-', lw=0.5)
        for (i, j) in HAND_EDGE:
            ax.plot([r_points[i, 0], r_points[j, 0]], 
                    [r_points[i, 1], r_points[j, 1]], 
                    [r_points[i, 2], r_points[j, 2]], 'k-', lw=0.5)
        if title is not None:
            # ax.set_title(title)
            fig.suptitle(title, fontsize=10)
        plt.axis('off')
        plt.savefig(os.path.join(output_path, '{:s}_{:d}.png'.format(prefix, index)))
        plt.close()

def plot_bvh_poses(motion, indices, center, half_range, output_path, prefix, title):
    if title is not None:
        title_sp = title.split(' ')
        if len(title_sp) > 20:
            title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
        elif len(title_sp) > 10:
            title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    x_center, y_center, z_center = center
    for index in indices:
        if index >= motion.shape[0]: continue
        points = motion[index, ...]

        max_xyz = np.max(points, axis=0)
        min_xyz = np.min(points, axis=0)
        ctr_xyz = 0.5 * (min_xyz + max_xyz)

        pxs = np.arange(x_center - 0.2, x_center + 0.21, 0.4)       # global plane
        pys = np.arange(y_center - 0.2, y_center + 0.21, 0.4)
        
        pxs, pys = np.meshgrid(pxs, pys)
        pzs = np.zeros_like(pxs)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(x_center - half_range - 0.2, x_center + half_range + 0.2)
        ax.set_ylim(y_center - half_range - 0.2, y_center + half_range + 0.2)
        # ax.set_zlim(z_center - half_range - 0.1, z_center + half_range + 0.1)
        
        ax.set_zlim(-0.2, 1.8)
        ax.plot_surface(pxs, pys, pzs, rstride=1, cstride=1, facecolors="green", alpha=0.5)
        ax.scatter(points[..., 0], points[..., 1], points[..., 2], marker='o', c='b', s=10, alpha=1)
        
        plt.xlabel('x')
        plt.ylabel('y')
        for (i, j) in HAND_EDGE:
            ax.plot([points[i, 0], points[j, 0]], 
                    [points[i, 1], points[j, 1]], 
                    [points[i, 2], points[j, 2]], 'k-', lw=2)
        if title is not None:
            fig.suptitle(title, fontsize=10)
        plt.axis('off')
        plt.savefig(os.path.join(output_path, '{:s}_{:d}.png'.format(prefix, index)))
        plt.close()

def plot_full_body_poses(motion, indices, center, half_range, output_path, prefix, title):
    if title is not None:
        title_sp = title.split(' ')
        if len(title_sp) > 20:
            title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
        elif len(title_sp) > 10:
            title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    x_center, y_center, z_center = center
    for index in indices:
        if index >= motion.shape[0]: continue
        points = motion[index, ...]

        max_xyz = np.max(points, axis=0)
        min_xyz = np.min(points, axis=0)
        ctr_xyz = 0.5 * (min_xyz + max_xyz)

        pxs = np.arange(x_center - 1.5, x_center + 1.51, 3.0)       # global plane
        pys = np.arange(y_center - 1.5, y_center + 1.51, 3.0)
        
        pxs, pys = np.meshgrid(pxs, pys)
        pzs = np.zeros_like(pxs)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(x_center - half_range - 1.0, x_center + half_range + 1.0)
        ax.set_ylim(y_center - half_range - 1.0, y_center + half_range + 1.0)
        # ax.set_zlim(z_center - half_range - 0.1, z_center + half_range + 0.1)
        
        ax.set_zlim(-0.2, 1.8)
        ax.plot_surface(pxs, pys, pzs, rstride=1, cstride=1, facecolors="green", alpha=0.5)
        ax.scatter(points[..., 0], points[..., 1], points[..., 2], marker='o', c='b', s=2, alpha=1)
        
        body_joints, left_hand_joints, right_hand_joints = points[:22], points[22:22+24], points[22+24:]
        plt.xlabel('x')
        plt.ylabel('y')
        for (i, j) in AIST_POSE_EDGE:
            if i >= body_joints.shape[0] or j >= body_joints.shape[0]: continue
            ax.plot([body_joints[i, 0], body_joints[j, 0]], 
                    [body_joints[i, 1], body_joints[j, 1]], 
                    [body_joints[i, 2], body_joints[j, 2]], 'k-', lw=2)
        for (i, j) in HAND_EDGE:
            ax.plot([left_hand_joints[i, 0], left_hand_joints[j, 0]], 
                    [left_hand_joints[i, 1], left_hand_joints[j, 1]], 
                    [left_hand_joints[i, 2], left_hand_joints[j, 2]], 'r-', lw=1)
        for (i, j) in HAND_EDGE:
            ax.plot([right_hand_joints[i, 0], right_hand_joints[j, 0]], 
                    [right_hand_joints[i, 1], right_hand_joints[j, 1]], 
                    [right_hand_joints[i, 2], right_hand_joints[j, 2]], 'b-', lw=1)
        if title is not None:
            fig.suptitle(title, fontsize=10)
        plt.axis('off')
        plt.savefig(os.path.join(output_path, '{:s}_{:d}.png'.format(prefix, index)))
        plt.close()
        
        
# def plot_smpl_and_bvh_poses(motions, indices, center, half_range, output_path, prefix, title):
#     if title is not None:
#         title_sp = title.split(' ')
#         if len(title_sp) > 20:
#             title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
#         elif len(title_sp) > 10:
#             title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

#     x_center, y_center, z_center = center
#     for index in indices:
#         if index >= motions[0].shape[0]: continue
#         points = motions[0][index, ...]
#         raw_points = motions[1][index, ...]

#         max_xyz = np.max(points, axis=0)
#         min_xyz = np.min(points, axis=0)
#         ctr_xyz = 0.5 * (min_xyz + max_xyz)

#         pxs = np.arange(x_center - 1.5, x_center + 1.51, 3.0)       # global plane
#         pys = np.arange(y_center - 1.5, y_center + 1.51, 3.0)
        
#         pxs, pys = np.meshgrid(pxs, pys)
#         pzs = np.zeros_like(pxs)

#         fig = plt.figure(figsize=(5, 5))
#         ax = fig.add_subplot(111, projection='3d')
#         ax.set_xlim(x_center - half_range - 1.0, x_center + half_range + 1.0)
#         ax.set_ylim(y_center - half_range - 1.0, y_center + half_range + 1.0)

#         ax.set_zlim(-0.2, 1.8)
#         ax.plot_surface(pxs, pys, pzs, rstride=1, cstride=1, facecolors="green", alpha=0.5)
#         ax.scatter(points[..., 0], points[..., 1], points[..., 2], marker='o', c='b', s=10, alpha=1)
        
#         plt.xlabel('x')
#         plt.ylabel('y')
#         for (i, j) in AIST_POSE_EDGE:
#             ax.plot([points[i, 0], points[j, 0]], 
#                     [points[i, 1], points[j, 1]], 
#                     [points[i, 2], points[j, 2]], 'k-', lw=2)
#             ax.plot([raw_points[i, 0], raw_points[j, 0]], 
#                     [raw_points[i, 1], raw_points[j, 1]], 
#                     [raw_points[i, 2], raw_points[j, 2]], 'r-', lw=2)
#         if title is not None:
#             # ax.set_title(title)
#             fig.suptitle(title, fontsize=10)
#         plt.axis('off')
#         plt.savefig(os.path.join(output_path, '{:s}_{:d}.png'.format(prefix, index)))
#         plt.close()

def convert_smpl(smpl_model, smpl_param):
    """
    :param smpl_param: [nframes, dim]
    """
    smpl_trans, smpl_poses = smpl_param[..., :3], smpl_param[..., 3:]
    smpl_poses = np.reshape(smpl_poses, (-1, 24, 3))
    smpl_joints = smpl_model(global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
                            body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
                            transl=torch.from_numpy(smpl_trans).float())
    keypoints3d = smpl_joints.joints.detach().cpu().numpy()
    vertices3d = smpl_joints.vertices.detach().cpu().numpy()
    return keypoints3d, vertices3d

def reorg(data_inp):
    xs, ys, zs = data_inp[..., 0], data_inp[..., 1], data_inp[..., 2]
    zs *= -1.0
    data_out = np.stack((xs, zs, ys), axis=-1)
    return data_out

def reorg_new(data_inp):
    xs, ys, zs = data_inp[..., 0], data_inp[..., 1], data_inp[..., 2]
    zs *= -1.0
    return np.stack((xs, ys, zs), axis=-1)

def calc_range(data_list):
    xs = [data[..., 0] for data in data_list]
    ys = [data[..., 1] for data in data_list]
    zs = [data[..., 2] for data in data_list]
    xs = np.stack(xs, axis=-1)
    ys = np.stack(ys, axis=-1)
    zs = np.stack(zs, axis=-1)
    x_range = np.max(xs) - np.min(xs)
    y_range = np.max(ys) - np.min(ys)
    z_range = np.max(zs) - np.min(zs)
    x_ctr = 0.5 * (np.max(xs) + np.min(xs))
    y_ctr = 0.5 * (np.max(ys) + np.min(ys))
    z_ctr = 0.5 * (np.max(zs) + np.min(zs))
    half_range = 0.5 * max(x_range, max(y_range, z_range))
    return x_ctr, y_ctr, z_ctr, half_range

def render_anim(smpl_param, title="", output_path="", name=""):
    if not os.path.exists(output_path): os.makedirs(output_path)
    temp_output_path = os.path.join(output_path, "temp")
    if not os.path.exists(temp_output_path): os.makedirs(temp_output_path)

    with torch.no_grad():
        keypoints, vertices = convert_smpl(SMPLModel, smpl_param[0])
    print('--- start animating')
    keypoints = reorg(keypoints)
    raw_keypoints = reorg(smpl_param[1])
    x_ctr, y_ctr, z_ctr, half_range = calc_range([vertices])

    indices_list = []
    num_proc = 16
    step_size = vertices.shape[0] // num_proc if vertices.shape[0] % num_proc == 0 else (vertices.shape[0] // num_proc) + 1
    for i in range(num_proc):
        indices_list.append(np.arange(i * step_size, (i + 1) * step_size).tolist())

    process_list = []
    for i in range(num_proc):
        p = Process(target=plot_smpl_and_bvh_poses, args=([keypoints[:, :24], raw_keypoints], indices_list[i], [x_ctr, y_ctr, z_ctr], half_range, temp_output_path, "temp", title))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    """ convert pngs to avi """
    img_width, img_height = 1024, 1024
    size = (img_width, img_height)
    vout = cv2.VideoWriter(os.path.join(output_path, "{:s}.mp4".format(name)), cv2.VideoWriter_fourcc(*"mp4v"), int(20), size)
    for id in range(vertices.shape[0]):
        image = imageio.imread(os.path.join(temp_output_path, "temp_{:d}.png".format(id)), pilmode='RGB')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_AREA)
        vout.write(image)
        os.remove(os.path.join(temp_output_path, 'temp_{:d}.png'.format(id)))
    vout.release()

def render_multiple_anim(smpl_params, titles, output_path="", name="", fps=60):
    if not os.path.exists(output_path): os.makedirs(output_path)
    temp_output_path = os.path.join(output_path, "temp")
    if not os.path.exists(temp_output_path): os.makedirs(temp_output_path)

    if os.path.exists(os.path.join(output_path, "{:s}.mp4".format(name))):
        print(' --- {:s} already rendered'.format(os.path.join(output_path, "{:s}.mp4".format(name))))
        return

    keypoints = []
    for smpl in smpl_params:
        kpts, _ = convert_smpl(SMPLModel, smpl)
        xy_ctr = kpts[:, :1, :2]
        kpts[:, :, :2] -= xy_ctr    # normalize the joints w.r.t xy-coordinates
        keypoints.append(kpts)

    x_ctr, y_ctr, z_ctr, half_range = calc_range(keypoints)

    indices_list = []
    num_proc = 32
    step_size = keypoints[0].shape[0] // num_proc \
        if keypoints[0].shape[0] % num_proc == 0 \
            else (keypoints[0].shape[0] // num_proc) + 1

    for i in range(num_proc):
        indices_list.append(np.arange(i * step_size, (i + 1) * step_size).tolist())
    
    for k in range(len(smpl_params)):
        plist = []
        for i in range(num_proc):
            p = Process(target=plot_smpl_poses, args=(keypoints[k][:, :24], indices_list[i], 
                                                      [x_ctr, y_ctr, z_ctr], half_range, 
                                                      temp_output_path, str(k), titles[k]))
            p.start()
            plist.append(p)

        for p in plist:
            p.join()

    """ convert .pngs to .mp4 """
    num_row = 1
    num_col = len(smpl_params)
    img_height, img_width = num_row * 512, num_col * 512
    size = (img_width, img_height)
    vout = cv2.VideoWriter(os.path.join(output_path, "{:s}.mp4".format(name)), cv2.VideoWriter_fourcc(*"mp4v"), int(fps), size)
    for id in range(keypoints[0].shape[0]):
        image = []  # final image
        for k in range(len(smpl_params)):
            img = cv2.imread(os.path.join(temp_output_path, "{:d}_{:d}.png".format(k, id)))
            h, w, _ = img.shape
            image.append(cv2.resize(img[100:400, 100:400], (w, h), interpolation=cv2.INTER_AREA))
            os.remove(os.path.join(temp_output_path, "{:d}_{:d}.png".format(k, id)))
        image = np.concatenate(image, axis=1)
        image = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_AREA)
        vout.write(image)
    vout.release()

def render_full_body_multiplt_anim(smpl_params, left_hand_joints, right_hand_joints, titles, output_path="", name="", fps=60):
    if not os.path.exists(output_path): os.makedirs(output_path)
    temp_output_path = os.path.join(output_path, "temp")
    if not os.path.exists(temp_output_path): os.makedirs(temp_output_path)

    if os.path.exists(os.path.join(output_path, "{:s}.mp4".format(name))):
        print(' --- {:s} already rendered'.format(os.path.join(output_path, "{:s}.mp4".format(name))))
        return

    keypoints = []
    for smpl in smpl_params:
        kpts, _ = convert_smpl(SMPLModel, smpl)
        xy_ctr = kpts[:, :1, :2]
        kpts[:, :, :2] -= xy_ctr    # normalize the joints w.r.t xy-coordinates
        keypoints.append(kpts)

    x_ctr, y_ctr, z_ctr, half_range = calc_range(keypoints)

    indices_list = []
    num_proc = 8
    step_size = keypoints[0].shape[0] // num_proc \
        if keypoints[0].shape[0] % num_proc == 0 \
            else (keypoints[0].shape[0] // num_proc) + 1

    for i in range(num_proc):
        indices_list.append(np.arange(i * step_size, (i + 1) * step_size).tolist())
    
    for k in range(len(smpl_params)):
        plist = []
        for i in range(num_proc):
            p = Process(target=plot_smpl_and_bvh_poses, 
                        args=(keypoints[k][:, :22], left_hand_joints[k], right_hand_joints[k], 
                              indices_list[i], [x_ctr, y_ctr, z_ctr], half_range, 
                              temp_output_path, str(k), titles[k]))
            p.start()
            plist.append(p)

        for p in plist:
            p.join()
    
    """ convert .pngs to .mp4 """
    num_row = 1
    num_col = len(smpl_params)
    img_height, img_width = num_row * 512, num_col * 512
    size = (img_width, img_height)
    vout = cv2.VideoWriter(os.path.join(output_path, "{:s}.mp4".format(name)), cv2.VideoWriter_fourcc(*"mp4v"), int(fps), size)
    for id in range(keypoints[0].shape[0]):
        image = []  # final image
        for k in range(len(smpl_params)):
            img = cv2.imread(os.path.join(temp_output_path, "{:d}_{:d}.png".format(k, id)))
            h, w, _ = img.shape
            image.append(cv2.resize(img[100:400, 100:400], (w, h), interpolation=cv2.INTER_AREA))
            os.remove(os.path.join(temp_output_path, "{:d}_{:d}.png".format(k, id)))
        image = np.concatenate(image, axis=1)
        image = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_AREA)
        vout.write(image)
    vout.release()

def plot(joints, titles, output_path, output_name, fmt="gif", fps=20):
    if not os.path.exists(output_path): os.makedirs(output_path)
    temp_output_path = os.path.join(output_path, "temp")
    if not os.path.exists(temp_output_path): os.makedirs(temp_output_path, exist_ok=True)

    x_ctr, y_ctr, z_ctr, half_range = calc_range([joints])
    indices = np.arange(0, joints.shape[0]).tolist()
    plot_smpl_poses(joints[:, :24], indices, [x_ctr, y_ctr, z_ctr], half_range, temp_output_path, "temp", titles)

    """ convert pngs to gif """
    if fmt == "gif":
        with imageio.get_writer(os.path.join(output_path, '{:s}.gif'.format(output_name)), mode='I') as writer:
            for id in range(joints.shape[0]):
                image = imageio.imread(os.path.join(temp_output_path, 'temp_{:d}.png'.format(id)))
                writer.append_data(image)
                os.remove(os.path.join(temp_output_path, 'temp_{:d}.png'.format(id)))
    elif fmt == "mp4":
        img_width, img_height = 1024, 1024
        size = (img_width, img_height)
        vout = cv2.VideoWriter(os.path.join(output_path, "{:s}.mp4".format(output_name)), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for id in range(joints.shape[0]):
            image = imageio.imread(os.path.join(temp_output_path, "temp_{:d}.png".format(id)), pilmode='RGB')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_AREA)
            vout.write(image)
            os.remove(os.path.join(temp_output_path, 'temp_{:d}.png'.format(id)))
        vout.release()

def plot_hands(joints, titles, output_path, output_name, fmt="gif", fps=20):
    if not os.path.exists(output_path): os.makedirs(output_path)
    temp_output_path = os.path.join(output_path, "temp")
    if not os.path.exists(temp_output_path): os.makedirs(temp_output_path, exist_ok=True)

    if len(joints.shape) == 2:
        joints = np.reshape(joints, (-1, 24, 3))
        
    x_ctr, y_ctr, z_ctr, half_range = calc_range([joints])
    indices = np.arange(0, joints.shape[0]).tolist()
    plot_bvh_poses(joints, indices, [x_ctr, y_ctr, z_ctr], half_range, temp_output_path, "temp", titles)

    """ convert pngs to gif """
    if fmt == "gif":
        with imageio.get_writer(os.path.join(output_path, '{:s}.gif'.format(output_name)), mode='I') as writer:
            for id in range(joints.shape[0]):
                image = imageio.imread(os.path.join(temp_output_path, 'temp_{:d}.png'.format(id)))
                writer.append_data(image)
                os.remove(os.path.join(temp_output_path, 'temp_{:d}.png'.format(id)))
    elif fmt == "mp4":
        img_width, img_height = 1024, 1024
        size = (img_width, img_height)
        vout = cv2.VideoWriter(os.path.join(output_path, "{:s}.mp4".format(output_name)), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for id in range(joints.shape[0]):
            image = imageio.imread(os.path.join(temp_output_path, "temp_{:d}.png".format(id)), pilmode='RGB')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_AREA)
            vout.write(image)
            os.remove(os.path.join(temp_output_path, 'temp_{:d}.png'.format(id)))
        vout.release()

def plot_full_bodys(joints, titles, output_path, output_name, fmt="gif", fps=20):
    if not os.path.exists(output_path): os.makedirs(output_path)
    temp_output_path = os.path.join(output_path, "temp")
    if not os.path.exists(temp_output_path): os.makedirs(temp_output_path, exist_ok=True)

    if len(joints.shape) == 2:
        joints = np.reshape(joints, (joints.shape[0], -1, 3))
        
    x_ctr, y_ctr, z_ctr, half_range = calc_range([joints])
    indices = np.arange(0, joints.shape[0]).tolist()
    
    x_ctr, y_ctr, z_ctr, half_range = calc_range([joints])
    indices = np.arange(0, joints.shape[0]).tolist()
    plot_full_body_poses(joints, indices, [x_ctr, y_ctr, z_ctr], half_range, temp_output_path, "temp", titles)
    
    """ convert pngs to gif """
    if fmt == "gif":
        with imageio.get_writer(os.path.join(output_path, '{:s}.gif'.format(output_name)), mode='I') as writer:
            for id in range(joints.shape[0]):
                image = imageio.imread(os.path.join(temp_output_path, 'temp_{:d}.png'.format(id)))
                writer.append_data(image)
                os.remove(os.path.join(temp_output_path, 'temp_{:d}.png'.format(id)))
    elif fmt == "mp4":
        img_width, img_height = 1024, 1024
        size = (img_width, img_height)
        vout = cv2.VideoWriter(os.path.join(output_path, "{:s}.mp4".format(output_name)), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for id in range(joints.shape[0]):
            image = imageio.imread(os.path.join(temp_output_path, "temp_{:d}.png".format(id)), pilmode='RGB')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_AREA)
            vout.write(image)
            os.remove(os.path.join(temp_output_path, 'temp_{:d}.png'.format(id)))
        vout.release()

def add_audio(input_video_file, input_audio_file, output_video_file):

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
    final_clip.write_videofile(
        output_video_file, codec='mpeg4', audio_codec='pcm_s16le', bitrate='50000k')

def plot_tokens(data, output_name):

    if "gt_tokens" not in data.keys() or "pred_tokens" not in data.keys():
        return

    gt = data["gt_tokens"]
    pred = data["pred_tokens"]
    length = min(gt.shape[1], pred.shape[1])
    length = min(5000, length)
    
    x = np.arange(0, length)
    y1 = gt[0, :length]
    y2 = pred[0, :length]

    fig, axs = plt.subplots(2, figsize=(100, 10))
    axs[0].plot(x, y1, linestyle='solid', linewidth=1.0, color="r")
    axs[0].set_title("gt tokens")
    axs[1].plot(x, y2, linestyle='solid', linewidth=1.0, color="b")
    axs[1].set_title("pred tokens")
    plt.show()
    plt.savefig(output_name)
    plt.close()

def plot_tokens_full_body(data, output_name):
    
    try:
        print('---', data.keys())
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
    
        for i in range(0, length-500, 500):
            
            x = np.arange(0, 500)

            fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(50, 15))
            f1, = axs[0, 0].plot(x, gt_body[0, i:i+500], linestyle='solid', linewidth=0.5, color="r")
            # f2, = axs[0, 0].plot(x, pred_body[0, i:i+500], linestyle='solid', linewidth=0.5, color="b")
            axs[0, 0].set_title("body tokens", fontsize=40)
            # axs[0, 0].legend([f1, f2], ["gt", "pred"], fontsize=40, loc='upper right')

            f1, = axs[1, 0].plot(x, gt_left[0, i:i+500], linestyle='solid', linewidth=0.5, color="r")
            # f2, = axs[1, 0].plot(x, pred_left[0, i:i+500], linestyle='solid', linewidth=0.5, color="b")
            axs[1, 0].set_title("left hand tokens", fontsize=40)
            # axs[1, 0].legend([f1, f2], ["gt", "pred"], fontsize=40, loc='upper right')

            f1, = axs[2, 0].plot(x, gt_right[0, i:i+500], linestyle='solid', linewidth=0.5, color="r")
            # f2, = axs[2, 0].plot(x, pred_right[0, i:i+500], linestyle='solid', linewidth=0.5, color="b")
            axs[2, 0].set_title("right hand tokens", fontsize=40)
            # axs[2, 0].legend([f1, f2], ["gt", "pred"], fontsize=40, loc='upper right')


            axs[0, 1].plot(x, word_tokens[i:i+500], linestyle='solid', linewidth=0.5, color='g')
            axs[0, 1].set_title("word tokens", fontsize=40)
            axs[1, 1].plot(x, onset_env[i:i+500], linestyle='solid', linewidth=0.5, color='black')
            axs[1, 1].set_title("audio onset env", fontsize=40)
            axs[2, 1].plot(x, onset_beat[i:i+500], linestyle='solid', linewidth=0.5, color='black')
            axs[2, 1].set_title("audio onset beat", fontsize=40)

            plt.tight_layout()
            plt.show()
            plt.savefig(output_name+"_{:d}_{:d}.png".format(i, i+500))
            plt.close()
            
    except:
        print("Unable to draw")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=3, help='')
    parser.add_argument('--vis_num_samples', type=int, default=3, help='')

    parser.add_argument('--base_folder', type=str, default='s2m_v6', help='')
    parser.add_argument('--sub_base_folder', type=str, default='output', help='')
    parser.add_argument('--input_folder', type=str, default='s2m', help='path of training folder')
    parser.add_argument('--output_folder', type=str, default='s2m', help='path of training folder')
    parser.add_argument('--cmp_folder', type=str, default='cmp_s2m', help='path of training folder')
    parser.add_argument('--fps', type=int, default=31, help='path of training folder')
    parser.add_argument('--animate', type=str2bool, default=True, help='animate or not')
    
    args = parser.parse_args()
    return args

def main():
    """animate smpl"""
    args = parse_args()

    num_samples = args.num_samples
    base_folder = args.base_folder
    sub_base_folder = args.sub_base_folder
    input_folder = args.input_folder
    output_folder = args.output_folder
    cmp_folder = args.cmp_folder

    FPS = args.fps

    input_path = "logs/speech2motion/eval/{:s}/{:s}/{:s}".format(base_folder, sub_base_folder, input_folder)
    audio_path = "../dataset/BEAT/raw"
    output_path = "logs/speech2motion/eval/{:s}/animation/{:s}".format(base_folder, output_folder)

    files = os.listdir(input_path)

    for idx, file in enumerate(files):
        folder_id = file.split("_")[0]

        data = np.load(os.path.join(input_path, file), allow_pickle=True).item()

        name = data["caption"][0]
        audio = torch.from_numpy(data["audio"])
        gt_motion = torch.from_numpy(data["gt"])
        rc_motion = torch.from_numpy(data["motion"])

        audio = audio.permute(0, 2, 1).detach().cpu().numpy()
        gt_motion = gt_motion.permute(0, 2, 1).detach().cpu().numpy()
        rc_motion = rc_motion.permute(0, 2, 1).detach().cpu().numpy()

        max_length = min(min(gt_motion.shape[1], rc_motion.shape[1]), audio.shape[1])
        gt_motion = gt_motion[:, :max_length]
        rc_motion = rc_motion[:, :max_length]

        print('[{:d}/{:d}]'.format(idx, len(files)), gt_motion.shape, rc_motion.shape, audio.shape)

        # Plot predicted tokens
        if not os.path.exists(os.path.join(output_path, name+".png")):
            plot_tokens(data, os.path.join(output_path, name+".png"))

        # Animate motion
        render_multiple_anim([gt_motion[0], rc_motion[0]], ["gt", "pred"], output_path, name=name, fps=FPS)

        # Add audio to animation
        input_video_file = os.path.join(output_path, name+".mp4")
        input_audio_file = os.path.join(audio_path, folder_id, name+".wav")
        output_video_file = os.path.join(output_path, name+"_audio.avi")
        if not os.path.exists(output_video_file):
            try:
                add_audio(input_video_file, input_audio_file, output_video_file)
            except:
                print("audio is not added to the video!")
       
def main_full():
    """animate smpl + bvh hand """
    """animate smpl"""
    args = parse_args()

    num_samples = args.num_samples
    base_folder = args.base_folder
    sub_base_folder = args.sub_base_folder
    input_folder = args.input_folder
    output_folder = args.output_folder
    cmp_folder = args.cmp_folder

    FPS = args.fps

    input_path = "logs/speech2motion/eval/{:s}/{:s}/{:s}".format(base_folder, sub_base_folder, input_folder)
    audio_path = "../dataset/BEAT/raw"
    output_path = "logs/speech2motion/eval/{:s}/animation/{:s}".format(base_folder, output_folder)

    files = os.listdir(input_path)

    for idx, file in enumerate(files):
        folder_id = file.split("_")[0]

        data = np.load(os.path.join(input_path, file), allow_pickle=True).item()
        name = data["caption"][0]
        audio = torch.from_numpy(data["audio"])

        gt_body = torch.from_numpy(data["gt"]["body"])
        gt_left = torch.from_numpy(data["gt"]["left"])
        gt_right = torch.from_numpy(data["gt"]["right"])
        rc_body = torch.from_numpy(data["pred"]["body"])
        rc_left = torch.from_numpy(data["pred"]["left"])
        rc_right = torch.from_numpy(data["pred"]["right"])

        audio = audio.permute(0, 2, 1).detach().cpu().numpy()
        gt_body = gt_body.permute(0, 2, 1).detach().cpu().numpy()
        gt_left = gt_left.permute(0, 2, 1).reshape(1, -1, 24, 3).detach().cpu().numpy()
        gt_right = gt_right.permute(0, 2, 1).reshape(1, -1, 24, 3).detach().cpu().numpy()
        rc_body = rc_body.permute(0, 2, 1).detach().cpu().numpy()
        rc_left = rc_left.permute(0, 2, 1).reshape(1, -1, 24, 3).detach().cpu().numpy()
        rc_right = rc_right.permute(0, 2, 1).reshape(1, -1, 24, 3).detach().cpu().numpy()

        max_length = min(min(gt_body.shape[1], rc_body.shape[1]), audio.shape[1])
        gt_body = gt_body[:, :max_length]
        gt_left = gt_left[:, :max_length]
        gt_right = gt_right[:, :max_length]
        rc_body = rc_body[:, :max_length]
        rc_left = rc_left[:, :max_length]
        rc_right = rc_right[:, :max_length]
        audio = audio[:, :max_length]

        print('[{:d}/{:d}]'.format(idx, len(files)), gt_body.shape, rc_body.shape, gt_left.shape, rc_left.shape, gt_right.shape, rc_right.shape)

        # Plot predicted tokens
        # if not os.path.exists(os.path.join(output_path, name+".png")):
        plot_tokens_full_body(data, os.path.join(output_path, name+".png"))

        # Animate motion
        if args.animate:
            r = 1
            render_full_body_multiplt_anim([gt_body[0][::r], rc_body[0][::r]], 
                                           [gt_left[0][::r], rc_left[0][::r]], 
                                           [gt_right[0][::r], rc_right[0][::r]], 
                                           ["gt", "pred"], output_path, name=name, fps=FPS)
    
            # Add audio to animation
            input_video_file = os.path.join(output_path, name+".mp4")
            input_audio_file = os.path.join(audio_path, folder_id, name+".wav")
            output_video_file = os.path.join(output_path, name+"_audio.avi")
            if not os.path.exists(output_video_file):
                try:
                    add_audio(input_video_file, input_audio_file, output_video_file)
                except:
                    print("audio is not added to the video!")

        
if __name__ == "__main__":

    # main()
    main_full()


    # """ combine (debug) """
    # cmmn_files = ["logs/speech2motion/eval/s2m_v6/animation/s2m/2_scott_0_9_16.mp4", 
    #               "logs/speech2motion/eval/s2m_v9/animation/s2m/2_scott_0_9_16.mp4"]
    # all_frames = {}
    # frame_infos = []
    # for file in cmmn_files:
    #     cap = cv2.VideoCapture(file)
    #     gt_frames = []
    #     frames = []
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if ret == True:
    #             # print(' --->>> ', os.path.join(path, file))
    #             width = frame.shape[1]
    #             # print(' ---> width', width)
    #             gt_frm = frame[:, :width//2]
    #             rc_frm = frame[:, width//2:]
    #             gt_frames.append(gt_frm)
    #             frames.append(rc_frm)
    #             img_height, img_width = rc_frm.shape[:2]
    #             img_width /= 2
    #         else:
    #             break
        
    #     all_frames[file] = frames
    #     frame_infos.append(len(frames))

    # num_frames = np.min(frame_infos)

    # output_video_path = "logs/speech2motion/eval/s2m_v9/animation/2_scott_0_9_16.mp4"
    # fps = 31
    # size = (int(img_width * len(all_frames)), int(img_height))
    # print(size, "|", num_frames)
    # vout = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), size)
    # for id in range(num_frames):
    #     imgs = []
    #     for k, frames in all_frames.items():
    #         img = frames[id]
    #         imgs.append(img)
    #     imgs = np.concatenate(imgs, axis=1)
    #     # print(' ---> ', imgs.shape)
    #     vout.write(imgs)
    # vout.release()

    # """ add audio (debug) """
    # add_audio("logs/speech2motion/eval/s2m_v9/2_scott_0_2_8.mp4", 
    #           "../dataset/BEAT/raw/2/2_scott_0_2_8.wav", 
    #           "logs/speech2motion/eval/s2m_v9/2_scott_0_2_8.avi")

    

   