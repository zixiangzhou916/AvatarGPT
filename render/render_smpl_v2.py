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
from networks.smplx_code import SMPLx
from dataset.speech2motion.BEAT_normalize_v2 import forward_kinematic, visualize_

import cv2
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from PIL import Image, ImageTk, ImageSequence
from multiprocessing import Process
import imageio

try:
    from moviepy.editor import *
    from moviepy.video.fx.all import crop
    from pydub import AudioSegment
except:
    print("Unable to import moviepy and pydub")
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# SMPLModel = SMPLx(model_path="./networks/smpl", gender="NEUTRAL", batch_size=1)

# SAMPLE_DATA = np.load("../dataset/BEAT/aligned_v2/9_miranda_1_9_10.npy", allow_pickle=True).item()
SAMPLE_DATA = np.load("dataset/speech2motion/hand_offsets.npy", allow_pickle=True).item()
left_offsets = np.reshape(SAMPLE_DATA["left_offsets"][:1], (1, 24, 3))
right_offsets = np.reshape(SAMPLE_DATA["right_offsets"][:1], (1, 24, 3))

SMPLx_BODY_POSE_EDGE = [(15, 12), (12, 9), (9, 13), (13, 16), (16, 18), (18, 20), 
    (20, 22), (9, 14), (14, 17), (17, 19), (19, 21), 
    (21, 23), (9, 6), (6, 3), (3, 0), (0, 2), (0, 1), 
    (2, 5), (5, 8), (8, 11), (1, 4), (4, 7), (7, 10)
]

SMPLx_HAND_POSE_EDGE = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), 
    (6, 7), (7, 8), (8, 9), (5, 10), (10, 11), (11, 12), 
    (12, 13), (13, 14), (0, 15), (15, 16), (16, 17), (17, 18), 
    (18, 19), (15, 20), (20, 21), (21, 22), (22, 23), 
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
    
def plot_tokens_full_body(data, output_name):
    try:
        print("---", data.keys())
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
            print(output_name)
            plt.show()
            plt.savefig(output_name+"_{:d}_{:d}.png".format(i, i+500))
            plt.close()
            
    except:
        # print("Unable to draw")
        raise ValueError
        
def plot_smplx_and_bvh(
    smplx_joint, 
    bvh_joint, 
    indices, 
    center, 
    half_range, 
    output_path, 
    output_name, 
    title
):
    """
    :param smplx_joint: [njoints, 3]
    :param bvh_joint: {'left': [njoints, 3], 'right': [njoints, 3]}
    """
    if title is not None:
        title_sp = title.split(' ')
        if len(title_sp) > 20:
            title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
        elif len(title_sp) > 10:
            title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
        
    x_center, y_center, z_center = center
    left_bvh_hand = bvh_joint["left"]
    right_bvh_hand = bvh_joint["right"]
    
    for index in indices:
        if index >= smplx_joint.shape[0] or index >= left_bvh_hand.shape[0] or index >= right_bvh_hand.shape[0]:
            continue
    
        body_pose = smplx_joint[index, ...]
        left_pose = left_bvh_hand[index, ...]
        right_pose = right_bvh_hand[index, ...]
        
        pxs = np.arange(x_center - 1.5, x_center + 1.51, 3.0)       # global plane
        pys = np.arange(y_center - 1.5, y_center + 1.51, 3.0)
        
        pxs, pys = np.meshgrid(pxs, pys)
        pzs = np.zeros_like(pxs)
        
        fig = plt.figure(figsize=(5, 5), dpi=200)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(x_center - 1.0, x_center + 1.0)
        ax.set_ylim(y_center - 1.0, y_center + 1.0)
        ax.set_zlim(-0.2, 1.8)
        ax.plot_surface(pxs, pys, pzs, rstride=1, cstride=1, facecolors="green", alpha=0.5)
        pts1 = ax.scatter(body_pose[:, 0], body_pose[:, 1], body_pose[:, 2], marker='.', c='black', s=10, alpha=1)
        pts2 = ax.scatter(left_pose[:, 0], left_pose[:, 1], left_pose[:, 2], marker='.', c='blue', s=10, alpha=1)
        pts3 = ax.scatter(right_pose[:, 0], right_pose[:, 1], right_pose[:, 2], marker='.', c='coral', s=10, alpha=1)
        
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.zlabel('z')
        for (i, j) in SMPLx_BODY_POSE_EDGE:
            if i >= body_pose.shape[0] or j >= body_pose.shape[0]: continue
            ax.plot([body_pose[i, 0], body_pose[j, 0]], 
                    [body_pose[i, 1], body_pose[j, 1]], 
                    [body_pose[i, 2], body_pose[j, 2]], 'k-', lw=1)
        for (i, j) in SMPLx_HAND_POSE_EDGE:
            ax.plot([left_pose[i, 0], left_pose[j, 0]], 
                    [left_pose[i, 1], left_pose[j, 1]], 
                    [left_pose[i, 2], left_pose[j, 2]], color="blue", linestyle="-", lw=1)
        for (i, j) in SMPLx_HAND_POSE_EDGE:
            ax.plot([right_pose[i, 0], right_pose[j, 0]], 
                    [right_pose[i, 1], right_pose[j, 1]], 
                    [right_pose[i, 2], right_pose[j, 2]], color="coral", linestyle="-", lw=1)
        
        plt.legend([pts1, pts2, pts3], ["body", "left_hand", "right_hand"])
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(os.path.join(output_path, '{:s}_{:d}.png'.format(output_name, index)))
        plt.close()

def plot_smplx_and_bvh_2d(
    smplx_joint, 
    bvh_joint, 
    indices, 
    center, 
    half_range, 
    output_path, 
    output_name, 
    title
):
    """
    :param smplx_joint: [njoints, 3]
    :param bvh_joint: {'left': [njoints, 3], 'right': [njoints, 3]}
    """
    
    if title is not None:
        title_sp = title.split(' ')
        if len(title_sp) > 20:
            title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
        elif len(title_sp) > 10:
            title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
        
    x_center, y_center, z_center = center
    left_bvh_hand = bvh_joint["left"]
    right_bvh_hand = bvh_joint["right"]
    
    for index in indices:
        if index >= smplx_joint.shape[0] or index >= left_bvh_hand.shape[0] or index >= right_bvh_hand.shape[0]:
            continue
    
        body_pose = smplx_joint[index, ...]
        left_pose = left_bvh_hand[index, ...]
        right_pose = right_bvh_hand[index, ...]

        fig = plt.figure(figsize=(10, 5), dpi=500)
        ax = fig.add_subplot(121)
        ax.set_xlim(x_center - half_range - 0.1, x_center + half_range + 0.1)
        ax.set_ylim(z_center - half_range - 0.1, z_center + half_range + 0.1)
        
        pts1 = ax.scatter(body_pose[:, 0], body_pose[:, 2], marker='.', c='black', s=1, alpha=1)
        pts2 = ax.scatter(left_pose[:, 0], left_pose[:, 2], marker='.', c='blue', s=1, alpha=1)
        pts3 = ax.scatter(right_pose[:, 0], right_pose[:, 2], marker='.', c='coral', s=1, alpha=1)

        for (i, j) in SMPLx_BODY_POSE_EDGE:
            if i >= body_pose.shape[0] or j >= body_pose.shape[0]: continue
            ax.plot([body_pose[i, 0], body_pose[j, 0]],
                    [body_pose[i, 2], body_pose[j, 2]], 'k-', lw=0.1)
        for (i, j) in SMPLx_HAND_POSE_EDGE:
            ax.plot([left_pose[i, 0], left_pose[j, 0]], 
                    [left_pose[i, 2], left_pose[j, 2]], color="blue", linestyle="-", lw=0.1)
        for (i, j) in SMPLx_HAND_POSE_EDGE:
            ax.plot([right_pose[i, 0], right_pose[j, 0]],  
                    [right_pose[i, 2], right_pose[j, 2]], color="coral", linestyle="-", lw=0.1)

        ax.legend([pts1, pts2, pts3], 
                   ["body", "left_hand", "right_hand"])
        
        ax = fig.add_subplot(122)
        ax.set_xlim(y_center - half_range - 0.1, y_center + half_range + 0.1)
        ax.set_ylim(z_center - half_range - 0.1, z_center + half_range + 0.1)

        pts1 = ax.scatter(body_pose[:, 1], body_pose[:, 2], marker='.', c='black', s=1, alpha=1)
        pts2 = ax.scatter(left_pose[:, 1], left_pose[:, 2], marker='.', c='blue', s=1, alpha=1)
        pts3 = ax.scatter(right_pose[:, 1], right_pose[:, 2], marker='.', c='coral', s=1, alpha=1)
        
        for (i, j) in SMPLx_BODY_POSE_EDGE:
            if i >= body_pose.shape[0] or j >= body_pose.shape[0]: continue
            ax.plot([body_pose[i, 1], body_pose[j, 1]],
                    [body_pose[i, 2], body_pose[j, 2]], 'k-', lw=0.1)
        for (i, j) in SMPLx_HAND_POSE_EDGE:
            ax.plot([left_pose[i, 1], left_pose[j, 1]], 
                    [left_pose[i, 2], left_pose[j, 2]], color="blue", linestyle="-", lw=0.1)
        for (i, j) in SMPLx_HAND_POSE_EDGE:
            ax.plot([right_pose[i, 1], right_pose[j, 1]],  
                    [right_pose[i, 2], right_pose[j, 2]], color="coral", linestyle="-", lw=0.1)

        ax.legend([pts1, pts2, pts3], 
                   ["body", "left_hand", "right_hand"])
        
        ax.legend([pts1, pts2, pts3], 
               ["body", "left_hand", "right_hand"])
        plt.savefig(os.path.join(output_path, '{:s}_{:d}.png'.format(output_name, index)))
        plt.close()

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

def animate(body_poses, left_poses, right_poses, 
            titles, output_path, name, fps=30, 
            plot_type="2d", video_fmt="mp4"):
    assert plot_type in ["2d", "3d"]
    assert video_fmt in ["mp4", "gif"]

    # print('--->', body_poses[0].shape)    
    if not os.path.exists(output_path): os.makedirs(output_path)
    temp_output_path = os.path.join(output_path, "temp")
    if not os.path.exists(temp_output_path): os.makedirs(temp_output_path)

    if os.path.exists(os.path.join(output_path, "{:s}.mp4".format(name))):
        print(' --- {:s} already rendered'.format(os.path.join(output_path, "{:s}.mp4".format(name))))
        return
    
    output = []
    for k in range(len(body_poses)):
        T = body_poses[k].shape[0]
        out = visualize_(body_poses[k].copy(), 
                         left_poses[k].copy(), 
                         left_offsets.repeat(T, axis=0), 
                         right_poses[k].copy(), 
                         right_offsets.repeat(T, axis=0), 
                         prefix=k, 
                         visualize_path=output_path)
        output.append(out)
    
    # def check_hand(output):
    #     left_arm = output["body"][0, 20] - output["body"][0, 18]
    #     right_arm = output["body"][0, 21] - output["body"][0, 19]
    #     left_hand = output["left"][0, 4] - output["left"][0, 0]
    #     right_hand = output["right"][0, 4] - output["right"][0, 0]

    #     flag = False
    #     if np.dot(left_arm, left_hand) < 0:
    #         left_poses[:, 0, 1] = 180
    #         flag = True

    #     if np.dot(right_arm, right_hand) < 0:
    #         right_poses[:, 0, 1] = 180
    #         flag = True
        
    #     return flag, left_poses, right_poses

    # output = []
    # with torch.no_grad():
    #     for k in range(len(body_poses)):
    #         T = body_poses[k].shape[0]
    #         out = forward_kinematic(body_poses[k].copy(), 
    #                                 left_poses[k].copy(), 
    #                                 left_offsets.repeat(T, axis=0), 
    #                                 right_poses[k].copy(), 
    #                                 right_offsets.repeat(T, axis=0))
    #         flag, _left_pose, _right_pose = check_hand(out)
    #         if flag:
    #             out = forward_kinematic(body_poses[k].copy(), 
    #                                     _left_pose.copy(), 
    #                                     left_offsets.repeat(T, axis=0), 
    #                                     _right_pose.copy(), 
    #                                     right_offsets.repeat(T, axis=0))
    #         output.append(out)
        
    # x_ctr, y_ctr, z_ctr, half_range = calc_range(output[0]["body"])
    
    # indices_list = []
    # num_proc = 32
    # step_size = output[0]["body"].shape[0] // num_proc \
    #     if output[0]["body"].shape[0] % num_proc == 0 \
    #         else (output[0]["body"].shape[0] // num_proc) + 1
    
    # for i in range(num_proc):
    #     indices_list.append(np.arange(i * step_size, (i + 1) * step_size).tolist())
        
    # for k in range(len(body_poses)):
    #     plist = []
    #     for i in range(num_proc):
    #         if plot_type == "2d":
    #             p = Process(target=plot_smplx_and_bvh_2d, 
    #                         args=(output[k]["body"][:, :24], 
    #                               output[k], indices_list[i], 
    #                               [x_ctr, y_ctr, z_ctr], half_range, 
    #                               temp_output_path, str(k), titles[k]))
    #         elif plot_type == "3d":
    #             p = Process(target=plot_smplx_and_bvh, 
    #                         args=(output[k]["body"][:, :24], 
    #                               output[k], indices_list[i], 
    #                               [x_ctr, y_ctr, z_ctr], half_range, 
    #                               temp_output_path, str(k), titles[k]))
    #         p.start()
    #         plist.append(p)

    #     for p in plist:
    #         p.join()
    
    if video_fmt == "gif":
        """ convert .pngs to .gif """
        with imageio.get_writer(os.path.join(output_path, "{:s}.gif".format(name)), mode="I") as writer:
            for id in range(output[0]["body"].shape[0]):
                image = []
                for k in range(len(body_poses)):
                    img = cv2.imread(os.path.join(temp_output_path, "{:d}_{:d}.png".format(k, id)))
                    h, w, _ = img.shape
                    if plot_type == "3d":
                        # image.append(cv2.resize(img[100:400, 100:400], (w, h), interpolation=cv2.INTER_AREA))
                        image.append(img)
                    elif plot_type == "2d":
                        image.append(img)
                    os.remove(os.path.join(temp_output_path, "{:d}_{:d}.png".format(k, id)))
                image = np.concatenate(image, axis=1)
                writer.append_data(image)
                
    elif video_fmt == "mp4":
        """ convert .pngs to .mp4 """
        num_row = 1
        num_col = len(body_poses)
        if plot_type == "2d":
            img_height, img_width = num_row * 512, num_col * 1024
        elif plot_type == "3d":
            img_height, img_width = num_row * 512, num_col * 512
        size = (img_width, img_height)
        vout = cv2.VideoWriter(os.path.join(output_path, "{:s}.mp4".format(name)), cv2.VideoWriter_fourcc(*"mp4v"), int(fps), size)
        for id in range(output[0]["body"].shape[0]):
            image = []  # final image
            for k in range(len(body_poses)):
                # print('--->', os.path.join(temp_output_path, "{:d}_{:d}.png".format(k, id)))
                img = cv2.imread(os.path.join(temp_output_path, "{:d}_{:d}.png".format(k, id)))
                h, w, _ = img.shape
                if plot_type == "3d":
                    # image.append(cv2.resize(img[100:400, 100:400], (w, h), interpolation=cv2.INTER_AREA))
                    image.append(img)
                elif plot_type == "2d":
                    image.append(img)
                os.remove(os.path.join(temp_output_path, "{:d}_{:d}.png".format(k, id)))
            image = np.concatenate(image, axis=1)
            image = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_AREA)
            vout.write(image)
        vout.release()
        
def plot_full_body(body_joints, left_joints, right_joints, title, save_dir, save_name, fmt="gif"):
    assert fmt in ["gif", "mp4"]

    temp_output_path = os.path.join(save_dir, "temp")
    if not os.path.exists(temp_output_path): os.makedirs(temp_output_path)
    indices = np.arange(0, body_joints.shape[0]).tolist()
    x_ctr, y_ctr, z_ctr, half_range = calc_range([body_joints])
    plot_smplx_and_bvh_2d(body_joints, 
                          {"left": left_joints, "right": right_joints}, 
                          indices, 
                          [x_ctr, y_ctr, z_ctr], 
                          half_range, 
                          temp_output_path, "temp", title)

    """ convert pngs to gif """
    if fmt == "gif":
        with imageio.get_writer(os.path.join(save_dir, '{:s}.gif'.format(save_name)), mode='I') as writer:
            for id in range(body_joints.shape[0]):
                image = imageio.imread(os.path.join(temp_output_path, 'temp_{:d}.png'.format(id)))
                writer.append_data(image)
                os.remove(os.path.join(temp_output_path, 'temp_{:d}.png'.format(id)))
    elif fmt == "mp4":
        pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=3, help='')
    parser.add_argument('--vis_num_samples', type=int, default=3, help='')

    parser.add_argument('--base_folder', type=str, default='s2m_v9_exp2', help='')
    parser.add_argument('--sub_base_folder', type=str, default='output', help='')
    parser.add_argument('--input_folder', type=str, default='s2m', help='path of training folder')
    parser.add_argument('--output_folder', type=str, default='s2m', help='path of training folder')
    parser.add_argument('--cmp_folder', type=str, default='cmp_s2m', help='path of training folder')
    parser.add_argument('--fps', type=int, default=31, help='path of training folder')
    parser.add_argument('--animate', type=str2bool, default=True, help='animate or not')
    parser.add_argument('--draw_tokens', type=str2bool, default=False, help='animate or not')
    parser.add_argument('--plot_type', type=str, default="2d", help='animate or not')
    parser.add_argument('--video_fmt', type=str, default="mp4", help='animate or not')
    parser.add_argument('--sample_rate', type=int, default=1, help='sampling rate')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
        
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
    image_path = "logs/speech2motion/eval/{:s}/image/{:s}".format(base_folder, output_folder)

    files = [f for f in os.listdir(input_path) if ".npy" in f]
    print(files)
    
    for idx, file in enumerate(files):
        folder_id = file.split("_")[0]
    
        data = np.load(os.path.join(input_path, file), allow_pickle=True).item()
        
        name = data["caption"][0]
        if "2_scott_0_2_8" not in name: continue
        speaker_id = name.split("_")[1]
        # if speaker_id not in ["2", "4", "6", "8"]: continue
        print('-' * 10, name, '-' * 10)
        # audio = torch.from_numpy(data["audio"])
        audio = data["audio"][0].transpose()
        # exit(0)

        # gt_body = torch.from_numpy(data["gt"]["body"])
        # gt_left = torch.from_numpy(data["gt"]["left"])
        # gt_right = torch.from_numpy(data["gt"]["right"])
        # rc_body = torch.from_numpy(data["pred"]["body"])
        # rc_left = torch.from_numpy(data["pred"]["left"])
        # rc_right = torch.from_numpy(data["pred"]["right"])
        
        # gt_body = gt_body.permute(0, 2, 1).detach().cpu().numpy()
        # gt_left = gt_left.permute(0, 2, 1).reshape(1, -1, 24, 3).detach().cpu().numpy()
        # gt_right = gt_right.permute(0, 2, 1).reshape(1, -1, 24, 3).detach().cpu().numpy()
        # rc_body = rc_body.permute(0, 2, 1).detach().cpu().numpy()
        # rc_left = rc_left.permute(0, 2, 1).reshape(1, -1, 24, 3).detach().cpu().numpy()
        # rc_right = rc_right.permute(0, 2, 1).reshape(1, -1, 24, 3).detach().cpu().numpy()
        
        gt_body = data["gt"]["body"][0].transpose()
        gt_left = data["gt"]["left"][0].transpose()
        gt_right = data["gt"]["right"][0].transpose()
        rc_body = data["pred"]["body"][0].transpose()
        rc_left = data["pred"]["left"][0].transpose()
        rc_right = data["pred"]["right"][0].transpose()
        
        
        # max_length = min(min(gt_body.shape[1], rc_body.shape[1]), audio.shape[1])
        max_length = 1000
        gt_body = gt_body[:max_length]
        gt_left = gt_left[:max_length]
        gt_right = gt_right[:max_length]
        rc_body = rc_body[:max_length]
        rc_left = rc_left[:max_length]
        rc_right = rc_right[:max_length]
        audio = audio[:max_length]
        
        rc_left[:, 0] *= 0.0
        rc_right[:, 0] *= 0.0

        print('[{:d}/{:d}]'.format(idx, len(files)), gt_body.shape, rc_body.shape, gt_left.shape, rc_left.shape, gt_right.shape, rc_right.shape)

        # Plot predicted tokens
        if args.draw_tokens:
            plot_tokens_full_body(data, os.path.join(image_path, name))
    
        if args.animate:
            # r = args.sample_rate
            # animate(body_poses=[gt_body[::r], rc_body[::r]], 
            #         left_poses=[gt_left.reshape((max_length, 24, 3))[::r], 
            #                     rc_left.reshape((max_length, 24, 3))[::r]], 
            #         right_poses=[gt_right.reshape((max_length, 24, 3))[::r], 
            #                      rc_right.reshape((max_length, 24, 3))[::r]], 
            #         titles=["gt", "pred"], 
            #         output_path=output_path, 
            #         name=name, 
            #         fps=FPS, 
            #         plot_type=args.plot_type, 
            #         video_fmt=args.video_fmt)
            
            # Add audio to animation
            input_video_file = os.path.join(output_path, name+".mp4")
            input_audio_file = os.path.join(audio_path, folder_id, name+".wav")
            output_video_file = os.path.join(output_path, name+"_audio.avi")
            if not os.path.exists(output_video_file):
                try:
                    add_audio(input_video_file, input_audio_file, output_video_file)
                except:
                    print("audio is not added to the video!")
        
        # exit(0)
    
    