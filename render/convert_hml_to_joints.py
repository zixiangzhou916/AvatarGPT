import torch
import numpy as np
import argparse, os, sys
sys.path.append(os.getcwd())

from funcs.hml3d.rotation2xyz import Rotation2xyz
from funcs.hml3d.simplify_loc2rot import Joints2SMPL as joints2smpl
from funcs.hml3d.conversion import recover_from_ric
from scipy.spatial.transform import Rotation as R

os.environ['PYOPENGL_PLATFORM'] = "osmesa"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def convert(motions, outdir="test_vis", device_id=0, name=None, pred=True):
    frames, njoints, nfeats = motions.shape
    MINS = motions.min(axis=0).min(axis=0)
    MAXS = motions.max(axis=0).max(axis=0)

    height_offset = MINS[1]
    motions[:, :, 1] -= height_offset
    trajec = motions[:, 0, [0, 2]]

    j2s = joints2smpl(num_frames=frames, device_id=0, cuda=True if torch.cuda.is_available() else False)
    rot2xyz = Rotation2xyz(device=device)
    faces = rot2xyz.smpl_model.faces
    
    if not os.path.exists(outdir + name+'.npy'): 
        print(f'Running SMPLify, it may take a few minutes.')
        obj = R.from_euler('x', 90, degrees=True)
        Rot = obj.as_matrix()
        for idx, pose in enumerate(motions):
            motions[idx] = np.matmul(Rot, pose.transpose()).transpose()

        motion_tensor, opt_dict = j2s.joint2smpl(motions)  # [nframes, njoints, 3]

        vertices = rot2xyz(torch.tensor(motion_tensor).clone(), mask=None,
                                        pose_rep='rot6d', translation=True, glob=True,
                                        jointstype='vertices',
                                        vertstrans=True)
    else:
        vertices = np.load(outdir + name + ".npy", allow_pickle=True).item()["pred"]["body"]
        vertices = torch.from_numpy(vertices).float().to(device).unsqueeze(dim=0)
    
    return vertices, faces
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filedir", type=str, default="logs/avatar_gpt/eval/flan_t5_large/exp2/output/t2m/", help='motion npy file dir')
    parser.add_argument("--joints_dir", type=str, default="logs/avatar_gpt/eval/flan_t5_large/exp2/meshes/t2m/", help='motion npy file dir')
    parser.add_argument("--video_dir", type=str, default="logs/avatar_gpt/eval/flan_t5_large/exp2/animation/t2m/", help='motion npy file dir')
    parser.add_argument('--motion-list', default=None, nargs="+", type=str, help="motion name list")
    parser.add_argument('--visualize', type=str2bool, default=False, help="motion name list")
    parser.add_argument('--convert_gt', type=str2bool, default=False, help="motion name list")
    args = parser.parse_args()
    
    # filename_list = args.motion_list
    filedir = args.filedir
    joints_dir = args.joints_dir
    video_dir = args.video_dir
    if not os.path.exists(joints_dir):
        os.makedirs(joints_dir)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    filename_list = [f.split(".")[0] for f in os.listdir(filedir) if ".npy" in f]
    
    for cnt, filename in enumerate(filename_list):
        if os.path.exists(joints_dir + filename + ".npy"):
            continue
        
        # if "B0002_T0000" not in filename: continue
        data = np.load(filedir + filename + ".npy", allow_pickle=True).item()
        if args.convert_gt:
            motions = data['gt']['body'].transpose(0, 2, 1)
        else:
            motions = data['pred']['body'].transpose(0, 2, 1)
        captions = data["caption"][0]
        print("[{:d}/{:d}] Caption: {:s}".format(cnt+1, len(filename_list), captions))
        captions = captions.replace(".", "").replace("/", "_")
        words = captions.split(" ")
        name = "_".join(words[:20])
        
        pose_xyz = recover_from_ric(torch.from_numpy(motions).float().to(device), 22)
        pose_xyz = pose_xyz[0]  # [T, N, 3]

        # obj = R.from_euler('x', 90, degrees=True)
        # Rot = obj.as_matrix()
        # for idx, pose in enumerate(pose_xyz):
        #     pose_xyz[idx] = np.matmul(Rot, pose.transpose()).transpose()
        
        output = {
            "pred": {"body": pose_xyz.data.cpu().numpy()}, 
            "caption": data["caption"], 
            "color_labels": data.get("color_labels", None)
        }
        np.save(joints_dir + filename + ".npy", output)
        