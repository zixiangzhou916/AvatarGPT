import os, sys, argparse
sys.path.append(os.getcwd())
import numpy as np
import torch
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as R

from funcs.hml3d.skeleton import Skeleton
from funcs.hml3d.quaternion import *
from funcs.hml3d.param_util import *
from funcs.hml3d.conversion import *
from funcs.hml3d.simplify_loc2rot import Joints2SMPL
from funcs.hml3d.rotation2xyz import Rotation2xyz
from funcs.hml3d import rotation_conversion as geometry

def rotate_joints(joints, rot):
    """
    :param joints: [nframes, njoints, 3]
    """
    T, J = joints.shape[:2]
    joints_reshape = np.reshape(joints, newshape=(T*J, 3))
    
    M = rot.as_matrix()   # [3, 3]
    joints_rot = np.matmul(M, joints_reshape.transpose(1, 0)).transpose(1, 0)
    joints_rot = np.reshape(joints_rot, newshape=(T, J, 3))
    return joints_rot

def joints2smpl(positions):
    num_frames = positions.shape[0]
    
    j2s = Joints2SMPL(num_frames=num_frames, device_id=0, cuda=True)
    motion_tensor, opt_dict = j2s.joint2smpl(positions) # [1, 25, 6, T]
    # motion_tensor = motion_tensor[0].permute(2, 0, 1)   # [T, 25, 6]
    
    # thetas = motion_tensor[:, :-1]
    # root_loc = motion_tensor[:, -1]
    # matrix = geometry.rotation_6d_to_matrix(thetas)
    # rotvec = geometry.matrix_to_axis_angle(matrix=matrix)   # [T, 24, 3]
    # rotvec = torch.reshape(rotvec, (rotvec.size(0), -1))
    # smpl = torch.cat([root_loc[..., :3], rotvec], dim=1)
    
    x_translations = motion_tensor[:, -1, :3]       # [1, 3, T]
    x_rotations = motion_tensor[:, :-1]
    x_rotations = x_rotations.permute(0, 3, 1, 2)
    nsamples, times, njoints, feats = x_rotations.shape
    
    mask = torch.ones((motion_tensor.shape[0], motion_tensor.shape[-1]), dtype=bool, device=motion_tensor.device)
    rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])
    
    global_orient = rotations[:, 0]         # [T, 3, 3]
    rotations = rotations[:, 1:]            # [T, J, 3, 3]
    global_orient = geometry.matrix_to_axis_angle(matrix=global_orient)
    rotvecs = geometry.matrix_to_axis_angle(matrix=rotations)
    print(global_orient.shape, "|", rotvecs.shape, "|", x_translations.shape)
    # exit(0)
    
    smpl_output = torch.cat([x_translations[0].t(), global_orient, rotvecs.view(times, -1)], dim=-1)
    
    return smpl_output

def main(input_dir, output_dir):
    
    l_idx1, l_idx2 = 5, 8
    # Right/Left foot
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    # l_hip, r_hip
    r_hip, l_hip = 2, 1
    joints_num = 22
    
    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)   # [22, 3]
    kinematic_chain = t2m_kinematic_chain
    # Get offsets of target skeleton
    example_file = "tools/000021.npy"
    example_data = np.load(example_file)
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])  # [22, 3]
    
    source_list = [f for f in os.listdir(input_dir) if ".npy" in f]
    
    for source_file in tqdm(source_list):
        output_file = os.path.join(output_dir, source_file)
        if os.path.exists(output_file):
            print("{:s} already converted".format(output_file))
            continue
        source_data = np.load(os.path.join(input_dir, source_file), allow_pickle=True).item()
        output_data = {}
        output_data.update(source_data)
        for key in ["gt", "pred"]:
            hml_data = source_data[key]["body"][0].transpose(1, 0)
            
            rec_ric_data = recover_from_ric(torch.from_numpy(hml_data).float(), joints_num).numpy()
            rec_ric_data = motion_temporal_filter(rec_ric_data)
            frames, njoints, nfeats = rec_ric_data.shape
            
            MINS = rec_ric_data.min(axis=0).min(axis=0)
            MAXS = rec_ric_data.max(axis=0).max(axis=0)
            height_offset = MINS[1]
            rec_ric_data[:, :, 1] -= height_offset
            trajec = rec_ric_data[:, 0, [0, 2]]
            
            j2s = Joints2SMPL(num_frames=frames, device_id=0, cuda=True)
            rot2xyz = Rotation2xyz(device=torch.device("cuda"))
            
            motion_tensor, opt_dict = j2s.joint2smpl(rec_ric_data)  # [nframes, njoints, 3]
            vertices = rot2xyz(torch.tensor(motion_tensor).clone(), mask=None,
                                            pose_rep='rot6d', translation=True, glob=True,
                                            jointstype='vertices',
                                            vertstrans=True)
            vertices = vertices.permute(0, 3, 1, 2).data.cpu().numpy()
            
            # Rotate the joint
            rot = R.from_euler("x", 90, degrees=True)
            vertices = rotate_joints(joints=vertices[0], rot=rot)
            output_data[key] = {"body": vertices}
        
        np.save(os.path.join(output_dir, source_file), output_data)
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='logs/avatar_gpt/eval/flan_t5_large/exp1/', help='')
    parser.add_argument('--input_dir', type=str, default="output/s1", help="")
    parser.add_argument('--output_dir', type=str, default="meshes/s1", help="")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    input_path = os.path.join(args.base_dir, args.input_dir)
    output_path = os.path.join(args.base_dir, args.output_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    main(input_dir=input_path, output_dir=output_path)