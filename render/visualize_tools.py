import os, sys
sys.path.append(os.getcwd())
import numpy as np
import torch
from render.ude_v2.render_smplx import convert_smpl2joints, render_animation
from networks import smplx_code
from networks.roma.mappings import *
from scipy.spatial.transform import Rotation as R

smpl_cfg = dict(
    model_path="networks/smpl/SMPL_NEUTRAL.pkl", 
    model_type="smpl", 
    gender="neutral", 
    batch_size=1,
)

def visualize_wo_orient(smpl_inp, output_path):
    """Visualize the SMPL joints without global orientation.
    """
    smpl_model = smplx_code.create(**smpl_cfg)
    
    temp_output_path = os.path.join(output_path, "temp")
    if not os.path.exists(temp_output_path): 
        os.makedirs(temp_output_path)
    
    # Set the global orientation of [0, -1, 0]
    rot = R.from_euler("X", 90, degrees=True).as_rotvec()
    smpl_inp[:, 3:6] = rot[None]
    smpl_inp[:, :3] *= 0.0
    smpl_pose = {"body_pose": torch.from_numpy(smpl_inp).float()[None]}
    joints = convert_smpl2joints(smpl_model, **smpl_pose)["joints"].data.cpu().numpy()    
    render_animation([joints[0][::10]], output_path, "seq", plot_type="smpl", prefixs=["temp"], video_type="mp4")
    
def visualize_global_orient_to_local_orient(smpl_inp, ouptut_path):
    """Visualize the SMPL joints with global orientation converted to local orientation.
    """
    smpl_model = smplx_code.create(**smpl_cfg)
    
    temp_output_path = os.path.join(output_path, "temp")
    if not os.path.exists(temp_output_path): 
        os.makedirs(temp_output_path)
            
    # Set the global orientation to (0.5*pi, 0, 0)
    rot = R.from_euler("X", 90, degrees=True).as_rotvec()
    glob_orien = np.copy(smpl_inp[:, 3:6])
    glob_trans = np.copy(smpl_inp[:, :3])
    
    # Set the global translation to (0, 0, 0)
    smpl_inp[:, :3] *= 0.0
    smpl_inp[:, 3:6] = rot[None]
    
    # Get the residual orientation between consecutive frames
    residual_orien = []
    for i in range(1, glob_orien.shape[0], 1):
        dst_rotvec = glob_orien[i:i+1]
        src_rotvec = glob_orien[i-1:i]
        dst_rot = R.from_rotvec(dst_rotvec).as_matrix()
        src_rot = R.from_rotvec(src_rotvec).as_matrix()
        src2dst_rot = np.matmul(dst_rot, np.linalg.inv(src_rot))
        src2dst_rotvec = R.from_matrix(src2dst_rot).as_rotvec()
        # Estimate the rotation matrix from src_rotvec to dst_rotvec
        residual_orien.append(src2dst_rotvec)
    # residual_orien = R.align_vectors(glob_orien[1:], glob_orien[:-1])[0].as_rotvec()
    residual_orien = np.stack(residual_orien, axis=0)
    
    # Animate the joints w/o orientation and translation
    smpl_pose = {"body_pose": torch.from_numpy(smpl_inp).float()[None]}
    joints = convert_smpl2joints(smpl_model, **smpl_pose)["joints"].data.cpu().numpy()    
    render_animation([joints[0][::10]], output_path, "seq", plot_type="smpl", prefixs=["temp"], video_type="mp4")
    
    # Restore the global orientation from residual orientation
    restored_orien = [glob_orien[0]]
    for i in range(residual_orien.shape[0]):
        m1 = R.from_rotvec(restored_orien[-1]).as_matrix()      # current global orientation
        m2 = R.from_rotvec(residual_orien[i]).as_matrix()       # Residual rotation matrix
        m3 = np.matmul(m2, m1)
        orien = R.from_matrix(m3).as_rotvec()
        # print(glob_orien[i+1], "|", orien, "|", restored_orien[-1], "|", residual_orien[i])
        # exit(0)
        restored_orien.append(orien[0])
    restored_orien = np.stack(restored_orien, axis=0)
    print(restored_orien.shape)
    
    # Animate the restored joints w/ orientation and translation
    smpl_inp[:, :3] = glob_trans
    smpl_inp[:, 3:6] = restored_orien
    smpl_pose = {"body_pose": torch.from_numpy(smpl_inp).float()[None]}
    joints = convert_smpl2joints(smpl_model, **smpl_pose)["joints"].data.cpu().numpy()    
    render_animation([joints[0][::10]], output_path, "seq_2", plot_type="smpl", prefixs=["temp"], video_type="mp4")
    
def visualize_global_orient_to_local_orient_torch(smpl_inp, output_path):
    """Visualize the SMPL joints with global orientation converted to local orientation.
    """
    smpl_model = smplx_code.create(**smpl_cfg)
    
    temp_output_path = os.path.join(output_path, "temp")
    if not os.path.exists(temp_output_path): 
        os.makedirs(temp_output_path)
    
    # Convert to torch.Tensor
    smpl_inp = torch.from_numpy(smpl_inp).float()
    
    # Set the global orientation to (0.5*pi, 0, 0)
    rot = R.from_euler("X", 90, degrees=True).as_rotvec()
    glob_orien = smpl_inp[:, 3:6].clone()
    glob_trans = smpl_inp[:, :3].clone()
        
    # Get the residual orientation between consecutive frames
    # 1). Convert rotvec to quanternion, then to rotation matrix
    quat = rotvec_to_unitquat(glob_orien)
    rot = unitquat_to_rotmat(quat)
    
    # 2). Compute the residual rotation matrixs between consecutive frames
    residual_orien = []
    src_rot = rot[:-1]
    dst_rot = rot[1:]
    src2dst_rot = torch.matmul(dst_rot, torch.linalg.inv(src_rot))
    residual_orien = rotmat_to_rotvec(src2dst_rot)
        
    # Restore the global orientation from residual orientation
    restored_orien = [glob_orien[0]]
    for i in range(residual_orien.shape[0]):
        m1 = rotvec_to_rotmat(restored_orien[-1])
        m2 = rotvec_to_rotmat(residual_orien[i])
        m3 = torch.matmul(m2, m1)
        orien = rotmat_to_rotvec(m3)
        restored_orien.append(orien)
    restored_orien = torch.stack(restored_orien, dim=0)
    print(restored_orien.shape)
    
    # Animate the restored joints w/ orientation and translation
    smpl_inp[:, :3] = glob_trans
    smpl_inp[:, 3:6] = restored_orien
    smpl_pose = {"body_pose": smpl_inp[None]}
    joints = convert_smpl2joints(smpl_model, **smpl_pose)["joints"].data.cpu().numpy()    
    render_animation([joints[0][::10]], output_path, "seq_3", plot_type="smpl", prefixs=["temp"], video_type="mp4")
    
 
    
if __name__ == "__main__":
    data = np.load("../dataset/AIST++/aligned/gWA_sFM_cAll_d27_mWA2_ch17.npy", allow_pickle=True).item()
    smpl_inp = data["motion_smpl"]
    output_path = "logs/ude_v2/debug/"
    visualize_global_orient_to_local_orient_torch(smpl_inp, output_path)
    
    
    
