import os, sys
sys.path.append(os.getcwd())
from funcs.hml3d.skeleton import Skeleton
from funcs.hml3d.quaternion import *
from funcs.hml3d.param_util import *
from funcs.hml3d.conversion import *
from funcs.hml3d.convert_d75_to_d263 import *
from networks import smplx_code

def main(input_dir, output_dir, plot=False):
    
    smpl_model = smplx_code.create(**smpl_cfg)
    
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
    example_file = "../dataset/HumanML3D/new_joints/000021.npy"
    example_data = np.load(example_file)
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])  # [22, 3]
    
    source_list = [f for f in os.listdir(input_dir) if ".npy" in f]
    for source_file in tqdm(source_list):
        try:
            source_data = np.load(os.path.join(input_dir, source_file), allow_pickle=True).item()
            smpl_data = source_data["motion_smpl"]  # [T, 75]
            joint = process_d75_to_joints(smpl_model=smpl_model, data=smpl_data, device=None)
            joint = joint[0, :, :joints_num] # [T, 22, 3]
            # Rotate the joints
            rot = R.from_euler("x", -90, degrees=True)
            joint = rotate_joints(joints=joint, rot=rot)

            hml3d_data, ground_positions, positions, l_velocity = process_hml3d(
                positions=joint, feet_thre=0.002,
                tgt_offsets=tgt_offsets, face_joint_indx=face_joint_indx, 
                n_raw_offsets=n_raw_offsets, 
                kinematic_chain=kinematic_chain, 
                fid_l=fid_l, fid_r=fid_r)

            output_data = {}
            output_data.update(source_data)
            output_data["motion_smpl"] = hml3d_data

            np.save(os.path.join(output_dir, source_file), output_data)
        except:
            pass
            
if __name__ == "__main__":
    input_dir = "../dataset/AIST++/aligned"
    output_dir = "../dataset/AIST++/aligned_d263"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    main(input_dir=input_dir, output_dir=output_dir, plot=False)


