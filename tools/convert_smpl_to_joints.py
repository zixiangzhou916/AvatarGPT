import os, sys, argparse
sys.path.append(os.getcwd())
import torch
from networks import smplx_code
import numpy as np
from tqdm import tqdm

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
    
    return {"joints": joints, "vertices": vertices3d}

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_path', type=str, default="logs/avatar_gpt/eval/lora_llama/exp1/output/t2m", help='')
    # parser.add_argument('--output_path', type=str, default="logs/avatar_gpt/eval/lora_llama/exp1/joints/t2m", help='')
    parser.add_argument('--input_path', type=str, default="/cpfs/user/zhouzixiang/projects/UDE2.0/logs/ude_v2/eval/ude/clip_mtr_hubert_w_condloss/exp4-t2m-a2m-s2m/output/a2m", help='')
    parser.add_argument('--output_path', type=str, default="/cpfs/user/zhouzixiang/projects/UDE2.0/logs/ude_v2/eval/ude/clip_mtr_hubert_w_condloss/exp4-t2m-a2m-s2m/joints/a2m", help='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    smpl_model = smplx_code.create(**smpl_cfg)
    smplx_model = smplx_code.create(**smplx_cfg)
    files = os.listdir(args.input_path)
    for file in tqdm(files):
        input_data = np.load(os.path.join(args.input_path, file), allow_pickle=True).item()
        num_frames = input_data["pred"]["body"].shape[2]
        smpl_poses = {key+"_pose": torch.from_numpy(val).permute(0, 2, 1).float() for key, val in input_data["pred"].items()}
        smpl_results = convert_smpl2joints(smpl_model, **smpl_poses)
        smplx_poses = smpl_poses
        smplx_poses["body_pose"] = smplx_poses["body_pose"][..., :69].clone()
        smplx_poses["left_pose"] = torch.zeros(1, num_frames, 12).float()
        smplx_poses["right_pose"] = torch.zeros(1, num_frames, 12).float()
        caption = input_data["caption"][0]
        
        results = convert_smplx2joints(smplx_model, **smplx_poses)
        smpl_joints = results['joints'].data.cpu().numpy()   # [1, T, 144, 3]
        smpl_vertices = results['vertices'].data.cpu().numpy()   # [1, T, J, 3]
        output = {
            "pose": smpl_joints[0], "caption": caption, "vertices": smpl_results["vertices"][0].data.cpu().numpy()
        }
        np.save(os.path.join(args.output_path, file), output)
