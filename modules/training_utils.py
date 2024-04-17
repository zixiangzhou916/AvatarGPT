import os
import torch
import torch.nn as nn
import numpy as np
import uuid
try:
    import torchaudio
except:
    print("Unable to import torchaudio")
from networks.llm.t5_model import AvatarGPT
# from networks.llama.gpt_model import AvatarGPT as gAvatarGPT
from networks import smplx_code
# from networks.encodec.utils import convert_audio
from render.render_smplx import plot_smplx_3d, plot_smpl_3d, plot_flame_3d, animate, animate_multiple
from funcs.hml3d.conversion import recover_from_ric, motion_temporal_filter

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

GPT_MODELS = {
    ".llm.t5_model.AvatarGPT": AvatarGPT
}

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
    betas = torch.zeros(B*T, 300).float().to(device)
    expression = torch.zeros(B*T, 100).float().to(device)
    jaw_pose = torch.zeros(B*T, 3).float().to(device)
    leye_pose = torch.zeros(B*T, 3).float().to(device)
    reye_pose = torch.zeros(B*T, 3).float().to(device)
    
    joints, vertices3d = [], []
    for i in range(B*T):
        out = smplx_model(
            betas=betas[i:i+1], 
            global_orient=global_orient.reshape(B*T, 1, -1)[i:i+1], 
            body_pose=body_pose.reshape(B*T, -1)[i:i+1], 
            left_hand_pose=left_pose.reshape(B*T, -1)[i:i+1], 
            right_hand_pose=right_pose.reshape(B*T, -1)[i:i+1], 
            expression=expression[i:i+1], 
            jaw_pose=jaw_pose[i:i+1], 
            leye_pose=leye_pose[i:i+1], 
            reye_pose=reye_pose[i:i+1],
            transl=transl.reshape(B*T, -1)[i:i+1]
        )
        joints.append(out.joints.reshape(1, -1, 3))
        vertices3d.append(out.vertices.reshape(1, -1, 3))
    joints = torch.cat(joints, dim=0).reshape(B, T, -1, 3)
    vertices3d = torch.cat(vertices3d, dim=0).reshape(B, T, -1, 3)
    
    return {"joints": joints, "vertices": vertices3d}

def prepare_for_visualization_d75(smpl_model, smplx_model, gt_batch, rc_batch):
    if len(gt_batch) == 3:
        gt_joints = convert_smplx2joints(smplx_model, **gt_batch)["joints"].data.cpu().numpy()
        rc_joints = convert_smplx2joints(smplx_model, **rc_batch)["joints"].data.cpu().numpy()
    elif len(gt_batch) == 1:
        gt_joints = convert_smpl2joints(smpl_model, **gt_batch)["joints"].data.cpu().numpy()
        rc_joints = convert_smpl2joints(smpl_model, **rc_batch)["joints"].data.cpu().numpy()
    elif len(gt_batch) == 4:
        gt_joints = convert_flame2joints(self.flame_model, **gt_batch)["lmk3d"].data.cpu().numpy()
        rc_joints = convert_flame2joints(self.flame_model, **rc_batch)["lmk3d"].data.cpu().numpy()
    return gt_joints, rc_joints

def prepare_for_visualization_d263(gt_batch, rc_batch, data_obj):
    gt_joints = []
    for batch in gt_batch["body_pose"]:
        data = data_obj.inv_transform(batch.data.cpu().numpy())
        gt_joints.append(recover_from_ric(torch.from_numpy(data).float(), joints_num=22).numpy())
    gt_joints = np.stack(gt_joints, axis=0)
    
    rc_joints = []
    for batch in rc_batch["body_pose"]:
        data = data_obj.inv_transform(batch.data.cpu().numpy())
        rc_joints.append(recover_from_ric(torch.from_numpy(data).float(), joints_num=22).numpy())
    rc_joints = np.stack(rc_joints, axis=0)
    return gt_joints, rc_joints

def caption_to_name(captions):
    names = []
    for caption in captions:
        names.append(caption.replace(" ", "_").replace(",", "").replace(".", "").replace("/", ""))
    return names

def load_partial_parameters(model, checkpoint, logger=None):
    loaded_params = dict()
    for name, val in checkpoint.items():
        name_new = name.replace('module.', '') if 'module.' in name else name
        loaded_params[name_new] = val
                
    model_params = dict()
    num_condition_encoder = 0
    for name, val in model.state_dict().items():
        name_new = name.replace('module.', '') if 'module.' in name else name
        model_params[name_new] = val

    valid_params = dict()
    valid_num_condition_encoder = 0
    for src_name, src_val in loaded_params.items():
        if src_name not in model_params.keys():
            continue
        src_val_shape = ', '.join(map(str, src_val.size()))
        dst_val = model_params[src_name]
        dst_val_shape = ', '.join(map(str, dst_val.size()))
        if src_val_shape != dst_val_shape:
            print("shape of {:s} does not match: {:s} <-> {:s}".format(src_name, src_val_shape, dst_val_shape))
            continue
        suffix = 'module.' if hasattr(model, "module") else ''
        valid_params[suffix + src_name] = src_val
            
    # assert valid_num_condition_encoder == num_condition_encoder
    if logger is not None:
        logger.info(' --- loading {:.5f}% params'.format(len(valid_params) / len(model_params) * 100.))
    else:
        print(' --- loading {:.5f}% params'.format(len(valid_params) / len(model_params) * 100.))
    model.load_state_dict(valid_params, strict=False)
            
    return model
    
def load_partial_embedding(model, checkpoint, logger=None):
    """Partially load codebook embedding weights.
    """
    loaded_params = dict()
    for name, val in checkpoint.items():
        name_new = name.replace('module.', '') if 'module.' in name else name
        loaded_params[name_new] = val
                
    model_params = dict()
    num_condition_encoder = 0
    for name, val in model.state_dict().items():
        name_new = name.replace('module.', '') if 'module.' in name else name
        model_params[name_new] = val
            
    valid_params = dict()
    for src_name, src_val in loaded_params.items():
        if src_name not in model_params.keys():
            continue
        loaded_num_embedding = src_val.size(0)
        loaded_embedding_dim = src_val.size(1)
        model_num_embedding = model_params[src_name].size(0)
        model_embedding_dim = model_params[src_name].size(1)
        num_embedding = min(loaded_num_embedding, model_num_embedding)
        embedding_dim = min(loaded_embedding_dim, model_embedding_dim)
                
        embedding = model_params[src_name].clone()
        embedding[:num_embedding, :embedding_dim] = src_val[:num_embedding, :embedding_dim]
        suffix = 'module.' if hasattr(model, "module") else ''
        valid_params[suffix + src_name] = embedding
            
    # assert valid_num_condition_encoder == num_condition_encoder
    if logger is not None:
        logger.info(' --- loading {:.5f}% params'.format(len(valid_params) / len(model_params) * 100.))
    else:
        print(' --- loading {:.5f}% params'.format(len(valid_params) / len(model_params) * 100.))
    model.load_state_dict(valid_params, strict=False)
            
    return model

def build_gpt_model(conf, device, logger=None, m_quantizer=None, a_quantizer=None):
    arch_path = conf["arch_path"]
    arch_name = conf["arch_name"]
    model = GPT_MODELS[arch_path+"."+arch_name](conf, logger=logger, 
                                                m_quantizer=m_quantizer,
                                                a_quantizer=a_quantizer)
    return model.to(device)
    
def resample_audio_sequence(inp_audio, inp_sample_rate, targ_sample_rate, channels):
    B = inp_audio.size(0)
    cvt_audio = []
    for b in range(B):
        cvt_audio.append(convert_audio(inp_audio[b:b+1], inp_sample_rate, targ_sample_rate, channels))
    cvt_audio = torch.cat(cvt_audio, dim=0)
    return cvt_audio

def get_unique_str():
    return uuid.uuid4().hex[:6].upper()

def save_eval_results(results, base_path, epoch, step, task):
    output_path = os.path.join(base_path, "E{:04d}".format(epoch), task)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i, result in enumerate(results):
        file_name = os.path.join(output_path, "B{:04d}_S{:02d}.npy".format(step, i))
        np.save(file_name, result)
    
def save_wavform(results, base_path, epoch, step):
    eval_output_path = os.path.join(base_path, "eval", "B{:03d}".format(epoch))
    if not os.path.exists(eval_output_path):
        os.makedirs(eval_output_path)
    
    for sample_id in range(results["gt_wav"].size(0)):
        torchaudio.save(os.path.join(eval_output_path, "S{:03d}_T{:03d}_gt.wav".format(step, sample_id)), 
                        results["gt_wav"][sample_id].cpu(), 16000)
        torchaudio.save(os.path.join(eval_output_path, "S{:03d}_T{:03d}_rc.wav".format(step, sample_id)), 
                        results["rc_wav"][sample_id].cpu(), 24000)
            
class TrainingOutput(object):
    def __init__(self):
        self.modes = ["t2m", "m2t", "a2m", "m2a", "t2t", "se", "m2m", "s2s", "pre", "dm", "ct2t", "cs2s", "ct2s", "cs2t", "t2c", "s2c", "t2s", "s2t"]
        self.reset()
        
    def reset(self):
        self.losses = {key: {} for key in self.modes}
        self.accuracy = {key: {} for key in self.modes}
        self.pred_tokens = {key: [] for key in self.modes}
        self.target_tokens = {key: [] for key in self.modes}
        self.num = 0
        
    def update(self, input_dict, mode="t2m"):
        assert mode in self.modes
        losses = input_dict.get("losses", None)
        accuracy = input_dict.get("accuracy", None)
        pred_tokens = input_dict.get("pred_tokens", None)
        target_tokens = input_dict.get("target_tokens", None)
        
        if losses is not None:
            self.update_losses(losses=losses, mode=mode)
        if accuracy is not None:
            self.update_accuracy(accuracy=accuracy, mode=mode)
        if pred_tokens is not None:
            self.update_pred_tokens(pred_tokens=pred_tokens, mode=mode)
        if target_tokens is not None:
            self.update_target_tokens(target_tokens=target_tokens, mode=mode)
        
    def update_num(self):
        self.num += 1
        
    def get_losses(self, mode="t2m"):
        output = {}
        for k, v in self.losses[mode].items():
            output[k] = v / self.num
        return output
    
    def get_accuracy(self, mode="t2m"):
        output = {}
        for k, v in self.accuracy[mode].items():
            output[k] = v / self.num
        return output
    
    def get_pred_tokens(self, mode="t2m"):
        return self.pred_tokens[mode]
    
    def get_target_tokens(self, mode="t2m"):
        return self.target_tokens[mode]
    
    def has_losses(self, mode="t2m"):
        data = self.losses[mode]
        if len(data) == 0: 
            return False
        else:
            return True
        
    def has_accuracy(self, mode="t2m"):
        data = self.accuracy[mode]
        if len(data) == 0: 
            return False
        else:
            return True
        
    def has_pred_tokens(self, mode="t2m"):
        data = self.pred_tokens[mode]
        if len(data) == 0: 
            return False
        else:
            return True
    
    def has_target_tokens(self, mode="t2m"):
        data = self.target_tokens[mode]
        if len(data) == 0: 
            return False
        else:
            return True
            
    def update_losses(self, losses, mode="t2m"):
        for k, v in losses.items():
            if k in self.losses[mode].keys():
                self.losses[mode][k] += v
            else:
                self.losses[mode][k] = v
                
    def update_accuracy(self, accuracy, mode="t2m"):
        for k, v in accuracy.items():
            if k in self.accuracy[mode].keys():
                self.accuracy[mode][k] += v
            else:
                self.accuracy[mode][k] = v
                
    def update_pred_tokens(self, pred_tokens, mode="t2m"):
        self.pred_tokens[mode] += [val for val in pred_tokens]
        
    def update_target_tokens(self, target_tokens, mode="t2m"):
        self.target_tokens[mode] += [val for val in target_tokens]

TASK_MAP = {
    "t2m": "Text-to-Motion", 
    "m2t": "Motion-to-Text", 
    "m2m": "Motion-to-Motion", 
    "dm": "Decision-Making", 
    "ct2t": "Scene-Task-to-Task", 
    "cs2s": "Scene-Steps-to-Steps", 
    "ct2s": "Scene-Task-to-Steps", 
    "cs2t": "Scene-Steps-to-Task", 
    "t2c": "Task-to-Scene", 
    "s2c": "Steps-to-Scene", 
    "t2s": "Task-to-Steps", 
    "s2t": "Steps-to-Task", 
    "pre": "Pre-Train"
}