import os, sys, argparse
sys.path.append(os.getcwd())
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import numpy as np
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy import linalg
from tqdm import tqdm
import codecs as cs
import pickle
from collections import OrderedDict
from networks.evaluation.modules import *
from networks import smplx_code
from funcs.hml3d.skeleton import Skeleton
from funcs.hml3d.quaternion import *
from funcs.hml3d.param_util import *
from funcs.hml3d.conversion import *
from funcs.hml3d.convert_d75_to_d263 import *
from tools.llm_planning_evaluator import *

import json
import numpy as np
import pickle
from os.path import join as pjoin

POS_enumerator = {
    'VERB': 0,
    'NOUN': 1,
    'DET': 2,
    'ADP': 3,
    'NUM': 4,
    'AUX': 5,
    'PRON': 6,
    'ADJ': 7,
    'ADV': 8,
    'Loc_VIP': 9,
    'Body_VIP': 10,
    'Obj_VIP': 11,
    'Act_VIP': 12,
    'Desc_VIP': 13,
    'OTHER': 14,
}

Loc_list = ('left', 'right', 'clockwise', 'counterclockwise', 'anticlockwise', 'forward', 'back', 'backward',
            'up', 'down', 'straight', 'curve')

Body_list = ('arm', 'chin', 'foot', 'feet', 'face', 'hand', 'mouth', 'leg', 'waist', 'eye', 'knee', 'shoulder', 'thigh')

Obj_List = ('stair', 'dumbbell', 'chair', 'window', 'floor', 'car', 'ball', 'handrail', 'baseball', 'basketball')

Act_list = ('walk', 'run', 'swing', 'pick', 'bring', 'kick', 'put', 'squat', 'throw', 'hop', 'dance', 'jump', 'turn',
            'stumble', 'dance', 'stop', 'sit', 'lift', 'lower', 'raise', 'wash', 'stand', 'kneel', 'stroll',
            'rub', 'bend', 'balance', 'flap', 'jog', 'shuffle', 'lean', 'rotate', 'spin', 'spread', 'climb')

Desc_list = ('slowly', 'carefully', 'fast', 'careful', 'slow', 'quickly', 'happy', 'angry', 'sad', 'happily',
             'angrily', 'sadly')

VIP_dict = {
    'Loc_VIP': Loc_list,
    'Body_VIP': Body_list,
    'Obj_VIP': Obj_List,
    'Act_VIP': Act_list,
    'Desc_VIP': Desc_list,
}

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] >= multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()

class Resizer(object):
    def __init__(self) -> None:
        self.joints_num = 22
        self.l_idx1 = 5 
        self.l_idx2 = 8
        
        # Right/Left foot
        self.fid_r = [8, 11]
        self.fid_l = [7, 10]
        # Face direction, r_hip, l_hip, sdr_r, sdr_l
        self.face_joint_indx = [2, 1, 17, 16]
        # l_hip, r_hip
        self.r_hip = 2
        self.l_hip = 1
        self.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)   # [22, 3]
        self.kinematic_chain = t2m_kinematic_chain
        example_file = "tools/000021.npy"
        example_data = np.load(example_file)
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        tgt_skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, 'cpu')
        self.tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])  # [22, 3]
    
    def recover_from_ric(self, data, joints_num):
        r_rot_quat, r_pos = recover_root_rot_pos(data)
        positions = data[..., 4:(joints_num - 1) * 3 + 4]
        positions = positions.view(positions.shape[:-1] + (-1, 3))

        '''Add Y-axis rotation to local joints'''
        positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

        '''Add root XZ to joints'''
        positions[..., 0] += r_pos[..., 0:1]
        positions[..., 2] += r_pos[..., 2:3]

        '''Concate root and joints'''
        positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

        return positions
    
    def uniform_skeleton(
        self, positions, target_offset, 
        face_joint_indx, n_raw_offsets, 
        kinematic_chain, l_idx1, l_idx2
    ):
        src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
        src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
        src_offset = src_offset.numpy()
        tgt_offset = target_offset.numpy()
        # print(src_offset)
        # print(tgt_offset)
        '''Calculate Scale Ratio as the ratio of legs'''
        src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
        tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

        scale_rt = tgt_leg_len / src_leg_len
        # print(scale_rt)
        src_root_pos = positions[:, 0]
        tgt_root_pos = src_root_pos * scale_rt

        '''Inverse Kinematics'''
        quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
        # print(quat_params.shape)

        '''Forward Kinematics'''
        src_skel.set_offset(target_offset)
        new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
        return new_joints
    
    def process_hml3d_joints(
        self, positions, feet_thre, 
        tgt_offsets, face_joint_indx, 
        n_raw_offsets, 
        kinematic_chain, 
        fid_l, fid_r
    ):
        '''Uniform Skeleton'''
        positions = self.uniform_skeleton(
            positions, tgt_offsets, 
            face_joint_indx=face_joint_indx, 
            n_raw_offsets=n_raw_offsets, 
            kinematic_chain=kinematic_chain, 
            l_idx1=5, l_idx2=8)
        
        '''Put on Floor'''
        floor_height = positions.min(axis=0).min(axis=0)[1]
        positions[:, :, 1] -= floor_height

        '''XZ at origin'''
        root_pos_init = positions[0]
        root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
        positions = positions - root_pose_init_xz

        '''All initially face Z+'''
        r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
        across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
        across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

        # forward (3,), rotate around y-axis
        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        # forward (3,)
        forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

        target = np.array([[0, 0, 1]])
        root_quat_init = qbetween_np(forward_init, target)
        root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

        positions_b = positions.copy()

        positions = qrot_np(root_quat_init, positions)

        '''New ground truth positions'''
        global_positions = positions.copy()
        
        """ Get Foot Contacts """
    
        def foot_detect(positions, thres):
            velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

            feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
            feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
            feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
            #     feet_l_h = positions[:-1,fid_l,1]
            #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
            feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

            feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
            feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
            feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
            #     feet_r_h = positions[:-1,fid_r,1]
            #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
            feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
            return feet_l, feet_r
    
        feet_l, feet_r = foot_detect(positions, feet_thre)
        
        '''Quaternion and Cartesian representation'''
        r_rot = None

        def get_rifke(positions):
            '''Local pose'''
            positions[..., 0] -= positions[:, 0:1, 0]
            positions[..., 2] -= positions[:, 0:1, 2]
            '''All pose face Z+'''
            positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
            return positions
        
        def get_quaternion(positions):
            skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
            # (seq_len, joints_num, 4)
            quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

            '''Fix Quaternion Discontinuity'''
            quat_params = qfix(quat_params)
            # (seq_len, 4)
            r_rot = quat_params[:, 0].copy()
            #     print(r_rot[0])
            '''Root Linear Velocity'''
            # (seq_len - 1, 3)
            velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
            #     print(r_rot.shape, velocity.shape)
            velocity = qrot_np(r_rot[1:], velocity)
            '''Root Angular Velocity'''
            # (seq_len - 1, 4)
            r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
            quat_params[1:, 0] = r_velocity
            # (seq_len, joints_num, 4)
            return quat_params, r_velocity, velocity, r_rot
        
        def get_cont6d_params(positions):
            skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
            # (seq_len, joints_num, 4)
            quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

            '''Quaternion to continuous 6D'''
            cont_6d_params = quaternion_to_cont6d_np(quat_params)
            # (seq_len, 4)
            r_rot = quat_params[:, 0].copy()
            #     print(r_rot[0])
            '''Root Linear Velocity'''
            # (seq_len - 1, 3)
            velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
            #     print(r_rot.shape, velocity.shape)
            velocity = qrot_np(r_rot[1:], velocity)
            '''Root Angular Velocity'''
            # (seq_len - 1, 4)
            r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
            # (seq_len, joints_num, 4)
            return cont_6d_params, r_velocity, velocity, r_rot
        
        cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
        positions = get_rifke(positions)
        
        '''Root height'''
        root_y = positions[:, 0, 1:2]

        '''Root rotation and linear velocity'''
        # (seq_len-1, 1) rotation velocity along y-axis
        # (seq_len-1, 2) linear velovity on xz plane
        r_velocity = np.arcsin(r_velocity[:, 2:3])
        l_velocity = velocity[:, [0, 2]]
        #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
        root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

        '''Get Joint Rotation Representation'''
        # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
        rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

        '''Get Joint Rotation Invariant Position Represention'''
        # (seq_len, (joints_num-1)*3) local joint position
        ric_data = positions[:, 1:].reshape(len(positions), -1)

        '''Get Joint Velocity Representation'''
        # (seq_len-1, joints_num*3)
        local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                            global_positions[1:] - global_positions[:-1])
        local_vel = local_vel.reshape(len(local_vel), -1)

        data = root_data
        data = np.concatenate([data, ric_data[:-1]], axis=-1)
        data = np.concatenate([data, rot_data[:-1]], axis=-1)
        #     print(data.shape, local_vel.shape)
        data = np.concatenate([data, local_vel], axis=-1)
        data = np.concatenate([data, feet_l, feet_r], axis=-1)
        
        return data, global_positions, positions, l_velocity

    @staticmethod
    def resize(inp_motion, target_length):
        """
        :inp_motion: [T, J, 3]
        """
        input_length = inp_motion.shape[0]
        y = np.reshape(inp_motion, newshape=(input_length, -1)) # [T, J*3]
        x = np.arange(0, target_length)
        if target_length < input_length:
            x = np.random.choice(target_length, size=(input_length,), replace=True)
        else:
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

    def run(self, input_data, target_length):
        """
        :param input_data: [T, 263]
        """
        # Convert input data to joints
        joints = self.recover_from_ric(
            torch.from_numpy(input_data).float(), 
            joints_num=self.joints_num).numpy()
        joints = motion_temporal_filter(joints)
        
        # Resize the joints sequence
        joints = self.resize(joints, target_length=target_length)
        
        data, ground_positions, positions, l_velocity = self.process_hml3d_joints(
            positions=joints, feet_thre=0.002,
            tgt_offsets=self.tgt_offsets, face_joint_indx=self.face_joint_indx, 
            n_raw_offsets=self.n_raw_offsets, 
            kinematic_chain=self.kinematic_chain, 
            fid_l=self.fid_l, fid_r=self.fid_r)
        
        return data
    
class WordVectorizer(object):
    def __init__(self, meta_root, prefix):
        vectors = np.load(pjoin(meta_root, '%s_data.npy'%prefix))
        words = pickle.load(open(pjoin(meta_root, '%s_words.pkl'%prefix), 'rb'))
        self.word2idx = pickle.load(open(pjoin(meta_root, '%s_idx.pkl'%prefix), 'rb'))
        self.word2vec = {w: vectors[self.word2idx[w]] for w in words}

    def _get_pos_ohot(self, pos):
        pos_vec = np.zeros(len(POS_enumerator))
        if pos in POS_enumerator:
            pos_vec[POS_enumerator[pos]] = 1
        else:
            pos_vec[POS_enumerator['OTHER']] = 1
        return pos_vec

    def __len__(self):
        return len(self.word2vec)

    def __getitem__(self, item):
        word, pos = item.split('/')
        if word in self.word2vec:
            word_vec = self.word2vec[word]
            vip_pos = None
            for key, values in VIP_dict.items():
                if word in values:
                    vip_pos = key
                    break
            if vip_pos is not None:
                pos_vec = self._get_pos_ohot(vip_pos)
            else:
                pos_vec = self._get_pos_ohot(pos)
        else:
            word_vec = self.word2vec['unk']
            pos_vec = self._get_pos_ohot('OTHER')
        return word_vec, pos_vec

class WordVectorizerV2(WordVectorizer):
    def __init__(self, meta_root, prefix):
        super(WordVectorizerV2, self).__init__(meta_root, prefix)
        self.idx2word = {self.word2idx[w]: w for w in self.word2idx}

    def __getitem__(self, item):
        word_vec, pose_vec = super(WordVectorizerV2, self).__getitem__(item)
        word, pos = item.split('/')
        if word in self.word2vec:
            return word_vec, pose_vec, self.word2idx[word]
        else:
            return word_vec, pose_vec, self.word2idx['unk']

    def itos(self, idx):
        if idx == len(self.idx2word):
            return "pad"
        return self.idx2word[idx]
    
class GTDataset(data.Dataset):
    def __init__(self, mean, std):
        self.w_vectorizer = WordVectorizer("tools/glove", "our_vab")
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = 196
        self.min_motion_len = 40
        self.unit_length = 4
        self.max_text_len = 20
        self.motion_dir = "../dataset/HumanML3D/new_joint_vecs"
        self.text_dir = "../dataset/HumanML3D/texts"
        
        data_dict = {}
        id_list = []
        with cs.open("../dataset/HumanML3D/test.txt", "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        
        new_name_list = []
        length_list = []
        for name in tqdm(id_list[:]):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if (len(motion)) < self.min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < self.min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {
                                    'motion': n_motion,
                                    'length': len(n_motion),
                                    'text':[text_dict]
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        'motion': motion,
                        'length': len(motion),
                        'text': text_data
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass
            
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        
    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [motion, np.zeros((self.max_motion_length - m_length, motion.shape[1]))], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)
    
class TextPreprocessor(object):
    def __init__(self, args):
        self.base_path = "../dataset/HumanML3D/texts"
        self.max_text_len = 20
        self.text_data = []
        self.w_vectorizer = WordVectorizer("tools/glove", "our_vab")
        
        files = [f for f in os.listdir(self.base_path) if ".txt" in f]
        for name in tqdm(files):
            text_data = []
            flag = False
            with cs.open(os.path.join(self.base_path, name)) as f:
                for line in f.readlines():
                    text_dict = {}
                    line_split = line.strip().split('#')
                    caption = line_split[0]
                    tokens = line_split[1].split(' ')
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag
                    
                    text_dict['caption'] = caption
                    text_dict['tokens'] = tokens
                    text_dict['file_name'] = name
                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                    
            if flag:
                self.text_data.append(text_data)
                
    def run(self):
        
        results = []
        
        for data in tqdm(self.text_data):
            text_list = data
            for text_data in text_list:
                caption, tokens, file_name = text_data["caption"], text_data["tokens"], text_data["file_name"]

                if len(tokens) < self.max_text_len:
                    tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                    sent_len = len(tokens)
                    tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
                else:
                    # crop
                    tokens = tokens[:self.max_text_len]
                    tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                    sent_len = len(tokens)
                if sent_len <= 0: 
                    print('-' * 10)
                    continue
                pos_one_hots = []
                word_embeddings = []
                for token in tokens:
                    word_emb, pos_oh = self.w_vectorizer[token]
                    pos_one_hots.append(pos_oh[None, :])
                    word_embeddings.append(word_emb[None, :])
                pos_one_hots = np.concatenate(pos_one_hots, axis=0)
                word_embeddings = np.concatenate(word_embeddings, axis=0)
                
                results.append(
                    {
                        "caption": caption,
                        "tokens": tokens, 
                        "word_emb": word_embeddings, 
                        "pos_oh": pos_one_hots, 
                        "sent_len": sent_len, 
                        "file_name": file_name
                    }
                )
                
        with open("data/preprocessed_texts.pickle", "wb") as f:
            pickle.dump(results, f)

class GTMotionPreprocessor(object):
    def __init__(self, args):
        self.mean = np.load(args.mean_dir)
        self.std = np.load(args.std_dir)
        self.unit_length = 4
        self.max_motion_length = 196
        self.min_motion_length = 40
        self.input_dir = args.motion_dir
        self.read_split_file(args.split_dir)
        self.read_processed_texts()
        
    def read_processed_texts(self):
        with open("data/preprocessed_texts.pickle", "rb") as f:
            self.text_dict = pickle.load(f)
        
        # Reorganize the text data using their filenames as key
        self.text_map = {}
        for item in self.text_dict:
            file_name = item["file_name"]
            new_item = {}
            new_item.update(item)
            if file_name in self.text_map.keys():
                self.text_map[file_name].append(new_item)
            else:
                self.text_map[file_name] = [new_item]
    
    def read_split_file(self, split_dir):
        file_lists = []
        with cs.open(split_dir, "r") as f:
            for line in f.readlines():
                file_lists.append(line.strip())
        
        all_files = [f for f in os.listdir(self.input_dir) if ".npy" in f]
        self.files = []
        for file in all_files:
            base_name = file.split(".")[0]
            if base_name in file_lists:
                self.files.append(file)
    
    def run(self):
        
        batches = []
        for file in tqdm(self.files):
            key = file.replace(".npy", ".txt")
            if key not in self.text_map.keys():
                continue
            
            text_info = self.text_map[key]
            text_data = random.choice(text_info)
            
            motion = np.load(os.path.join(self.input_dir, file))
            m_length = motion.shape[0]
            
            if m_length < self.min_motion_length or m_length >= 200: 
                continue
            
            if self.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'
            
            if coin2 == 'double':
                m_length = (m_length // self.unit_length - 1) * self.unit_length
            elif coin2 == 'single':
                m_length = (m_length // self.unit_length) * self.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx+m_length]
            
            """ Z Normalization """
            motion = (motion - self.mean) / self.std
            
            if m_length < self.max_motion_length:
                motion = np.concatenate([motion, np.zeros((self.max_motion_length - m_length, motion.shape[1]))], axis=0)
            elif m_length > self.max_motion_length:
                motion = motion[:self.max_motion_length]
                
            if m_length < 10: continue
            
            batch = {
                "motion": motion, 
                "m_length": m_length, 
            }
            batch.update(text_data)
            batches.append(batch)
        
        with open("data/preprocessed_gt_motions.pickle", "wb") as f:
            pickle.dump(batches, f)

class PredMotionPreprocessor(object):
    def __init__(self, arg) -> None:
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.motion_repr = arg.motion_repr
        self.mean = np.load(args.mean_dir)
        self.std = np.load(args.std_dir)
        self.motion_dir = args.motion_dir
        self.min_motion_length = 40
        self.max_motion_length = 196
        self.unit_length = 4
        self.smpl_model = smplx_code.create(**smpl_cfg).to(self.device)
        self.files = [f for f in os.listdir(self.motion_dir) if ".npy" in f]
        self.read_processed_texts()
        self.resizer = Resizer()
        
    def read_processed_texts(self):
        with open("data/preprocessed_texts.pickle", "rb") as f:
            self.text_dict = pickle.load(f)
        
        # Reorganize the text data using their filenames as key
        self.text_map = {}
        for item in self.text_dict:
            caption = item["caption"]
            new_item = {}
            new_item.update(item)
            if caption in self.text_map.keys():
                self.text_map[caption].append(new_item)
            else:
                self.text_map[caption] = [new_item]
    
    def run_d75(self):
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

        batches = {}
        for file in tqdm(self.files):
            # Parse the input file to get the tid
            base_name = file.split(".")[0]
            tid = base_name.split("_")[1]
            
            data = np.load(os.path.join(self.motion_dir, file), allow_pickle=True).item()
            caption = data["caption"][0]
            
            if caption not in self.text_map.keys(): continue
            text_info = self.text_map[caption]
            text_data = random.choice(text_info)
            
            try:
                motion = data["pred"]["body"]
            except:
                motion = data["motion"]
            motion = motion[0].transpose(1, 0)  # [T, 75]
            joints = process_d75_to_joints(smpl_model=self.smpl_model, data=motion, device=self.device)
            joints = joints[0, :, :joints_num]
            # Rotate the joints
            rot = R.from_euler("x", -90, degrees=True)
            joints = rotate_joints(joints=joints, rot=rot)
            joints = motion_temporal_filter(joints)
            # Convert joints to poses
            poses, ground_positions, positions, l_velocity = process_hml3d(
                positions=joints, feet_thre=0.002,
                tgt_offsets=tgt_offsets, face_joint_indx=face_joint_indx, 
                n_raw_offsets=n_raw_offsets, 
                kinematic_chain=kinematic_chain, 
                fid_l=fid_l, fid_r=fid_r)

            m_length = poses.shape[0]
            
            if m_length < self.min_motion_length or m_length > 200:
                continue
            
            if self.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'
            
            if coin2 == 'double':
                m_length = (m_length // self.unit_length - 1) * self.unit_length
            elif coin2 == 'single':
                m_length = (m_length // self.unit_length) * self.unit_length
            idx = random.randint(0, len(poses) - m_length)
            poses = poses[idx:idx+m_length]
            
            """ Z Normalization """
            poses = (poses - self.mean) / self.std
            
            if m_length < self.max_motion_length:
                poses = np.concatenate([poses, np.zeros((self.max_motion_length - m_length, poses.shape[1]))], axis=0)
            elif m_length > self.max_motion_length:
                poses = poses[:self.max_motion_length]
            
            if m_length < 10: continue
            
            batch = {
                "motion": poses, 
                "m_length": m_length, 
                "caption": caption
            }
            # batch.update(text_data)
            
            if tid in batches.keys():
                batches[tid].append(batch)
            else:
                batches[tid] = [batch]
        
        output_dir = os.path.split(self.motion_dir)[0]
        with open("{:s}/preprocessed_motions.pickle".format(output_dir), "wb") as f:
            pickle.dump(batches, f)
    
    def run_d263(self):
        batches = {}
        for file in tqdm(self.files):
            # Parse the input file to get the tid
            base_name = file.split(".")[0]
            tid = base_name.split("_")[1]
            
            data = np.load(os.path.join(self.motion_dir, file), allow_pickle=True).item()
            caption = data["caption"][0]
            
            if caption not in self.text_map.keys(): continue
            text_info = self.text_map[caption]
            text_data = random.choice(text_info)
            
            try:
                gt_motion = data["gt"]["body"]
                rc_motion = data["pred"]["body"]
            except:
                gt_motion = data["gt"]
                rc_motion = data["motion"]
            gt_poses = gt_motion[0].transpose(1, 0)  # [T, 263]    
            rc_poses = rc_motion[0].transpose(1, 0)  # [T, 263]
                
            min_len = min(gt_poses.shape[0], rc_poses.shape[0])
            # gt_poses = gt_poses[:min_len]   # Crop the sequence to minimum length
            # rc_poses = rc_poses[:min_len]   # Crop the sequence to minimum length
            """Case 1. We would like to resize the length of predicted motion."""
            # m_length = gt_poses.shape[0]
            # poses = self.resizer.run(rc_poses, target_length=gt_poses.shape[0]+1)
            """Case 2. We keep the original length of predicted motion."""
            m_length = rc_poses.shape[0]
            poses = rc_poses.copy()
            
            if m_length < self.min_motion_length or m_length > 200:
                continue
            
            if self.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'
            
            if coin2 == 'double':
                m_length = (m_length // self.unit_length - 1) * self.unit_length
            elif coin2 == 'single':
                m_length = (m_length // self.unit_length) * self.unit_length
            # idx = random.randint(0, len(poses) - m_length)
            # poses = poses[idx:idx+m_length]
            poses = poses[:m_length]
            
            """ Z Normalization """
            poses = (poses - self.mean) / self.std
            
            if m_length < self.max_motion_length:
                poses = np.concatenate([poses, np.zeros((self.max_motion_length - m_length, poses.shape[1]))], axis=0)
            elif m_length > self.max_motion_length:
                poses = poses[:self.max_motion_length]
                m_length = poses.shape[0]
            
            if m_length < 10: continue
            
            if poses.shape[1] != 263 or poses.shape[0] != self.max_motion_length:
                print('Invalid', poses.shape)
                        
            batch = {
                "motion": poses, 
                "m_length": m_length, 
                "caption": caption
            }
            if tid in batches.keys():
                batches[tid].append(batch)
            else:
                batches[tid] = [batch]
        
        output_dir = os.path.split(self.motion_dir)[0]
        with open("{:s}/preprocessed_motions.pickle".format(output_dir), "wb") as f:
            pickle.dump(batches, f)
    
    def run(self):
        if self.motion_repr == "d75":
            self.run_d75()
        elif self.motion_repr == "d263":
            self.run_d263()

class RPrecision(object):
    def __init__(self, args):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.replications = args.replications
        self.motion_dir = args.motion_dir
        self.unit_length = 4
        self.movement_enc = MovementConvEncoder(input_size=263-4, hidden_size=512, output_size=512).to(self.device)
        self.text_enc = TextEncoderBiGRUCo(
            word_size=300,
            pos_size=15,
            hidden_size=512,
            output_size=512,
            device="cpu").to(self.device)
        self.motion_enc = MotionEncoderBiGRUCo(
            input_size=512,
            hidden_size=1024,
            output_size=512,
            device="cpu").to(self.device)
        
        checkpoint = torch.load(
            "networks/evaluation/pretrained/text_mot_match/model/finest.tar", 
            map_location=torch.device("cpu"))
        self.movement_enc.load_state_dict(checkpoint['movement_encoder'])
        self.text_enc.load_state_dict(checkpoint['text_encoder'])
        self.motion_enc.load_state_dict(checkpoint['motion_encoder'])
        print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
        
        self.movement_enc.eval()
        self.text_enc.eval()
        self.motion_enc.eval()
        
        # Load preprocessed texts
        self.read_processed_texts()
        
        # Load preprocessed motions
        input_dir = os.path.split(self.motion_dir)[0]
        with open("{:s}/preprocessed_motions.pickle".format(input_dir), "rb") as f:
            self.batches = pickle.load(f)
    
    def read_processed_texts(self):
        with open("data/preprocessed_texts.pickle", "rb") as f:
            self.text_dict = pickle.load(f)
        
        # Reorganize the text data using their filenames as key
        self.text_map_by_caption = {}
        self.text_map_by_filename = {}
        for item in self.text_dict:
            caption = item["caption"]
            filename = item["file_name"]
            new_item = {}
            new_item.update(item)
            if caption in self.text_map_by_caption.keys():
                self.text_map_by_caption[caption].append(new_item)
            else:
                self.text_map_by_caption[caption] = [new_item]
            if filename in self.text_map_by_filename.keys():
                self.text_map_by_filename[filename].append(new_item)
            else:
                self.text_map_by_filename[filename] = [new_item]
                
    def prepare_batch(self, inp_batch):
        """Prepare the input batch."""
        out_batch = []
        for batch in inp_batch:
            caption = batch["caption"]
            try:
                info_by_caption = self.text_map_by_caption[caption]
                # filenames = [d["file_name"] for d in info_by_caption]
                # info_by_caption = random.choice(info_by_caption)
                # filename = info_by_caption["file_name"]
                # if len(filenames) == 1:
                #     filename = filenames[0]
                # info_by_filename = self.text_map_by_filename[filename]
                # info_by_filename = random.choice(info_by_filename)
                batch.update(info_by_caption[0])
                out_batch.append(batch)
            except:
                pass
        return out_batch
                
    @torch.no_grad()
    def get_co_embeddings(self, batches):
        """Calculate the text embedding and motion embedding.
        This function is called when evaluating metrics of Prediction.
        """
        cap_lens = [d["sent_len"] for d in batches]
        sort_index = np.argsort(cap_lens)[::-1].copy()
        batches = [batches[i] for i in sort_index]
        
        word_embs, pos_ohot, motions, m_lens, cap_lens = [], [], [], [], []
        for batch in batches:
            word_embs.append(batch["word_emb"])
            pos_ohot.append(batch["pos_oh"])
            motions.append(batch["motion"])
            m_lens.append(batch["m_length"])
            cap_lens.append(batch["sent_len"])
        word_embs = np.stack(word_embs, axis=0)
        pos_ohot = np.stack(pos_ohot, axis=0)
        motions = np.stack(motions, axis=0)
        m_lens = np.asarray(m_lens)
        cap_lens = torch.from_numpy(np.asarray(cap_lens)).to(self.device).long()
        
        word_embs = torch.from_numpy(word_embs).to(self.device).float()
        pos_ohot = torch.from_numpy(pos_ohot).to(self.device).float()
        motions = torch.from_numpy(motions).to(self.device).float()
        align_idx = np.argsort(m_lens.tolist())[::-1].copy()
        
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]
        
        # print('---', cap_lens.min(), np.min(m_lens))
        
        '''Movement Encoding'''
        movements = self.movement_enc(motions[..., :-4]).detach()
        m_lens = m_lens // self.unit_length
        motion_embedding = self.motion_enc(movements, m_lens)
        '''Text Encoding'''
        text_embedding = self.text_enc(word_embs, pos_ohot, cap_lens)
        text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding
    
    @torch.no_grad()
    def get_text_motion_embeddings(self, batch):
        """Calculate the text embedding and motion embedding.
        This function is called when evaluating metrics on GT.
        """
        word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
        word_embs = word_embeddings.detach().to(self.device).float()
        pos_ohot = pos_one_hots.detach().to(self.device).float()
        motions = motions.detach().to(self.device).float()
        
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]
        
        """Movement Encoding"""
        movements = self.movement_enc(motions[..., :-4]).detach()
        m_lens = m_lens // self.unit_length
        motion_embedding = self.motion_enc(movements, m_lens)
            
        """Text Encoding"""
        text_embedding = self.text_enc(word_embs, pos_ohot, sent_lens)
        text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding
    
    @staticmethod
    def get_metric_statistics(values, replication_times):
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        conf_interval = 1.96 * std / np.sqrt(replication_times)
        return mean, conf_interval
    
    def euclidean_distance_matrix(self, matrix1, matrix2):
        """
        :param matrix1: [N1, D]
        :param matrix2: [N2, D]
        """
        assert matrix1.shape[1] == matrix2.shape[1]
        d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
        d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
        d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
        dists = np.sqrt(d1 + d2 + d3)  # broadcasting
        return dists
    
    def calculate_top_k(self, mat, top_k):
        size = mat.shape[0]
        gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
        bool_mat = (mat == gt_mat)
        correct_vec = False
        top_k_list = []
        for i in range(top_k):
            correct_vec = (correct_vec | bool_mat[:, i])
            top_k_list.append(correct_vec[:, None])
        top_k_mat = np.concatenate(top_k_list, axis=1)
        return top_k_mat
    
    def run_pred(self):
        
        all_metrics = OrderedDict({
            'Matching Score': OrderedDict({}),
            'R_precision': OrderedDict({}),
            'FID': OrderedDict({}),
            'Diversity': OrderedDict({}),
            'MultiModality': OrderedDict({})}
        )
        for tid, batches in self.batches.items():
                        
            for _ in range(self.replications):
            
                match_score_dict = OrderedDict({})
                R_precision_dict = OrderedDict({})
                activation_dict = OrderedDict({})

                random.shuffle(batches)
                all_motion_embeddings = []
                score_list = []
                all_size = 0
                matching_score_sum = 0
                top_k_count = 0
                for i in range(0, len(batches), 32):
                    batch = batches[i:i+32]
                    
                    if len(batch) < 32: 
                        continue
                    
                    # Get batch
                    batch = self.prepare_batch(inp_batch=batch)
                    
                    text_embeddings, motion_embeddings = self.get_co_embeddings(batches=batch)
                    dist_mat = self.euclidean_distance_matrix(
                        text_embeddings.cpu().numpy(),
                        motion_embeddings.cpu().numpy())
                    matching_score_sum += dist_mat.trace()
                    argsmax = np.argsort(dist_mat, axis=1)
                    # print(len(batch), "|", argsmax.shape)
                    # exit(0)
                    top_k_mat = self.calculate_top_k(argsmax, top_k=3)
                    top_k_count += top_k_mat.sum(axis=0)

                    all_size += text_embeddings.shape[0]

                    all_motion_embeddings.append(motion_embeddings.cpu().numpy())

                all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
                matching_score = matching_score_sum / all_size
                R_precision = top_k_count / all_size
                match_score_dict["pred"] = matching_score
                R_precision_dict["pred"] = R_precision
                activation_dict["pred"] = all_motion_embeddings

                print(f'---> Matching Score: {matching_score:.4f}')

                line = f'---> R_precision: '
                for i in range(len(R_precision)):
                    line += '(top %d): %.4f ' % (i+1, R_precision[i])
                print(line)

                for key, item in match_score_dict.items():
                    if key not in all_metrics["Matching Score"]:
                        all_metrics["Matching Score"][key] = [item]
                    else:
                        all_metrics["Matching Score"][key] += [item]

                for key, item in R_precision_dict.items():
                    if key not in all_metrics["R_precision"]:
                        all_metrics['R_precision'][key] = [item]
                    else:
                        all_metrics['R_precision'][key] += [item]
        
        for metric_name, metric_dict in all_metrics.items():
            for model_name, values in metric_dict.items():
                mean, conf_interval = self.get_metric_statistics(np.array(values), replication_times=len(self.batches))
                
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)

    def run_gt(self):
        
        # Build GT dataloader
        mean = np.load("tools/mean.npy")
        std = np.load("tools/std.npy")
        dataset = GTDataset(mean=mean, std=std)
        dataloader = DataLoader(dataset=dataset, batch_size=32, num_workers=4, drop_last=True, 
                                collate_fn=collate_fn, shuffle=True)
        
        match_score_dict = OrderedDict({})
        R_precision_dict = OrderedDict({})
        activation_dict = OrderedDict({})
        
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        for idx, batch in enumerate(dataloader):
            text_embeddings, motion_embeddings = self.get_text_motion_embeddings(batch=batch)
            dist_mat = self.euclidean_distance_matrix(
                text_embeddings.cpu().numpy(),
                motion_embeddings.cpu().numpy())
            matching_score_sum += dist_mat.trace()
            argsmax = np.argsort(dist_mat, axis=1)
            top_k_mat = self.calculate_top_k(argsmax, top_k=3)
            top_k_count += top_k_mat.sum(axis=0)
            
            all_size += text_embeddings.shape[0]
            
            all_motion_embeddings.append(motion_embeddings.cpu().numpy())
        
        all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
        matching_score = matching_score_sum / all_size
        R_precision = top_k_count / all_size
        match_score_dict["RPrecision"] = matching_score
        R_precision_dict["RPrecision"] = R_precision
        activation_dict["RPrecision"] = all_motion_embeddings
        
        print(f'---> Matching Score: {matching_score:.4f}')
        
        line = f'---> R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)

    def run(self):
        # self.run_gt()
        self.run_pred()

class FeatureMetrics(object):
    def __init__(self, args):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.replications = args.replications
        self.motion_dir = args.motion_dir
        self.mean = np.load(args.mean_dir)
        self.std = np.load(args.std_dir)
        self.unit_length = 4
        self.min_motion_length = 40
        self.max_motion_length = 196
        self.movement_enc = MovementConvEncoder(input_size=263-4, hidden_size=512, output_size=512).to(self.device)
        self.text_enc = TextEncoderBiGRUCo(
            word_size=300,
            pos_size=15,
            hidden_size=512,
            output_size=512,
            device="cpu").to(self.device)
        self.motion_enc = MotionEncoderBiGRUCo(
            input_size=512,
            hidden_size=1024,
            output_size=512,
            device="cpu").to(self.device)
        
        checkpoint = torch.load(
            "networks/evaluation/pretrained/text_mot_match/model/finest.tar", 
            map_location=torch.device("cpu"))
        self.movement_enc.load_state_dict(checkpoint['movement_encoder'])
        self.text_enc.load_state_dict(checkpoint['text_encoder'])
        self.motion_enc.load_state_dict(checkpoint['motion_encoder'])
        print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
        
        self.movement_enc.eval()
        self.text_enc.eval()
        self.motion_enc.eval()
        
    def prepare_one(self, pose, m_length):
        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'
            
        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        pose = pose[:m_length]
        
        """ Z Normalization """
        pose = (pose - self.mean) / self.std
        
        if m_length < self.max_motion_length:
            pose = np.concatenate([pose, np.zeros((self.max_motion_length-m_length, pose.shape[1]))], axis=0)
        elif m_length > self.max_motion_length:
            pose = pose[:self.max_motion_length]
            
        return pose, m_length
        
    def prepare(self):
        self.batches = {}
        self.files = [f for f in os.listdir(self.motion_dir) if ".npy" in f]
        self.tids = []
        for file in tqdm(self.files, desc="Preparing batches"):
            # Parse the input file to get the tid
            base_name = file.split(".")[0]
            tid = base_name.split("_")[1]   # sample id
            bid = base_name.split("_")[0]   # batch id
            
            # if int(bid[1:]) % 10 != 0: 
            #     continue
            
            self.tids.append(tid)
            
            data = np.load(os.path.join(self.motion_dir, file), allow_pickle=True).item()
            
            try:
                gt_motion = data["gt"]["body"]
                rc_motion = data["pred"]["body"]
            except:
                gt_motion = data["gt"]
                rc_motion = data["motion"]
            gt_poses = gt_motion[0].transpose(1, 0)  # [T, 263]    
            rc_poses = rc_motion[0].transpose(1, 0)  # [T, 263]
            
            min_len = min(gt_poses.shape[0], rc_poses.shape[0])
            rc_len = rc_poses.shape[0]
            gt_len = gt_poses.shape[0]
            
            if rc_len < self.min_motion_length or rc_len > 200:
                continue
            if gt_len < self.min_motion_length or gt_len > 200:
                continue
            
            # if rc_len > 200:
            #     indices = np.random.choice(rc_len, 200, replace=False)
            #     indices = np.sort(indices)
            #     rc_poses = rc_poses[indices]
            #     rc_len = 200
            # if gt_len > 200:
            #     indices = np.random.choice(gt_len, 200, replace=False)
            #     indices = np.sort(indices)
            #     gt_poses = gt_poses[indices]
            #     gt_len = 200
            
            gt_poses, gt_len = self.prepare_one(gt_poses, gt_len)
            rc_poses, rc_len = self.prepare_one(rc_poses, rc_len)
            
            if gt_len < self.min_motion_length or rc_len < self.min_motion_length: 
                continue
            
            batch = {
                "rc_pose": rc_poses, 
                "gt_pose": gt_poses, 
                "rc_len": rc_len, 
                "gt_len": gt_len, 
                "bid": bid
            }
            if tid in self.batches.keys():
                self.batches[tid].append(batch)
            else:
                self.batches[tid] = [batch]

        self.tids = list(set(self.tids))
        
    @torch.no_grad()
    def calc_one_batch(self, motions, lengths):
        align_idx = np.argsort(lengths)[::-1].copy()
        motions = np.stack(motions, axis=0)
        motions = motions[align_idx]
        m_lens = np.asarray(lengths)
        m_lens = m_lens[align_idx]
        
        motions = torch.from_numpy(motions).float().to(self.device)
        m_lens = torch.from_numpy(m_lens).long().to(self.device)
        '''Movement Encoding'''
        movements = self.movement_enc(motions[..., :-4]).detach()
        m_lens = m_lens // self.unit_length
        motion_embedding = self.motion_enc(movements, m_lens)
        
        valid_motion_embedding = []
        for emb in motion_embedding:
            if torch.isnan(emb).sum() > 0:
                pass
            else:
                valid_motion_embedding.append(emb)
        return torch.stack(valid_motion_embedding, dim=0)
    
    @torch.no_grad()
    def calc_embeddings(self):
        self.motion_embeddings = {"gt": {}, "rc": {}, "bid": {}}
        for tid, batches in self.batches.items():
            all_gt_embeddings = []
            all_rc_embeddings = []
            all_bids = []
            for i in range(0, len(batches), 32):
                batch = batches[i:i+32]
                # if len(batch) < 32:
                #     continue
                
                gt_motions = [b["gt_pose"] for b in batch]
                gt_lenghts = [b["gt_len"] for b in batch]
                rc_motions = [b["rc_pose"] for b in batch]
                rc_lengths = [b["rc_len"] for b in batch]
                all_bids += [b["bid"] for b in batch]
                
                gt_embeddings = self.calc_one_batch(gt_motions, gt_lenghts)
                rc_embeddings = self.calc_one_batch(rc_motions, rc_lengths)
                
                # gt_embeddings = F.normalize(gt_embeddings, dim=-1)
                # rc_embeddings = F.normalize(rc_embeddings, dim=-1)
                
                all_gt_embeddings.append(gt_embeddings.data.cpu().numpy())
                all_rc_embeddings.append(rc_embeddings.data.cpu().numpy())
            
            all_gt_embeddings = np.concatenate(all_gt_embeddings, axis=0)
            all_rc_embeddings = np.concatenate(all_rc_embeddings, axis=0)
            self.motion_embeddings["gt"][tid] = all_gt_embeddings
            self.motion_embeddings["rc"][tid] = all_rc_embeddings
            self.motion_embeddings["bid"][tid] = all_bids
    
    @staticmethod
    def calculate_activation_statistics(activations):
        """
        Params:
        -- activation: num_samples x dim_feat
        Returns:
        -- mu: dim_feat
        -- sigma: dim_feat x dim_feat
        """
        mu = np.mean(activations, axis=0)
        cov = np.cov(activations, rowvar=False)
        return mu, cov

    def calc_frechet_distance(self):
        tid_list = list(self.motion_embeddings["gt"].keys())
        gt_motion_embeddinsg = self.motion_embeddings["gt"][tid_list[0]]
        gt_mu, gt_cov = self.calculate_activation_statistics(gt_motion_embeddinsg)
        
        fid_scores = []
        for rc_motion_embedding in self.motion_embeddings["rc"].values():
            rc_mu, rc_cov = self.calculate_activation_statistics(rc_motion_embedding)
            fid = calculate_frechet_distance(gt_mu, gt_cov, rc_mu, rc_cov)
            fid_scores.append(fid)
        return fid_scores
    
    def calc_diversity(self):
        div_scores = []
        for rc_motion_embedding in self.motion_embeddings["rc"].values():
            div = calculate_diversity(rc_motion_embedding, diversity_times=300)
            div_scores.append(div)
        return div_scores
    
    def calc_multimodality(self):
        mm_scores = []
        tids = list(self.motion_embeddings["rc"].keys())
        # Re-organize the embeddings
        embeddings_reorg = {}
        for tid in tids:
            rc_motion_embedding = self.motion_embeddings["rc"][tid]
            bids = self.motion_embeddings["bid"][tid]
            for (embed, bid) in zip(rc_motion_embedding, bids):
                if bid in embeddings_reorg.keys():
                    embeddings_reorg[bid].append(embed)
                else:
                    embeddings_reorg[bid] = [embed]
        
        # for bid, embeds in embeddings_reorg.items():
        #     if len(embeds) != 5:
        #         print(bid)
        
        min_num_samples = 10000
        for bid, embeds in embeddings_reorg.items():
            if len(embeds) < len(self.tids): continue
            min_num_samples = min(min_num_samples, len(embeds))
            embeddings_reorg[bid] = np.stack(embeds, axis=0)    # [num_samples, num_dim]
        print('-----', min_num_samples)
        # Calculate multimodality
        for _ in range(self.replications):
            embeddings_cat = []
            for _, embeds in embeddings_reorg.items():
                if len(embeds) < min_num_samples:
                    continue
                elif len(embeds) == min_num_samples:
                    i = 0
                else:
                    i = random.randint(0, len(embeds) - min_num_samples)
                embeddings_cat.append(embeds[i:i+min_num_samples])
            embeddings_cat = np.stack(embeddings_cat, axis=0)   # [num_batches, num_samples, num_dim]
        
            multimodality = calculate_multimodality(embeddings_cat, multimodality_times=min(min_num_samples, len(self.tids)))
            # multimodality = calculate_multimodality(embeddings_cat, multimodality_times=min(3, len(self.tids)))
            mm_scores.append(multimodality)
       
        return mm_scores
    
    def run(self):
        # Prepare the data batches
        self.prepare()
        # Calculate the embeddings
        self.calc_embeddings()
        # Calculate FID
        fid_scores = self.calc_frechet_distance()
        # Calculate Div
        div_scores = self.calc_diversity()
        # Calculate MM
        if len(self.tids) > 1:
            mm_scores = self.calc_multimodality()
            
        # FID
        fid_mean, fid_conf = RPrecision.get_metric_statistics(fid_scores, replication_times=len(fid_scores))
        print("FID: Mean: {:.4f}, CInf: {:.4f}".format(fid_mean, fid_conf))
        # Div
        div_mean, div_conf = RPrecision.get_metric_statistics(div_scores, replication_times=len(div_scores))
        print("DIV: Mean: {:.4f}, CInf: {:.4f}".format(div_mean, div_conf))
        # MM
        if len(self.tids) > 1:
            mm_mean, mm_conf = RPrecision.get_metric_statistics(mm_scores, replication_times=len(mm_scores))
            print("MM: Mean: {:.4f}, CInf: {:.4f}".format(mm_mean, mm_conf))
        
class PlanningEvaluator(object):
    def __init__(self, args):
        self.planning_task = args.planning_task.split(",")
        self.planning_dir = args.planning_dir
        self.planning_file = os.path.join(args.planning_dir, args.planning_task, "result.json")
        self.task_mapping = {
            # "ct2t": "CT2T", 
            # "cs2s": "CS2S", 
            "ct2s": "CT2S", 
            # "cs2t": "CS2T", 
            # "t2c": "T2C", 
            # "s2c": "S2C",
            # "t2s": "T2S", 
            # "s2t": "S2T"
        }
        
    def run_one_task(self, task):
        planning_file = os.path.join(self.planning_dir, task, "result.json")
        if not os.path.exists(planning_file):
            return
        with open(planning_file, "r") as f:
            planning_data = json.load(f)
        random.shuffle(planning_data)
        total_score = 0.0
        total_cnt = 0
        for i in tqdm(range(0, len(planning_data), 1), desc="Evaluating {:s} task".format(self.task_mapping[task])):
            inp_batch = planning_data[i]["data"]
            # Convert list[str] to str
            for key, val in inp_batch.items():
                if isinstance(val, list):
                    inp_batch[key] = val[0]
            score = FUNC_MAP[task](**inp_batch)
            if score != -1:
                total_score += score
                total_cnt += 1
        return total_score / total_cnt
    
    def run(self):
        scores = {}
        for task in self.task_mapping.keys():
            score = self.run_one_task(task=task)
            scores[task] = score
        
        for task, score in scores.items():
            print("Task {:s} | score: {:.5f}".format(self.task_mapping[task], score))

        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="PlanningMetrics", help="")
    parser.add_argument('--motion_dir', type=str, 
                        default="logs/avatar_gpt/eval/flan_t5_large/exp9/output/m2m", 
                        help="")
    parser.add_argument('--planning_dir', type=str, 
                        # default="logs/avatar_gpt/eval/flan_t5_large/exp9/postprocessed", 
                        default="test", 
                        # default="logs/avatar_gpt/eval/gpt_large/exp9/postprocessed", 
                        help="")
    parser.add_argument('--planning_task', type=str, default="ct2t,cs2s,ct2s,cs2t,t2c,s2c,t2s,s2t", help="")
    parser.add_argument('--motion_repr', type=str, default="d263", help="motion representation, choose from 1) d75, and 2) d263")
    parser.add_argument('--text_dir', type=str, 
                        default="../dataset/HumanML3D/texts/", 
                        help="")
    parser.add_argument('--split_dir', type=str, 
                        default="../dataset/HumanML3D/test.txt", 
                        help="")
    parser.add_argument('--mean_dir', type=str, 
                        default="tools/mean.npy", 
                        help="")
    parser.add_argument('--std_dir', type=str, 
                        default="tools/std.npy", 
                        help="")
    parser.add_argument('--preprocessed_motion_dir', type=str, 
                        default="data/preprocessed_gt_motions.pickle", 
                        help="")
    parser.add_argument('--replications', type=int, default=20, help="")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    
    if args.task == "TextPreprocessor":
        Agent = TextPreprocessor(args=args)
    elif args.task == "GTMotionPreprocessor":
        Agent = GTMotionPreprocessor(args=args)
    elif args.task == "PredMotionPreprocessor":
        Agent = PredMotionPreprocessor(arg=args)
    elif args.task == "RPrecision":
        Agent = RPrecision(args=args)
    elif args.task == "FeatureMetrics":
        Agent = FeatureMetrics(args=args)
    elif args.task == "PlanningMetrics":
        Agent = PlanningEvaluator(args=args)
    
    Agent.run()
    

