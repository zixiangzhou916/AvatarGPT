from typing import Optional, Dict, Union
import os
import os.path as osp

import pickle
from typing import NewType, Union, Optional
from dataclasses import dataclass, asdict, fields
import numpy as np

import torch
import torch.nn as nn

from .body_models import *
from .lbs_v2 import lbs

Tensor = NewType('Tensor', torch.Tensor)
Array = NewType('Array', np.ndarray)

RIGHT_HAND_JOINTS_HIERARCHCY = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 8, 5, 10, 11, 12, 13, 0, 15, 16, 17, 18, 15, 20, 21, 22]
LEFT_HAND_JOINTS_HIERARCHCY = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 8, 5, 10, 11, 12, 13, 0, 15, 16, 17, 18, 15, 20, 21, 22]

# RIGHT_HAND_ROOT_HIERARCHY = 19
# LEFT_HAND_ROOT_HIERARCHY = 18
RIGHT_HAND_ROOT_HIERARCHY = 21
LEFT_HAND_ROOT_HIERARCHY = 20

@dataclass
class ModelOutput:
    vertices: Optional[Tensor] = None
    joints: Optional[Tensor] = None
    left_joints: Optional[Tensor] = None
    right_joints: Optional[Tensor] = None
    full_pose: Optional[Tensor] = None
    global_orient: Optional[Tensor] = None
    transl: Optional[Tensor] = None
    v_shaped: Optional[Tensor] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)

@dataclass
class SMPLOutput(ModelOutput):
    betas: Optional[Tensor] = None
    body_pose: Optional[Tensor] = None

class SMPLx(SMPL):
    
    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    SHAPE_SPACE_DIM = 300
    
    def __init__(self, 
                 model_path: str, 
                 kid_template_path: str = '', 
                 data_struct: Optional[Struct] = None, 
                 create_betas: bool = True, 
                 betas: Optional[Tensor] = None, 
                 num_betas: int = 10, 
                 create_global_orient: bool = True, 
                 global_orient: Optional[Tensor] = None, 
                 create_body_pose: bool = True, 
                 body_pose: Optional[Tensor] = None, 
                 create_transl: bool = True, 
                 transl: Optional[Tensor] = None, 
                 dtype=torch.float32, 
                 batch_size: int = 1, 
                 joint_mapper=None, 
                 gender: str = 'neutral', 
                 age: str = 'adult', 
                 vertex_ids: Dict[str, int] = None, 
                 v_template: Optional[Union[Tensor, Array]] = None, 
                 **kwargs):
        super().__init__(
            model_path, 
            kid_template_path, 
            data_struct, 
            create_betas, 
            betas, 
            num_betas, 
            create_global_orient, 
            global_orient, 
            create_body_pose, 
            body_pose, 
            create_transl, 
            transl, 
            dtype, 
            batch_size, 
            joint_mapper, 
            gender, 
            age, 
            vertex_ids, 
            v_template, 
            **kwargs)
        
        left_hand_parents = to_tensor(to_np(LEFT_HAND_JOINTS_HIERARCHCY)).long()
        self.register_buffer('left_hand_parents', left_hand_parents)
        right_hand_parents = to_tensor(to_np(RIGHT_HAND_JOINTS_HIERARCHCY)).long()
        self.register_buffer('right_hand_parents', right_hand_parents)
        left_hand_root_parent = to_tensor(to_np(LEFT_HAND_ROOT_HIERARCHY)).long()
        self.register_buffer('left_hand_root_parent', left_hand_root_parent)
        right_hand_root_parent = to_tensor(to_np(RIGHT_HAND_ROOT_HIERARCHY)).long()
        self.register_buffer('right_hand_root_parent', right_hand_root_parent)
        
    def forward(
        self, 
        betas: Optional[Tensor] = None, 
        body_pose: Optional[Tensor] = None, 
        left_hand_pose: Optional[Tensor] = None, 
        right_hand_pose: Optional[Tensor] = None, 
        global_orient: Optional[Tensor] = None, 
        transl: Optional[Tensor] = None, 
        return_verts=True, 
        return_full_pose: bool = False, 
        pose2rot: bool = True, 
        **kwargs):
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
        '''
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None and hasattr(self, 'transl'):
            transl = self.transl

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        batch_size = max(betas.shape[0], global_orient.shape[0],
                         body_pose.shape[0])

        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = betas.expand(num_repeats, -1)

        joints, left_joints, right_joints, vertices = \
            lbs(betas, 
                full_pose, 
                left_hand_pose, 
                right_hand_pose, 
                self.v_template, 
                self.shapedirs, 
                self.posedirs, 
                self.J_regressor, 
                self.parents, 
                self.left_hand_parents, 
                self.left_hand_root_parent, 
                self.right_hand_parents, 
                self.right_hand_root_parent, 
                self.lbs_weights, 
                pose2rot=pose2rot)
                
        joints = self.vertex_joint_selector(vertices, joints)
        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)
        
        # print(left_joints[0, :, 0])
        left_offsets = left_joints[:, 0].unsqueeze(dim=1).repeat(1, 24, 1)
        left_joints -= left_offsets
        # print(left_joints[0, :, 0])
        right_offset = right_joints[:, 0].unsqueeze(dim=1).repeat(1, 24, 1)
        right_joints -= right_offset
        left_offsets = joints[:, self.left_hand_root_parent].unsqueeze(dim=1)
        right_offsets = joints[:, self.right_hand_root_parent].unsqueeze(dim=1)
        left_joints += left_offsets
        right_joints += right_offsets

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            left_joints += transl.unsqueeze(dim=1)
            right_joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = SMPLOutput(vertices=vertices if return_verts else None,
                            global_orient=global_orient,
                            body_pose=body_pose,
                            joints=joints,
                            left_joints=left_joints, 
                            right_joints=right_joints,
                            betas=betas,
                            full_pose=full_pose if return_full_pose else None)

        return output