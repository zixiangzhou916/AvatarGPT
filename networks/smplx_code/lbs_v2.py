# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from typing import Tuple, List
import numpy as np

import torch
import torch.nn.functional as F

from .utils import rot_mat_to_euler, Tensor
from .lbs import *

def Rx(angs, in_radians=False):
    dim = angs.dim()
    if not in_radians:
        angs = np.radians(angs)
        
    if angs.dim() == 2:
        n, j = angs.shape[:2]
        angs = np.reshape(angs, (-1))
    
    nframes = angs.shape[0]
    R = np.zeros((nframes, 3, 3))
    for i, ang in enumerate(angs):
        R[i] = np.array([
            [1, 0, 0], 
            [0, np.cos(ang), -1*np.sin(ang)], 
            [0, np.sin(ang), np.cos(ang)]
        ])
        
    return R if dim == 1 else np.reshape(R, (n, j, 3, 3))

def Ry(angs, in_radians=False):
    dim = angs.dim()
    if not in_radians:
        angs = np.radians(angs)
    
    if angs.dim() == 2:
        n, j = angs.shape[:2]
        angs = np.reshape(angs, (-1))
    
    nframes = angs.shape[0]
    R = np.zeros((nframes, 3, 3))
    for i, ang in enumerate(angs):
        R[i] = np.array([
            [np.cos(ang), 0, np.sin(ang)],
            [0, 1, 0],
            [-1*np.sin(ang), 0, np.cos(ang)]
        ])
    return R if dim == 1 else np.reshape(R, (n, j, 3, 3))

def Rz(angs, in_radians=False):
    dim = angs.dim()
    if not in_radians:
        angs = np.radians(angs)
        
    if angs.dim() == 2:
        n, j = angs.shape[:2]
        angs = np.reshape(angs, (-1))
    
    nframes = angs.shape[0]
    R = np.zeros((nframes, 3, 3))
    for i, ang in enumerate(angs):
        R[i] = np.array([
            [np.cos(ang), -1*np.sin(ang), 0],
            [np.sin(ang), np.cos(ang), 0],
            [0, 0, 1]
        ])
    return R if dim == 1 else np.reshape(R, (n, j, 3, 3))

def angle_to_rotation(angs):
    Rot = np.eye(3)[None, None]
    Rot = np.tile(Rot, (angs.shape[0], angs.shape[1], 1, 1))
    rx = Rx(angs=angs[..., 0].cpu())
    ry = Ry(angs=angs[..., 1].cpu())
    rz = Rz(angs=angs[..., 2].cpu())
    # print(angs[0, 0, 0], "|", rx[0, 0])
    # print('=' * 20)
    # print(angs[0, 0, 1], "|", ry[0, 0])
    # print('=' * 20)
    # print(angs[0, 0, 2], "|", rz[0, 0])
    # print('=' * 20)
    Rot = np.matmul(Rot, rx)
    Rot = np.matmul(Rot, ry)
    Rot = np.matmul(Rot, rz)
    # print(Rot[0, 0])
    # exit(0)
    return torch.from_numpy(Rot).float()

def batch_rigid_transform_v2(
    rot_mats: Tensor,
    joints: Tensor,
    parents: Tensor, 
    left_hand_pose: Tensor, 
    left_hand_parents: Tensor, 
    left_hand_root_parent: Tensor, 
    right_hand_pose: Tensor, 
    right_hand_parents: Tensor, 
    right_hand_root_parent: Tensor, 
    dtype=torch.float32
):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    # 1. transform body
    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]    # list of [B, 4, 4] transformation matrixs
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)
        
    transforms = torch.stack(transform_chain, dim=1)    # articulated transformation matrixs, [B, 24, 4, 4]
    
    # 2. transform left hand
    
    # for i in [0, 250, 500, 750]:
    #     print(left_hand_pose[i, 0, 3:].data.cpu().numpy())
        
    # left_hand_pose[:, 0, 3:] *= 0.0
    # left_hand_pose[:, 0, 4] = 0
    left_rot_mats = angle_to_rotation(left_hand_pose[:, :, 3:]).to(left_hand_pose.device)

    # print('-' * 50)
    # for i in [0, 250, 500, 750]:
    #     print(left_hand_pose[i, 0, 3:].data.cpu().numpy())
    # print('-' * 50)
    
    # for i in [0, 250, 500, 750]:
    #     print('left', '-' * 20, i, '-' * 20)
    #     print(transform_chain[left_hand_root_parent][i].data.cpu().numpy())
    #     print('right', '-' * 20, i, '-' * 20)
    #     print(transform_chain[right_hand_root_parent][i].data.cpu().numpy())
        
    left_transforms_mat = transform_mat(
        left_rot_mats.reshape(-1, 3, 3), 
        left_hand_pose[:, :, :3].reshape(-1, 3, 1)).reshape(-1, left_hand_pose.shape[1], 4, 4)
    
    # left_transform_chain = [left_transforms_mat[:, 0]]      # exp01
    left_transform_chain = [torch.matmul(transform_chain[left_hand_root_parent], left_transforms_mat[:, 0])]
    for i in range(1, left_hand_parents.shape[0]):
        curr_res = torch.matmul(left_transform_chain[left_hand_parents[i]], left_transforms_mat[:, i])
        left_transform_chain.append(curr_res)
    left_transforms = torch.stack(left_transform_chain, dim=1)
    
    # 3. transform right hand
    # right_hand_pose[:, 0, 3:] *= 0.0
    # right_hand_pose[:, 0, 4] = 0
    right_rot_mats = angle_to_rotation(right_hand_pose[:, :, 3:]).to(right_hand_pose.device)
    right_transforms_mat = transform_mat(
        right_rot_mats.reshape(-1, 3, 3), 
        right_hand_pose[:, :, :3].reshape(-1, 3, 1)).reshape(-1, right_hand_pose.shape[1], 4, 4)
    
    # right_transform_chain = [right_transforms_mat[:, 0]]    # exp01
    right_transform_chain = [torch.matmul(transform_chain[right_hand_root_parent], right_transforms_mat[:, 0])]
    for i in range(1, right_hand_parents.shape[0]):
        curr_res = torch.matmul(right_transform_chain[right_hand_parents[i]], right_transforms_mat[:, i])
        right_transform_chain.append(curr_res)
    right_transforms = torch.stack(right_transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]
    left_posed_joints = left_transforms[:, :, :3, 3]
    right_posed_joints = right_transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, left_posed_joints, right_posed_joints, rel_transforms

def lbs(
    betas: Tensor,
    pose: Tensor, 
    left_hand_pose: Tensor, 
    right_hand_pose: Tensor, 
    v_template: Tensor,
    shapedirs: Tensor,
    posedirs: Tensor,
    J_regressor: Tensor,
    parents: Tensor, 
    left_hand_parents: Tensor, 
    left_hand_root_parent: Tensor, 
    right_hand_parents: Tensor, 
    right_hand_root_parent: Tensor, 
    lbs_weights: Tensor,
    pose2rot: bool = True,
):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(
            pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, left_J_transformed, right_J_transformed, A = \
        batch_rigid_transform_v2(rot_mats=rot_mats, 
                                 joints=J, 
                                 parents=parents, 
                                 left_hand_pose=left_hand_pose, 
                                 left_hand_parents=left_hand_parents, 
                                 left_hand_root_parent=left_hand_root_parent, 
                                 right_hand_pose=right_hand_pose, 
                                 right_hand_parents=right_hand_parents, 
                                 right_hand_root_parent=right_hand_root_parent, 
                                 dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return J_transformed, left_J_transformed, right_J_transformed, verts
