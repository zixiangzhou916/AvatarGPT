import os, sys, argparse
sys.path.append(os.getcwd())
import colorsys

import os, sys, argparse
sys.path.append(os.getcwd())
import glob
try:
    from omegaconf import OmegaConf
except:
    print('---- Unable to import OmegaConf')
import yaml
import random
import pickle
from tqdm import tqdm
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F
from networks import smplx_code
from networks.flame.DecaFLAME import FLAME

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.io import imread
import imageio
from render.ude_v2 import util
from torchvision import transforms
from render.ude_v2.pytorch3d_utils.obj_io import save_obj

from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, load_objs_as_meshes
# from pytorch3d.io.obj_io_new import save_obj
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex, 
    # TexturedPhongShader
)

try:
    with open("networks/flame/cfg.yaml", "r") as f:
        flame_cfg = OmegaConf.load(f)
    flame_cfg = flame_cfg.coarse.model
except:
    pass

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
   
def convert_flame2joints(
    flame_model, expr_pose=None, neck_pose=None, jaw_pose=None, leye_pose=None, reye_pose=None, **kwargs
):
    """
    :param expression: [batch_size, seq_len, 100]
    :param neck_pose: [batch_size, seq_len, 3]
    :param jaw_pose: [batch_size, seq_len, 3]
    :param leye_pose: [batch_size, seq_len, 3]
    :param reye_pose: [batch_size, seq_len, 3]
    """
    B, T, _ = expr_pose.shape
    shapes = torch.zeros(B*T, 300).float().to(expr_pose.device)
    # pose = torch.zeros(B*T, 3).float().to(expr_pose.device)
    pose = neck_pose.reshape(B*T, -1) if neck_pose is not None else torch.zeros(B*T, 3).float().to(expr_pose.device)
    pose = torch.cat([pose, jaw_pose.reshape(B*T, -1)], dim=-1)
    eye_pose = torch.cat([leye_pose, reye_pose], dim=-1)
    
    results = flame_model(shapes, 
                          expr_pose.reshape(B*T, -1), 
                          pose, 
                          eye_pose.reshape(B*T, -1))
    vertices = results[0].reshape(B, T, -1, 3)
    lmk_2d = results[1].reshape(B, T, -1, 3)
    lmk_3d = results[2].reshape(B, T, -1, 3)
    
    return {"vertices": vertices, "joints": lmk_3d}

def set_rasterizer(type = 'pytorch3d'):
    if type == 'pytorch3d':
        global Meshes, load_obj, rasterize_meshes
        from pytorch3d.structures import Meshes
        from pytorch3d.io import load_obj
        from pytorch3d.renderer.mesh import rasterize_meshes
    elif type == 'standard':
        global standard_rasterize, load_obj
        import os
        from .util import load_obj
        # Use JIT Compiling Extensions
        # ref: https://pytorch.org/tutorials/advanced/cpp_extension.html
        from torch.utils.cpp_extension import load, CUDA_HOME
        curr_dir = os.path.dirname(__file__)
        standard_rasterize_cuda = \
            load(name='standard_rasterize_cuda', 
                sources=[f'{curr_dir}/rasterizer/standard_rasterize_cuda.cpp', f'{curr_dir}/rasterizer/standard_rasterize_cuda_kernel.cu'], 
                extra_cuda_cflags = ['-std=c++14', '-ccbin=$$(which gcc-7)']) # cuda10.2 is not compatible with gcc9. Specify gcc 7 
        from standard_rasterize_cuda import standard_rasterize
        # If JIT does not work, try manually installation first
        # 1. see instruction here: pixielib/utils/rasterizer/INSTALL.md
        # 2. add this: "from .rasterizer.standard_rasterize_cuda import standard_rasterize" here

def transform_points(self, points, tform, points_scale=None, normalize = True):
    points_2d = points[:,:,:2]
        
    #'input points must use original range'
    if points_scale:
        assert points_scale[0]==points_scale[1]
        points_2d = (points_2d*0.5 + 0.5)*points_scale[0]

    batch_size, n_points, _ = points.shape
    trans_points_2d = torch.bmm(
                    torch.cat([points_2d, torch.ones([batch_size, n_points, 1], device=points.device, dtype=points.dtype)], dim=-1), 
                    tform
                    ) 
    trans_points = torch.cat([trans_points_2d[:,:,:2], points[:,:,2:]], dim=-1)
    if normalize:
        trans_points[:,:,:2] = trans_points[:,:,:2]/self.crop_size*2 - 1
    return trans_points
    
class StandardRasterizer(nn.Module):
    """ Alg: https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation
    Notice:
        x,y,z are in image space, normalized to [-1, 1]
        can render non-squared image
        not differentiable
    """
    def __init__(self, height, width=None):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        if width is None:
            width = height
        self.h = h = height; self.w = w = width

    def forward(self, vertices, faces, attributes=None, h=None, w=None):
        device = vertices.device
        if h is None:
            h = self.h
        if w is None:
            w = self.h; 
        bz = vertices.shape[0]
        depth_buffer = torch.zeros([bz, h, w]).float().to(device) + 1e6
        triangle_buffer = torch.zeros([bz, h, w]).int().to(device) - 1
        baryw_buffer = torch.zeros([bz, h, w, 3]).float().to(device)
        vert_vis = torch.zeros([bz, vertices.shape[1]]).float().to(device)
        vertices = vertices.clone().float()
        # compatibale with pytorch3d ndc, see https://github.com/facebookresearch/pytorch3d/blob/e42b0c4f704fa0f5e262f370dccac537b5edf2b1/pytorch3d/csrc/rasterize_meshes/rasterize_meshes.cu#L232
        vertices[...,:2] = -vertices[...,:2]
        vertices[...,0] = vertices[..., 0]*w/2 + w/2
        vertices[...,1] = vertices[..., 1]*h/2 + h/2
        vertices[...,0] = w - 1 - vertices[..., 0]
        vertices[...,1] = h - 1 - vertices[..., 1]
        vertices[...,0] = -1 + (2*vertices[...,0] + 1)/w
        vertices[...,1] = -1 + (2*vertices[...,1] + 1)/h
        #
        vertices = vertices.clone().float()
        vertices[...,0] = vertices[..., 0]*w/2 + w/2 
        vertices[...,1] = vertices[..., 1]*h/2 + h/2 
        vertices[...,2] = vertices[..., 2]*w/2
        f_vs = util.face_vertices(vertices, faces)

        standard_rasterize(f_vs, depth_buffer, triangle_buffer, baryw_buffer, h, w)
        pix_to_face = triangle_buffer[:,:,:,None].long()
        bary_coords = baryw_buffer[:,:,:,None,:]
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone(); attributes = attributes.view(attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:,:,:,0].permute(0,3,1,2)
        pixel_vals = torch.cat([pixel_vals, vismask[:,:,:,0][:,None,:,:]], dim=1)
        return pixel_vals

class Pytorch3dRasterizer(nn.Module):
    ## TODO: add support for rendering non-squared images, since pytorc3d supports this now
    """  Borrowed from https://github.com/facebookresearch/pytorch3d
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin':  None,
            'perspective_correct': False,
        }
        raster_settings = util.dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None, h=None, w=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[...,:2] = -fixed_vertices[...,:2]
        raster_settings = self.raster_settings
        if h is None and w is None:
            image_size = raster_settings.image_size
        else:
            image_size = [h, w]
            if h>w:
                fixed_vertices[..., 1] = fixed_vertices[..., 1]*h/w
            else:
                fixed_vertices[..., 0] = fixed_vertices[..., 0]*w/h
        
        # TODO
        fixed_vertices[..., -2] += 1.0  # control height position
        fixed_vertices[..., -1] += 100.
        fixed_vertices[..., -3] += 0.0  # control width position
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
        
        print(pix_to_face.min(), pix_to_face.max())
        print(zbuf.min(), zbuf.max())
        print(bary_coords.min(), bary_coords.max())
        print(dists.min(), dists.max())
        # exit(0)
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone(); attributes = attributes.view(attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:,:,:,0].permute(0,3,1,2)
        pixel_vals = torch.cat([pixel_vals, vismask[:,:,:,0][:,None,:,:]], dim=1)
        # print(image_size)
        # import ipdb; ipdb.set_trace()
        return pixel_vals

class SRenderY(nn.Module):
    def __init__(self, image_size, obj_filename, uv_size=256, rasterizer_type='pytorch3d'):
        super(SRenderY, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size
        if rasterizer_type == 'pytorch3d':
            self.rasterizer = Pytorch3dRasterizer(image_size)
            self.uv_rasterizer = Pytorch3dRasterizer(uv_size)
            verts, faces, aux = load_obj(obj_filename)
            uvcoords = aux.verts_uvs[None, ...]      # (N, V, 2)
            uvfaces = faces.textures_idx[None, ...] # (N, F, 3)
            faces = faces.verts_idx[None,...]
        elif rasterizer_type == 'standard':
            self.rasterizer = StandardRasterizer(image_size)
            self.uv_rasterizer = StandardRasterizer(uv_size)
            verts, uvcoords, faces, uvfaces = load_obj(obj_filename)
            verts = verts[None, ...]
            uvcoords = uvcoords[None, ...]
            faces = faces[None, ...]
            uvfaces = uvfaces[None, ...]
        else:
            NotImplementedError

        # faces
        dense_triangles = util.generate_triangles(uv_size, uv_size)
        self.register_buffer('dense_faces', torch.from_numpy(dense_triangles).long()[None,:,:])
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)

        # uv coords
        uvcoords = torch.cat([uvcoords, uvcoords[:,:,0:1]*0.+1.], -1) #[bz, ntv, 3]
        uvcoords = uvcoords*2 - 1; uvcoords[...,1] = -uvcoords[...,1]
        face_uvcoords = util.face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)

        # shape colors, for rendering shape overlay
        colors = torch.tensor([180, 180, 180])[None, None, :].repeat(1, faces.max()+1, 1).float()/255.
        face_colors = util.face_vertices(colors, faces)
        self.register_buffer('face_colors', face_colors)

        ## SH factors for lighting
        pi = np.pi
        constant_factor = torch.tensor([1/np.sqrt(4*pi), ((2*pi)/3)*(np.sqrt(3/(4*pi))), ((2*pi)/3)*(np.sqrt(3/(4*pi))),\
                           ((2*pi)/3)*(np.sqrt(3/(4*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))),\
                           (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3/2)*(np.sqrt(5/(12*pi))), (pi/4)*(1/2)*(np.sqrt(5/(4*pi)))]).float()
        self.register_buffer('constant_factor', constant_factor)
    
    def forward(self, vertices, transformed_vertices, albedos, lights=None, h=None, w=None, light_type='point', background=None):
        '''
        -- Texture Rendering
        vertices: [batch_size, V, 3], vertices in world space, for calculating normals, then shading
        transformed_vertices: [batch_size, V, 3], range:normalized to [-1,1], projected vertices in image space (that is aligned to the iamge pixel), for rasterization
        albedos: [batch_size, 3, h, w], uv map
        lights: 
            spherical homarnic: [N, 9(shcoeff), 3(rgb)]
            points/directional lighting: [N, n_lights, 6(xyzrgb)]
        light_type:
            point or directional
        '''
        batch_size = vertices.shape[0]
        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        # transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10
        transformed_vertices[..., 2] -= 10.0
        vertices[..., 2] += 10
        
        # attributes
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1))
        face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1))
        transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        
        attributes = torch.cat([self.face_uvcoords.expand(batch_size, -1, -1, -1), 
                                transformed_face_normals.detach(), 
                                face_vertices.detach(), 
                                face_normals], 
                                -1)
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes, h, w)
        
        ####
        # vis mask
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        print('---- alpha_images', alpha_images.shape, alpha_images.min(), alpha_images.max())
        img = alpha_images[0,0].data.cpu().numpy()
        imageio.imwrite("alpha_images.png", img)

        # albedo
        uvcoords_images = rendering[:, :3, :, :]; grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
        albedo_images = F.grid_sample(albedos, grid, align_corners=False)
        print('---- uvcoords_images', uvcoords_images.shape, uvcoords_images.min(), uvcoords_images.max())
        img = alpha_images[0].permute(1,2,0).data.cpu().numpy()
        imageio.imwrite("albedo_images.png", img)
        
        # visible mask for pixels with positive normal direction
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < -0.05).float()

        # shading
        normal_images = rendering[:, 9:12, :, :]
        print('---- normal_images', normal_images.shape, normal_images.min(), normal_images.max())
        img = normal_images[0,0].data.cpu().numpy()
        imageio.imwrite("normal_images.png", img)
        
        if lights is not None:
            if lights.shape[1] == 9:
                shading_images = self.add_SHlight(normal_images, lights)
            else:
                if light_type=='point':
                    vertice_images = rendering[:, 6:9, :, :].detach()
                    shading = self.add_pointlight(vertice_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
                    shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2)
                else:
                    shading = self.add_directionlight(normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
                    shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2)
            images = albedo_images*shading_images
        else:
            images = albedo_images
            shading_images = images.detach()*0.
            
        if background is not None:
            images = images*alpha_images + background*(1.-alpha_images)
            albedo_images = albedo_images*alpha_images + background*(1.-alpha_images)
        else:
            images = images*alpha_images 
            albedo_images = albedo_images*alpha_images 

        print('---- images', images.shape, images.min(), images.max())
        img = images[0,0].data.cpu().numpy()
        imageio.imwrite("images.png", img)
        print('---- albedo_images', albedo_images.shape, albedo_images.min(), albedo_images.max())
        img = albedo_images[0,0].data.cpu().numpy()
        imageio.imwrite("albedo_images.png", img)
        # exit(0)
        
        outputs = {
            'images': images,
            'albedo_images': albedo_images,
            'alpha_images': alpha_images,
            'pos_mask': pos_mask,
            'shading_images': shading_images,
            'grid': grid,
            'normals': normals,
            'normal_images': normal_images*alpha_images,
            'transformed_normals': transformed_normals,
        }
        
        return outputs

    def add_SHlight(self, normal_images, sh_coeff):
        '''
            sh_coeff: [bz, 9, 3]
        '''
        N = normal_images
        sh = torch.stack([
                N[:,0]*0.+1., N[:,0], N[:,1], \
                N[:,2], N[:,0]*N[:,1], N[:,0]*N[:,2], 
                N[:,1]*N[:,2], N[:,0]**2 - N[:,1]**2, 3*(N[:,2]**2) - 1
                ], 
                1) # [bz, 9, h, w]
        sh = sh*self.constant_factor[None,:,None,None]
        shading = torch.sum(sh_coeff[:,:,:,None,None]*sh[:,:,None,:,:], 1) # [bz, 9, 3, h, w]  
        return shading

    def add_pointlight(self, vertices, normals, lights):
        '''
            vertices: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_positions = lights[:,:,:3]; light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_positions[:,:,None,:] - vertices[:,None,:,:], dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)

    def add_directionlight(self, normals, lights):
        '''
            normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_direction = lights[:,:,:3]; light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_direction[:,:,None,:].expand(-1,-1,normals.shape[1],-1), dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        # normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)

    def render_shape(self, vertices, transformed_vertices, colors = None, images=None, detail_normal_images=None, 
                lights=None, return_grid=False, uv_detail_normals=None, h=None, w=None):
        '''
        -- rendering shape with detail normal map
        '''
        batch_size = vertices.shape[0]
        # set lighting
        if lights is None:
            light_positions = torch.tensor(
                [
                [-1,1,1],
                [1,1,1],
                [-1,-1,1],
                [1,-1,1],
                [0,0,1]
                ]
            )[None,:,:].expand(batch_size, -1, -1).float()
            light_intensities = torch.ones_like(light_positions).float()*1.7
            lights = torch.cat((light_positions, light_intensities), 2).to(vertices.device)
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10

        # Attributes
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1)); face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1)); transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        if colors is None:
            colors = self.face_colors.expand(batch_size, -1, -1, -1)
        attributes = torch.cat([colors, 
                        transformed_face_normals.detach(), 
                        face_vertices.detach(), 
                        face_normals,
                        self.face_uvcoords.expand(batch_size, -1, -1, -1)], 
                        -1)
        # rasterize
        # import ipdb; ipdb.set_trace()
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes, h, w)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        albedo_images = rendering[:, :3, :, :]
        # mask
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < 0.15).float()

        # shading
        normal_images = rendering[:, 9:12, :, :].detach()
        vertice_images = rendering[:, 6:9, :, :].detach()
        if detail_normal_images is not None:
            normal_images = detail_normal_images

        shading = self.add_directionlight(normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
        shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2).contiguous()        
        shaded_images = albedo_images*shading_images

        alpha_images = alpha_images*pos_mask
        if images is None:
            shape_images = shaded_images*alpha_images + torch.zeros_like(shaded_images).to(vertices.device)*(1-alpha_images)
        else:
            shape_images = shaded_images*alpha_images + images*(1-alpha_images)
        if return_grid:
            uvcoords_images = rendering[:, 12:15, :, :]; 
            grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
            return shape_images, normal_images, grid, alpha_images
        else:
            return shape_images
    
    def render_depth(self, transformed_vertices):
        '''
        -- rendering depth
        '''
        batch_size = transformed_vertices.shape[0]

        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] - transformed_vertices[:,:,2].min()
        z = -transformed_vertices[:,:,2:].repeat(1,1,3).clone()
        z = z-z.min()
        z = z/z.max()
        # Attributes
        attributes = util.face_vertices(z, self.faces.expand(batch_size, -1, -1))
        # rasterize
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        depth_images = rendering[:, :1, :, :]
        return depth_images
    
    def render_colors(self, transformed_vertices, colors):
        '''
        -- rendering colors: could be rgb color/ normals, etc
            colors: [bz, num of vertices, 3]
        '''
        batch_size = colors.shape[0]

        # Attributes
        attributes = util.face_vertices(colors, self.faces.expand(batch_size, -1, -1))
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)
        ####
        alpha_images = rendering[:, [-1], :, :].detach()
        images = rendering[:, :3, :, :]* alpha_images
        return images

    def world2uv(self, vertices):
        '''
        warp vertices from world space to uv space
        vertices: [bz, V, 3]
        uv_vertices: [bz, 3, h, w]
        '''
        batch_size = vertices.shape[0]
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        uv_vertices = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1), self.uvfaces.expand(batch_size, -1, -1), face_vertices)[:, :3]
        return uv_vertices

def animate_multiple(input_path, output_name, prefixs, fmt="gif", fps=30):
    assert fmt in ["gif", "mp4"]
    
    files = [int(f.split(".")[0].split("_")[-1]) for f in os.listdir(input_path) if ".png" in f]
    files = sorted(list(set(files)))
        
    if fmt == "gif":
        # Generate GIF from images
        with imageio.get_writer(output_name, mode="I") as writer:
            for file in tqdm(files, desc="Generating GIF..."):
                image = []
                for pref in prefixs:
                    # print(os.path.join(input_path, "{:s}_{:d}.png".format(pref, file)))
                    img = np.array(imageio.imread(os.path.join(input_path, "{:s}_{:d}.png".format(pref, file))), dtype=np.uint8)
                    image.append(img)
                    os.remove(os.path.join(input_path, "{:s}_{:d}.png".format(pref, file)))
                image = np.concatenate(image, axis=1)
                writer.append_data(image[..., :3])
                
    elif fmt == "mp4":
        # Generate MP4 from images
        with imageio.get_writer(output_name, fps=fps) as writer:
            for file in tqdm(files, desc="Generating MP4..."):
                image = []
                for pref in prefixs:
                    img = np.array(imageio.imread(os.path.join(input_path, "{:s}_{:d}.png".format(pref, file))), dtype=np.uint8)
                    image.append(img)
                    os.remove(os.path.join(input_path, "{:s}_{:d}.png".format(pref, file)))
                image = np.concatenate(image, axis=1)
                writer.append_data(image[..., :3])
    else:
        raise ValueError("format {:s} not recognized!".format(fmt))

# クラス定義
class MeshColorRendering(nn.Module):
    def __init__(self, device, filename, batch_size=1, image_size=256):
        super().__init__()
        # バッチサイズの設定 - レンダリングする異なる視点の数
        self.batch_size = batch_size
        # メッシュの読込み(バッチサイズ分作成)
        self.meshes = self.loadMeshes(device, filename)
        self.device = device

        # Get a batch of viewing angles.
        # elev = torch.linspace(0, 360, self.batch_size)
        # azim = torch.linspace(0, 360, self.batch_size)
 
        # camerasのヘルパー関数は、入力とブロードキャストの混合型をサポートしています。
        # そのため、距離は1.7で固定して、角度だけ変更するようなことも可能です。
        # R, T = look_at_view_transform(dist=2.0, elev=elev, azim=azim)
        R, T = look_at_view_transform(dist=2.0, elev=0, azim=0)
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

        # ラスタライザの設定
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            bin_size=0
        )

        # ポイントライトの設置
        lights = PointLights(device=self.device, location=[[0.5, 2.5, -4.0]])

        # フォンのレンダラの作成。フォンシェーダはテクスチャのuv座標を補間します。
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=cameras,
                lights=lights
            )
        )
        self.renderer = renderer
        self.filename = filename
        self.images = None

    # forward
    def forward(self):
        # メッシュの描画
        self.images = self.renderer(meshes_world=self.meshes)
        return self.images * 255.
    
    # objファイルの読込み
    def loadMeshes(self, device, obj_filename):
        verts, faces, aux = load_obj(obj_filename)
        vertNum = verts.shape[0]
        faceNum =  faces.verts_idx.shape[0]
        print("vertNum=", vertNum)
        print("faceNum=", faceNum)
        self.verts = verts
        self.faces = faces

        # get vertex_uvs
        if aux.verts_uvs is not None:
            verts_uvs = aux.verts_uvs[None,...]
        else:
            # 0～1の乱数を適当に設定
            verts_uvs = torch.rand(1, vertNum, 2)
        print("verts_uvs.shape=", verts_uvs.shape)

        # get face_uvs
        if len(faces.textures_idx) > 0:
            faces_uvs = faces.textures_idx[None,...]
        else:
            # 面のインデックスで代用
            faces_uvs = faces.verts_idx[None]
        print("faces_uvs.shape=", faces_uvs.shape)

        # get texture data
        if aux.texture_images is not None:
            tex_maps = aux.texture_images
            texture_image = list(tex_maps.values())[0]
            texture_image = texture_image[None, ...]
        else:
            # ダミーの白色画像を準備
            texture_image = imread('render/ude_v2/assets/mean_texture.jpg')
            texture_image = torch.from_numpy(texture_image/255.0)[None]
            texture_image = texture_image.float()
        print("texture_image.shape=", texture_image.shape)

        # tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)
        tex = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)
        self.verts_uvs = verts_uvs
        self.faces_uvs = faces_uvs
        self.maps = texture_image
        self.tex = tex

        # centering, scaling
        center = verts.mean(0)
        verts = verts - center
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale
        
        meshes = Meshes(verts=[verts], faces=[faces.verts_idx], textures=tex).to(device)

        # バッチサイズ分のメッシュを作る必要があります。
        # `extend`関数を使うと、簡単に作成できます。テクスチャも拡張してくれます。
        meshes = meshes.extend(self.batch_size)

        return meshes
    
    # タイル画像として保存
    # https://note.nkmk.me/python-opencv-hconcat-vconcat-np-tile/
    def saveTileImage(self, filename):
        im = self.images.cpu().numpy()
        for i in range(20):
            im[i] = cv2.cvtColor(255*im[i], cv2.COLOR_BGRA2RGBA)
        im_tile = self.concat_tile([[im[0], im[1], im[2], im[3], im[4]],
                                    [im[5], im[6], im[7], im[8], im[9]],
                                    [im[10], im[11], im[12], im[13], im[14]],
                                    [im[15], im[16], im[17], im[18], im[19]]])
        cv2.imwrite(filename, im_tile)
    
    def concat_tile(self, im_list_2d):
        return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])
    
    def update_mesh(self, verts, faces, device):
        """
        :param verts: [V, 3]
        :param faces: [F, 3]
        """
        vertNum = verts.shape[0]
        faceNum =  faces.shape[0]
        
        # centering, scaling
        center = verts.mean(0)
        verts = verts - center
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale
        
        meshes = Meshes(verts=[verts], faces=[faces], textures=self.tex).to(device)
        meshes = meshes.extend(self.batch_size)
        self.meshes = meshes
    
class MeshNormalRendering(nn.Module):
    def __init__(self, device, filename, batch_size=1, image_size=256):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        verts, faces, aux = load_obj("render/ude_v2/assets/head_template.obj")
        self.verts_uvs = aux.verts_uvs[None, ...].to(device)       # (N, V, 2)
        self.faces_uvs = faces.textures_idx[None, ...].to(device)  # (N, F, 3)
        
        R, T = look_at_view_transform(dist=2.0, elev=0, azim=0)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1
        )
        self.rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
    
    def update_mesh(self, verts, faces, device=None):
        center = verts.mean(0)
        verts = verts - center
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale
        
        mesh = Meshes(verts=[verts], faces=[faces])
        vertex_normals = mesh.verts_normals_packed() #[53149,3]
        faces_normals = vertex_normals[faces]#[105694,3,3]
        verts_mesh = mesh.verts_packed()
        
        fragments = self.rasterizer(mesh)
        pixel_normals = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_normals)
        pixel_normals = pixel_normals.reshape(1, self.image_size, self.image_size, 3)
        images = (pixel_normals + 1)/2
        self.images = images * 255.
    
    def forward(self):
        """
        :param verts: [V, 3]
        :param faces: [F, 3]
        """
        return self.images
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=3, help='')
    parser.add_argument('--vis_num_samples', type=int, default=3, help='')

    parser.add_argument('--audio_path', type=str, default="../dataset/BEAT_v0.2.1/beat_english_v0.2.1", help="")
    parser.add_argument('--base_folder', type=str, default='ude', help='')
    parser.add_argument('--sub_base_folder', type=str, default='debug', help='')
    parser.add_argument('--input_folder', type=str, default='t2m', help='path of training folder')
    parser.add_argument('--output_folder', type=str, default='t2m', help='path of training folder')
    parser.add_argument('--cmp_folder', type=str, default='cmp_t2m', help='path of training folder')
    parser.add_argument('--fps', type=int, default=30, help='path of training folder')
    parser.add_argument('--animate', type=str2bool, default=True, help='animate or not')
    parser.add_argument('--draw_tokens', type=str2bool, default=False, help='animate or not')
    parser.add_argument('--add_audio', type=str2bool, default=False, help='animate or not')
    parser.add_argument('--plot_type', type=str, default="2d", help='animate or not')
    parser.add_argument('--video_fmt', type=str, default="mp4", help='animate or not')
    parser.add_argument('--sample_rate', type=int, default=1, help='sampling rate')
    parser.add_argument('--num_proc', type=int, default=16, help='sampling rate')
    parser.add_argument('--max_length', type=int, default=10000, help='maximum length to animate')
    parser.add_argument('--visualize_gt', type=str2bool, default=False, help='visualize the GT joints or not')
    parser.add_argument('--post_process', type=str2bool, default=True, help='conduct post-process or not')
    
    args = parser.parse_args()
    return args
   
def main(args):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # # Render RGB images, to implement this, make sure the texture uv map is available!
    # model = MeshColorRendering(device, 
    #                            filename="render/ude_v2/assets/head_template.obj", 
    #                            batch_size=1, 
    #                            image_size=512).to(device)
    
    # Render Normal images
    model = MeshNormalRendering(device, 
                               filename="render/ude_v2/assets/head_template.obj", 
                               batch_size=1, 
                               image_size=512).to(device)
    
    flame_model = FLAME(flame_cfg, flame_full=True).to(device)
    
    num_samples = args.num_samples
    base_folder = args.base_folder
    sub_base_folder = args.sub_base_folder
    input_folder = args.input_folder
    output_folder = args.output_folder
    cmp_folder = args.cmp_folder

    FPS = args.fps
    
    input_path = "logs/ude_v2/eval/{:s}/{:s}/{:s}".format(base_folder, sub_base_folder, input_folder)
    audio_path = args.audio_path
    output_path = "logs/ude_v2/eval/{:s}/animation/{:s}".format(base_folder, output_folder)
    image_path = "logs/ude_v2/eval/{:s}/image/{:s}".format(base_folder, output_folder)

    files = os.listdir(input_path)
    
    for idx, file in enumerate(files):
        try:
            tid = file.split(".")[0].split("_")[1]
        except:
            tid = "T000"
            
        data = np.load(os.path.join(input_path, file), allow_pickle=True).item()

        name = data["caption"][0]
        name = name.replace(" ", "_").replace(",", "").replace(".", "").replace("/", "")    # Remove specific char
        
        rc_poses = {key+"_pose": torch.from_numpy(val[..., :args.max_length]).permute(0, 2, 1).float().to(device) for key, val in data["pred"].items()}
        
        rc_output = convert_flame2joints(flame_model, **rc_poses)
        rc_faces = flame_model.faces_tensor
        rc_repre = "flame"
        rc_vertices = rc_output["vertices"]
        
        num_frames = rc_vertices.size(1)
        image_output_path = os.path.join(output_path, tid)
        print("Number of frame = {:d}".format(num_frames))
        for n in range(0, num_frames):
            model.update_mesh(verts=rc_vertices[0, n], faces=rc_faces, device=device)
            rendered_image = model()
            for _, image in enumerate(rendered_image):
                img_path = os.path.join(image_output_path, "temp", "temp_{:d}.png".format(n))
                imageio.imwrite(img_path, image[..., :3].data.cpu().numpy().astype(np.uint8))
                
        # Generate video
        image_output_path = os.path.join(output_path, tid)
        animate_multiple(os.path.join(image_output_path, "temp"), 
                         os.path.join(image_output_path, name+"."+"mp4"), 
                         ["temp"], fmt="mp4", fps=FPS)
        
if __name__ == "__main__":
    
    args = parse_args()
    main(args)
