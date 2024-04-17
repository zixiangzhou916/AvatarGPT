import os, sys, argparse
sys.path.append(os.getcwd())
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import colorsys

import trimesh
from trimesh import Trimesh
import pyrender
from pyrender.constants import RenderFlags
import torch
import numpy as np
from tqdm import tqdm
import math
from math import factorial
# import cv2
from scipy.spatial.transform import Rotation as R
import imageio
from smplx import SMPL
    
class Animator(object):
    def __init__(self, vertices, color_labels, faces):
        """
        :param vertices: [T, 6890, 3]
        """
        vertices = self.put_to_ground(vertices=vertices)
        self.MINS = np.min(np.min(vertices, axis=0), axis=0)
        self.MAXS = np.max(np.max(vertices, axis=0), axis=0)
        self.vertices = vertices
        self.color_labels = color_labels
        if color_labels is not None:
            self.segments_labels = self.crop_motion_segment(labels=color_labels)
        else:
            self.segments_labels = None
        self.faces = faces
        
        self.minx = self.MINS[0] - 0.5
        self.maxx = self.MAXS[0] + 0.5
        self.minz = self.MINS[2] - 0.5 
        self.maxz = self.MAXS[2] + 0.5
    
    @staticmethod
    def put_to_ground(vertices):
        offset_z = np.min(vertices[0, :, 2])
        vertices[..., 2] -= offset_z
        return vertices
    
    @staticmethod
    def crop_motion_segment(labels):
        segments = []
        cur_segment = [0]
        for i in range(1, len(labels), 1):
            if labels[i-1] == 1 and labels[i] == 1:
                cur_segment.append(i)
            if labels[i-1] == 1 and labels[i] == 0:
                segments.append(cur_segment)
            if labels[i-1] == 0 and labels[i] == 1:
                cur_segment = [i]
        if len(cur_segment) != 0:
            segments.append(cur_segment)       
        return segments
    
    def run(self, outdir, name, mode, fps=20, num_frame=5):
        if mode == "dynamic":
            self.run_dynamic(outdir=outdir, name=name, fps=fps)
        elif mode == "static":
            self.run_static(outdir=outdir, name=name, num_frame=num_frame)
        elif mode == "static_mib":
            self.run_one_sequence_mib(vertices=self.vertices, outdir=outdir, name=name, num_frame=num_frame)
        
    def run_dynamic(self, outdir, name, fps=20):
        """Render motion sequence in mesh format. 
        The output is video.
        """
        frames = self.vertices.shape[0]
        vid = []
        
        traj_range_x = self.MAXS[0]-self.MINS[0]
        traj_range_y = self.MAXS[1]-self.MINS[1]
        traj_center_x = (self.MAXS[0]+self.MINS[0]) / 2
        traj_center_y = (self.MAXS[1]+self.MINS[1]) / 2
        
        # plane = trimesh.creation.box(extents=(traj_range_x + 0.5, traj_range_y + 0.5, 0.01))
        plane = trimesh.creation.box(extents=(traj_range_x + 1.5, traj_range_y + 1.5, 0.01))
        plane = pyrender.Mesh.from_trimesh(plane, smooth=False)
        plane_node = pyrender.Node(mesh=plane, translation=np.array([traj_center_x, traj_center_y, 0.0]))
        
        for i in tqdm(range(frames)):
            subdivided = trimesh.remesh.subdivide(self.vertices[i], self.faces)
            mesh = Trimesh(vertices=subdivided[0], faces=subdivided[1])
            
            # base_color = (0.11, 0.53, 0.8, 0.5)
            if self.color_labels is None:
                base_color = (1, 0.706, 0)
            else:
                if self.color_labels[i] == 1:
                    base_color = (1, 0.706, 0)
                else:
                    base_color = (0.11, 0.53, 0.8, 0.5)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.7,
                alphaMode='OPAQUE',
                baseColorFactor=base_color
            )
            mesh_face_color = np.array([base_color]).repeat(mesh.faces.shape[0], axis=0)
            mesh.visual.face_colors = mesh_face_color
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            
            bg_color = [1, 1, 1, 0.8]
            scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))

            sx, sy, tx, ty = [0.75, 0.75, 0, 0.10]

            camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))

            light = pyrender.DirectionalLight(color=[0.8, 0.8, 0.8], intensity=5.0)
            spot_l = pyrender.SpotLight(color=[.5, .1, .1], intensity=10.0, innerConeAngle=np.pi / 16, outerConeAngle=np.pi /12)
            scene = pyrender.Scene(ambient_light=np.array([0.4, 0.4, 0.4, 1.0]))
            scene.add(light)
            scene.add(spot_l)
            
            scene.add(mesh)
            scene.add_node(plane_node)
            
            # c = np.pi / 2            
            c = -np.pi / 6
            
            """ Static Camera."""
            cam_pose = np.array(
                [[ 1, 0, 0, (self.minx+self.maxx)/2],
                [ 0, np.cos(0), -np.sin(0), self.MINS[1]-2.5],
                [ 0, np.sin(0), np.cos(0), 3.0],
                # [ 0, np.cos(0), -np.sin(0), self.MINS[1]-4.5],
                # [ 0, np.sin(0), np.cos(0), 5.0],
                [ 0, 0, 0, 1]]
            )
            
            """ Dynamic Camera. """
            i1 = max(i-20, 0)
            i2 = min(i+20, frames)
            cur_window = self.vertices[i1:i2]
            x_center = cur_window[..., 0].mean()
            y_center = cur_window[..., 1].mean()
            # x_center = self.vertices[i, :, 0].mean()  # [J, 3]
            # y_center = self.vertices[i, :, 1].mean()
            cam_pose = np.array(
                [[1, 0, 0, x_center], 
                 [0, np.cos(0), -np.sin(0), y_center-2.5], 
                 [0, np.sin(0), np.cos(0), 3.0], 
                 [0, 0, 0, 1]]
            )
            
            Rot = R.from_euler("X", angles=60, degrees=True).as_matrix()
            cam_pose[:3, :3] = Rot
            scene.add(camera, pose=cam_pose)
            
            # render scene
            r = pyrender.OffscreenRenderer(960, 960)
            color, _ = r.render(scene, flags=RenderFlags.SHADOWS_DIRECTIONAL)
            vid.append(color)
            r.delete()
        
        out = np.stack(vid, axis=0)
        imageio.mimsave(outdir + name+'.mp4', out, fps=fps)

    def run_static(self, outdir, name, num_frame=5):
        """Render motion sequence in mesh format. 
        The output is image.
        """
        # self.run_one_sequence_static(vertices=self.vertices, outdir=outdir, name=name, num_frame=num_frame)
        if self.segments_labels is None:
            self.run_one_sequence_static(vertices=self.vertices, outdir=outdir, name=name, num_frame=num_frame)
        else:
            for idx, indices in enumerate(self.segments_labels):
                vertices = self.vertices[indices]
                self.run_one_sequence_static(vertices=vertices, outdir=outdir, name=name+"_{:01d}".format(idx), num_frame=num_frame)

    def run_one_sequence_static(self, vertices, outdir, name, num_frame=5):
        frames = vertices.shape[0]
        vid = []
        
        MINS = np.min(np.min(vertices, axis=0), axis=0)
        MAXS = np.max(np.max(vertices, axis=0), axis=0)
        
        traj_range_x = MAXS[0] - MINS[0]
        traj_range_y = MAXS[1] - MINS[1]
        traj_center_x = (MAXS[0] + MINS[0]) / 2
        traj_center_y = (MAXS[1] + MINS[1]) / 2
        
        plane = trimesh.creation.box(extents=(traj_range_x + 0.5, traj_range_y + 0.5, 0.01))
        # plane_color = np.array([[192/255,192/255,192/255]])
        plane.visual.face_colors = [35,35,35,200]
        plane = pyrender.Mesh.from_trimesh(plane, smooth=False)
        plane_node = pyrender.Node(mesh=plane, translation=np.array([0.0, 0.0, 0.0]))
        
        plane_outer = trimesh.creation.box(extents=(traj_range_x + 2.0, traj_range_y + 2.0, 0.01))
        # plane_outer_color = np.array([[135/255,135/255,135/255]])
        plane_outer.visual.face_colors = [135,135,135,200]
        plane_outer = pyrender.Mesh.from_trimesh(plane_outer, smooth=False)
        plane_outer_node = pyrender.Node(mesh=plane_outer, translation=np.array([0.0, 0.0, -0.01]))
        
        Rs = np.asarray([255] * frames)
        Gs = np.arange(180, 0, (0-180)/frames)
        Bs = np.asarray([0] * frames)
        
        bg_color = [1, 1, 1, 0.8]
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))
        sx, sy, tx, ty = [0.75, 0.75, 0, 0.10]
        # Perspective matrix. The larger the denominator is, the closer the view point is.
        camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))
        light = pyrender.DirectionalLight(color=[0.8, 0.8, 0.8], intensity=5.0)
        spot_l = pyrender.SpotLight(color=[.5, .1, .1], intensity=10.0, innerConeAngle=np.pi / 16, outerConeAngle=np.pi /12)
        scene = pyrender.Scene(ambient_light=np.array([0.4, 0.4, 0.4, 1.0]))
        scene.add(light)
        scene.add(spot_l)
        scene.add_node(plane_node)
        scene.add_node(plane_outer_node)
        
        c = +np.pi / 6
        
        # Rot = R.from_euler("xz", angles=[60,30], degrees=True).as_matrix()
        Rot = R.from_euler("X", angles=60, degrees=True).as_matrix()
        Rot2 = R.from_euler("z", angles=-30, degrees=True).as_matrix()
        # Camera local pose.
        # trans = np.array([(self.minx+self.maxx)/2+1, self.MINS[1]-2.5, 0.0])
        trans = np.array([0.0, -4.0, 0.0])
        # trans = np.array([0.0, -7.0, 0.0])
        # trans = np.matmul(trans, Rot2)
        cam_pose = np.array(
            [[ 1, 0, 0, trans[0]],
            [ 0, np.cos(0), -np.sin(0), trans[1]],
            [ 0, np.sin(0), np.cos(0), 3.0],
            # [ 0, np.sin(0), np.cos(0), 5.0],
            [ 0, 0, 0, 1]]
        )
        cam_pose[:3, :3] = Rot 
        scene.add(camera, pose=cam_pose)
        
        steps = frames // num_frame
        for i in tqdm(range(0, frames, steps)):
            subdivided = trimesh.remesh.subdivide(vertices[i], self.faces)
            mesh = Trimesh(vertices=subdivided[0], faces=subdivided[1])
            
            base_color = (Rs[i]/255, Gs[i]/255, Bs[i]/255)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.7,
                roughnessFactor=0.9, 
                alphaMode='OPAQUE',
                smooth=True, 
                baseColorFactor=base_color
            )
            mesh_face_color = np.array([base_color]).repeat(mesh.faces.shape[0], axis=0)
            mesh.visual.face_colors = mesh_face_color
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            mesh_node = pyrender.Node(mesh=mesh, translation=np.array([-traj_center_x, -traj_center_y, 0.0]))
            
            scene.add_node(mesh_node)            
            
        # render scene
        r = pyrender.OffscreenRenderer(960, 960)
        color, _ = r.render(scene, flags=RenderFlags.SHADOWS_DIRECTIONAL)
        imageio.imsave(outdir + name+'.png', color)
    
    def run_one_sequence_mib(self, vertices, outdir, name, num_frame=5):
        frames = vertices.shape[0]
        vid = []
        
        MINS = np.min(np.min(vertices, axis=0), axis=0)
        MAXS = np.max(np.max(vertices, axis=0), axis=0)
        
        traj_range_x = MAXS[0] - MINS[0]
        traj_range_y = MAXS[1] - MINS[1]
        traj_center_x = (MAXS[0] + MINS[0]) / 2
        traj_center_y = (MAXS[1] + MINS[1]) / 2
        
        plane = trimesh.creation.box(extents=(traj_range_x + 0.5, traj_range_y + 0.5, 0.01))
        # plane_color = np.array([[192/255,192/255,192/255]])
        plane.visual.face_colors = [35,35,35,200]
        plane = pyrender.Mesh.from_trimesh(plane, smooth=False)
        plane_node = pyrender.Node(mesh=plane, translation=np.array([0.0, 0.0, 0.0]))
        
        plane_outer = trimesh.creation.box(extents=(traj_range_x + 2.0, traj_range_y + 2.0, 0.01))
        # plane_outer_color = np.array([[135/255,135/255,135/255]])
        plane_outer.visual.face_colors = [135,135,135,200]
        plane_outer = pyrender.Mesh.from_trimesh(plane_outer, smooth=False)
        plane_outer_node = pyrender.Node(mesh=plane_outer, translation=np.array([0.0, 0.0, -0.01]))
        
        Rs = np.zeros((frames,))
        Gs = np.zeros((frames,))
        Bs = np.zeros((frames,))
        for i in range(len(self.color_labels)):
            if self.color_labels[i] == 0:
                Rs[i] = 255
                Gs[i] = 180
                Bs[i] = 0
            else:
                Rs[i] = 255
                Gs[i] = 200
                Bs[i] = 255
        
        bg_color = [1, 1, 1, 0.8]
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))
        sx, sy, tx, ty = [0.75, 0.75, 0, 0.10]
        # Perspective matrix. The larger the denominator is, the closer the view point is.
        camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))
        light = pyrender.DirectionalLight(color=[0.8, 0.8, 0.8], intensity=5.0)
        spot_l = pyrender.SpotLight(color=[.5, .1, .1], intensity=10.0, innerConeAngle=np.pi / 16, outerConeAngle=np.pi /12)
        scene = pyrender.Scene(ambient_light=np.array([0.4, 0.4, 0.4, 1.0]))
        scene.add(light)
        scene.add(spot_l)
        scene.add_node(plane_node)
        scene.add_node(plane_outer_node)
        
        c = +np.pi / 6
        
        # Rot = R.from_euler("xz", angles=[60,30], degrees=True).as_matrix()
        Rot = R.from_euler("X", angles=60, degrees=True).as_matrix()
        Rot2 = R.from_euler("z", angles=-30, degrees=True).as_matrix()
        # Camera local pose.
        # trans = np.array([(self.minx+self.maxx)/2+1, self.MINS[1]-2.5, 0.0])
        trans = np.array([0.0, -4.0, 0.0])
        # trans = np.array([0.0, -7.0, 0.0])
        # trans = np.matmul(trans, Rot2)
        cam_pose = np.array(
            [[ 1, 0, 0, trans[0]],
            [ 0, np.cos(0), -np.sin(0), trans[1]],
            [ 0, np.sin(0), np.cos(0), 3.0],
            # [ 0, np.sin(0), np.cos(0), 5.0],
            [ 0, 0, 0, 1]]
        )
        cam_pose[:3, :3] = Rot 
        scene.add(camera, pose=cam_pose)
        
        interp_indices = []
        for i in range(len(self.color_labels)):
            if self.color_labels[i] == 1:
                interp_indices.append(i)
        # print('----', interp_indices)
        start_indices = [i for i in range(interp_indices[0])]
        end_indices = [i for i in range(interp_indices[-1]+1, len(self.color_labels), 1)]
        
        start_len = len(start_indices)
        end_len = len(end_indices)
        interp_len = len(interp_indices)
        start_step = start_len // 2
        end_step = end_len // 2
        interp_step = interp_len // 5
        indices = start_indices[::start_step] + interp_indices[::interp_step] + end_indices[::end_step]
        
        
        steps = frames // num_frame
        for i in tqdm(indices):
            if i >= len(self.vertices): continue
            subdivided = trimesh.remesh.subdivide(vertices[i], self.faces)
            mesh = Trimesh(vertices=subdivided[0], faces=subdivided[1])
            
            base_color = (Rs[i]/255, Gs[i]/255, Bs[i]/255)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.7,
                roughnessFactor=0.9, 
                alphaMode='OPAQUE',
                smooth=True, 
                baseColorFactor=base_color
            )
            mesh_face_color = np.array([base_color]).repeat(mesh.faces.shape[0], axis=0)
            mesh.visual.face_colors = mesh_face_color
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            mesh_node = pyrender.Node(mesh=mesh, translation=np.array([-traj_center_x, -traj_center_y, 0.0]))
            
            scene.add_node(mesh_node)            
            
        # render scene
        r = pyrender.OffscreenRenderer(960, 960)
        color, _ = r.render(scene, flags=RenderFlags.SHADOWS_DIRECTIONAL)
        imageio.imsave(outdir + name+'.png', color)
    
if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", type=str, default="meshes/", help='motion npy file dir')
    parser.add_argument("--video_dir", type=str, default="animations/", help='motion npy file dir')
    parser.add_argument("--run_mode", type=str, default="static", help='1. dynamic, 2. static')
    parser.add_argument("--sample_rate", type=int, default=1, help='1. dynamic, 2. static')
    parser.add_argument("--num_frame", type=int, default=5, help='number of frame to visualize in the image')
    args = parser.parse_args()
    
    smpl_mode = SMPL(model_path="pretrained_models/human_models/smpl", gender="NEUTRAL", batch_size=1)
    faces = smpl_mode.faces
    
    mesh_dir = args.mesh_dir
    video_dir = args.video_dir
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    filename_list = [f.split(".")[0] for f in os.listdir(mesh_dir) if ".npy" in f]
    
    # with open("logs/avatar_gpt/eval/t2m_compare.json", "r") as f:
    #     compare_list = json.load(f)
    
    for i, file in enumerate(filename_list):
        
        # if "B1120_T0000" not in file: continue
        
        data = np.load(mesh_dir+file+".npy", allow_pickle=True).item()
        vertices = data["pred"]["body"]
        captions = data["caption"][0]
        
        """ For comparison between multiple methods only. """
        # if captions not in compare_list.keys():
        #     continue
        
        # """ For MotionGPT only. """
        # obj = R.from_euler('x', 90, degrees=True)
        # Rot = obj.as_matrix()
        # for idx, pose in enumerate(vertices):
        #     vertices[idx] = np.matmul(Rot, pose.transpose()).transpose()
            
        color_labels = data.get("color_labels", None)
        # print(color_labels)
        captions = captions.replace(".", "").replace("/", "_")
        words = captions.split(" ")
        name = "_".join(words[:20])
        
        fmt = ".mp4" if args.run_mode == "dynamic" else ".png"
        if os.path.exists(os.path.join(video_dir, name+fmt)):
            print("Rendering [{:d}/{:d}] | Caption: {:s} | Done".format(i+1, len(filename_list), file))
            continue
        
        print("Rendering [{:d}/{:d}] | Caption: {:s}".format(i+1, len(filename_list), captions))
        try:
            animator = Animator(
                vertices=vertices[::args.sample_rate], 
                color_labels=color_labels[::args.sample_rate] if color_labels is not None else None, 
                # color_labels=None, 
                faces=faces)
            
            # segments_labels = animator.segments_labels
            # # print(len(segments_labels))
            # indices = [i for i in range(segments_labels[10][0], segments_labels[14][-1], 1)]
            # new_vertices = vertices[indices]
            # # print(indices)
            
            # selected_indices = [i for i in range(0, 200, 20)] + [i for i in range(200, len(indices), 100)]
            # selected_vertices = new_vertices[selected_indices]
            # steps = 50
            # num_frame = math.ceil(new_vertices.shape[0] / steps)
            
            # animator = Animator(
            #     vertices=new_vertices[::args.sample_rate], 
            #     color_labels=None, 
            #     faces=faces)
            
            num_frame = args.num_frame
            animator.run(
                outdir=video_dir, name=name, 
                mode=args.run_mode, 
                # fps=20/args.sample_rate, 
                fps=20, 
                num_frame=num_frame)
        except:
            pass