import colorsys
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import trimesh


from lib.body_model.utils import get_smpl_skeleton
from lib.utils.transforms import get_rotation_matrix_x, rotate_points


def visualize_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, title=None, output_path=None, ax_lims=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if ax_lims:
        ax.set_xlim(ax_lims[0], ax_lims[1])
        ax.set_ylim(ax_lims[2], ax_lims[3])
        ax.set_zlim(ax_lims[4] - 0.4, ax_lims[5] - 0.2)
        # Set the same scale and view for all axes
        scale = np.max([ax_lims[1]-ax_lims[0], ax_lims[3]-ax_lims[2], ax_lims[5]-ax_lims[4]])
        ax.set_box_aspect([scale, scale, scale])

    ax.view_init(0, -90)  # Elevation, Azimuthal angles

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
        y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
        z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])

        if kpt_3d_vis[i1, 0] > 0 and kpt_3d_vis[i2, 0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1, 0] > 0:
            ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], color=colors[l], marker='o')
        if kpt_3d_vis[i2, 0] > 0:
            ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], color=colors[l], marker='o')

    if title is not None:
        ax.set_title(title)

    # Remove all axes, grid and background
    ax.axis('off')
    ax.grid(False)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close(fig)  # Close the figure to free up memory


def visualize_skeleton_sequence(joints_seq, kpt_3d_vis, kps_lines, output_path):
    num_frames = joints_seq.shape[0]
    joint_min = np.min(joints_seq.reshape(-1, 3), axis=0)
    joint_max = np.max(joints_seq.reshape(-1, 3), axis=0)
    padding_ratio = 1.2
    range_x = joint_max[0] - joint_min[0]
    range_y = joint_max[1] - joint_min[1]
    range_z = joint_max[2] - joint_min[2]

    center_x = (joint_max[0] + joint_min[0]) / 2
    center_y = (joint_max[1] + joint_min[1]) / 2
    center_z = (joint_max[2] + joint_min[2]) / 2

    ax_lims = [
        center_x - padding_ratio * range_x / 2, center_x + padding_ratio * range_x / 2,
        center_y - padding_ratio * range_y / 2, center_y + padding_ratio * range_y / 2,
        center_z - padding_ratio * range_z / 2, center_z + padding_ratio * range_z / 2
    ]

    if output_path.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

        for i in range(num_frames):
            img_path = f'temp_frame_{i:04d}.png'
            visualize_3d_skeleton(joints_seq[i], kpt_3d_vis, kps_lines, output_path=img_path, ax_lims=ax_lims)
            frame = cv2.imread(img_path)
            out.write(frame)
            os.remove(img_path)

        out.release()
    elif not os.path.splitext(output_path)[1]:  # Treat as directory
        os.makedirs(output_path, exist_ok=True)
        for i in range(num_frames):
            img_path = os.path.join(output_path, f'frame_{i:04d}.png')
            visualize_3d_skeleton(joints_seq[i], kpt_3d_vis, kps_lines, output_path=img_path, ax_lims=ax_lims)
    else:
        raise ValueError("The output_path must end with .mp4 or no extension!")


def vis_skeletons(joints_3d, output_path):
    rotation_angle_x = np.pi  # 180 degrees rotation around X-axis
    rotation_matrix_x = get_rotation_matrix_x(rotation_angle_x)
    joints_3d = rotate_points(joints_3d, rotation_matrix_x)

    kpt_3d_vis = np.ones((22, 1))
    kps_lines = get_smpl_skeleton()

    # Check the dimensions of the joints_data
    if len(joints_3d.shape) == 2:
        visualize_3d_skeleton(joints_3d, kpt_3d_vis, kps_lines, output_path=output_path)
    elif len(joints_3d.shape) == 3:
        visualize_skeleton_sequence(joints_3d, kpt_3d_vis, kps_lines, output_path)


def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0] + 1) + '/' + str(f[i][0] + 1) + ' ' + str(f[i][1] + 1) + '/' + str(
            f[i][1] + 1) + ' ' + str(f[i][2] + 1) + '/' + str(f[i][2] + 1) + '\n')
    obj_file.close()


def render_mesh(img, mesh, face, cam_param, view='random'):
    # mesh
    mesh = trimesh.Trimesh(mesh, face)

    centroid = np.mean(mesh.vertices, axis=0)
    translation_to_origin = trimesh.transformations.translation_matrix(-centroid)
    mesh.apply_transform(translation_to_origin)

    if view == 'random':
        options_side = ['half', '']
        options_direction = ['left', 'right', 'front', 'back']
        options_height = ['above', 'bottom', '']

        chosen_side = random.choice(options_side)
        chosen_direction = random.choice(options_direction)
        chosen_height = random.choice(options_height)

        view = '_'.join([opt for opt in [chosen_side, chosen_direction, chosen_height] if opt])

    if 'half' in view:
        side_angle = 45
    else:
        side_angle = 90

    if 'left' in view:
        angle = np.radians(-side_angle)
    elif 'right' in view:
        angle = np.radians(side_angle)
    elif 'back' in view:
        angle = np.radians(180)
    else:  # front
        angle = np.radians(0)
    axis = [0, 1, 0]
    rotation = trimesh.transformations.rotation_matrix(angle, axis)
    mesh.apply_transform(rotation)

    if 'above' in view:
        angle = np.radians(30)
    elif 'bottom' in view:
        angle = np.radians(-30)
    else:  # nothing
        angle = np.radians(0)
    axis = [1, 0, 0]
    rotation = trimesh.transformations.rotation_matrix(angle, axis)
    mesh.apply_transform(rotation)

    translation_to_centroid = trimesh.transformations.translation_matrix(centroid)
    mesh.apply_transform(translation_to_centroid)

    mesh.vertices[:, 2] -= 7
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE',
                                                  # baseColorFactor=(1.0, 1.0, 0.9, 1.0),
                                                  baseColorFactor=(0.93, 0.6, 0.4, 1.0),
                                                  )
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]

    # save to image
    render_img = rgb * valid_mask + img * (1 - valid_mask)
    return render_img


import torch
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.structures import Meshes


def multiple_render(samples, Normalizer, body_model, target_path, img_name, convert=True,
                    idx_map=None, faster=True, device=None, bg_img=None, focal=None, princpt=None, view='front'):
    os.makedirs(target_path, exist_ok=True)
    assert len(samples.shape) == 2
    sample_num = samples.shape[0]
    if convert:
        samples = Normalizer.offline_denormalize(samples, to_axis=True)
    body_out = body_model(pose_body=samples)
    if faster:
        assert device is not None
        faster_render(body_out.v, body_out.f, target_path, img_name, device, idx_map)
    else:
        meshes = body_out.v.detach().cpu().numpy()
        faces = body_out.f.cpu().numpy()
        for idx in range(sample_num):
            mesh = meshes[idx]
            rendered_img = render_mesh(bg_img, mesh, faces, {'focal': focal, 'princpt': princpt}, view=view)
            save_idx = idx if idx_map is None else idx_map[idx]
            cv2.imwrite(os.path.join(target_path, img_name.format(save_idx + 1)), rendered_img)


# from pose-ndf
def faster_render(vertices, faces, target_path, img_name, device, idx_map=None):
    os.makedirs(target_path, exist_ok=True)
    R, T = look_at_view_transform(2.0, 0, 0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=256,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )
    # create mesh from vertices
    verts_rgb = torch.ones_like(vertices)  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    meshes = Meshes(vertices, faces.unsqueeze(0).repeat(len(vertices), 1, 1), textures=textures)
    images = renderer(meshes)

    for idx in range(len(vertices)):
        save_idx = idx if idx_map is None else idx_map[idx]
        cv2.imwrite(os.path.join(target_path, img_name.format(save_idx + 1)),
                    cv2.cvtColor(images[idx, ..., :3].detach().cpu().numpy() * 255, cv2.COLOR_RGB2BGR))


class Renderer(object):

    def __init__(self, focal_length=600, img_w=512, img_h=512,
                 camera_center=None, faces=None, same_mesh_color=False):
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w,
                                                   viewport_height=img_h,
                                                   point_size=1.0)
        if camera_center is None:
            self.camera_center = [torch.div(img_w, 2, rounding_mode='trunc'), torch.div(img_h, 2, rounding_mode='trunc')]
        else:
            self.camera_center = camera_center
        self.focal_length = focal_length
        self.faces = faces
        self.same_mesh_color = same_mesh_color

    def render_front_view(self, verts, bg_img_rgb=None, bg_color=(0, 0, 0, 0)):
        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=np.ones(3) * 0)
        # Create camera. Camera will always be at [0,0,0]
        camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                                  cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=np.eye(4))

        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        # for DirectionalLight, only rotation matters
        light_pose = trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0])
        scene.add(light, pose=light_pose)
        light_pose = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])
        scene.add(light, pose=light_pose)

        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        # multiple person
        num_people = len(verts)

        # for every person in the scene
        for n in range(num_people):
            mesh = trimesh.Trimesh(verts[n], self.faces)
            mesh.apply_transform(rot)
            if self.same_mesh_color:
                mesh_color = (0.4, 0.6, 0.93, 1.0)
            else:
                mesh_color = colorsys.hsv_to_rgb(float(n) / num_people, 0.5, 1.0)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=mesh_color)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material, wireframe=False)
            scene.add(mesh, 'mesh')

        # Alpha channel was not working previously, need to check again
        # Until this is fixed use hack with depth image to get the opacity
        color_rgba, depth_map = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color_rgb = color_rgba[:, :, :3]
        if bg_img_rgb is None:
            return color_rgb
        else:
            mask = depth_map > 0
            bg_img_rgb[mask] = color_rgb[mask]
            return bg_img_rgb

    def render_side_view(self, verts):
        centroid = verts.mean(axis=(0, 1))  # n*6890*3 -> 3
        # make the centroid at the image center (the X and Y coordinates are zeros)
        centroid[:2] = 0
        aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0][np.newaxis, ...]  # 1*3*3
        pred_vert_arr_side = np.matmul((verts - centroid), aroundy) + centroid
        side_view = self.render_front_view(pred_vert_arr_side)
        return side_view

    def delete(self):
        """
        Need to delete before creating the renderer next time
        """
        self.renderer.delete()
