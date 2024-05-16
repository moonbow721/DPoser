import colorsys
import os
from functools import partial
import random
from typing import List, Optional
from ml_collections import ConfigDict

import cv2
import matplotlib.pyplot as plt
import numpy as np

os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import trimesh

from lib.body_model.utils import get_smpl_skeleton, get_openpose_skeleton, get_openpose_hand_skeleton, \
    get_openpose_face_skeleton, merge_skeleton
from lib.utils.transforms import get_rotation_matrix_x, rotate_points


def vis_keypoints_with_skeleton(img, kps, kps_lines=None, kp_thresh=0.4, alpha=0.7, radius=10):
    if kps_lines is None:
        if kps.shape[0] == 25:
            kps_lines = get_openpose_skeleton()
        elif kps.shape[0] == 22:
            kps_lines = get_smpl_skeleton()
        elif kps.shape[0] == 21:
            kps_lines = get_openpose_hand_skeleton()
        elif kps.shape[0] == 68:
            kps_lines = get_openpose_face_skeleton()
        elif kps.shape[0] == 25+21*2+68:
            kps_lines = merge_skeleton([get_openpose_skeleton(), get_openpose_hand_skeleton(),
                                        get_openpose_hand_skeleton(), get_openpose_face_skeleton()])
        else:
            raise NotImplementedError

    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1, i2 = kps_lines[l]
        p1 = (kps[i1, 0].astype(np.int32), kps[i1, 1].astype(np.int32))
        p2 = (kps[i2, 0].astype(np.int32), kps[i2, 1].astype(np.int32))
        # Assuming that the keypoints' confidence scores are in the third column,
        # which is not provided in the code snippet. You would need to adapt this
        # if your data structure is different.
        if kps[i1, 2] > kp_thresh and kps[i2, 2] > kp_thresh:
            cv2.line(kp_mask, p1, p2, color=colors[l], thickness=radius//2, lineType=cv2.LINE_AA)
        if kps[i1, 2] > kp_thresh:
            cv2.circle(kp_mask, p1, radius=radius, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[i2, 2] > kp_thresh:
            cv2.circle(kp_mask, p2, radius=radius, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def visualize_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, title=None, output_path=None, ax_lims=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if ax_lims:
        ax.set_xlim(ax_lims[0], ax_lims[1])
        ax.set_ylim(ax_lims[2], ax_lims[3])
        ax.set_zlim(ax_lims[4], ax_lims[5])
        # Set the same scale and view for all axes
        scale = np.max([ax_lims[1] - ax_lims[0], ax_lims[3] - ax_lims[2], ax_lims[5] - ax_lims[4]])
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
    padding_ratio = 1.8
    range_x = joint_max[0] - joint_min[0]
    range_y = joint_max[1] - joint_min[1]
    range_z = joint_max[2] - joint_min[2]

    center_x = (joint_max[0] + joint_min[0]) / 2
    center_y = (joint_max[1] + joint_min[1]) / 2
    center_z = (joint_max[2] + joint_min[2]) / 2

    ax_lims = [
        center_x - padding_ratio * range_x / 2, center_x + padding_ratio * range_x / 2,
        center_y - padding_ratio * range_y / 2, center_y + padding_ratio * range_y / 2,
        center_z - padding_ratio * range_z / 2 - 0.5, center_z + padding_ratio * range_z / 2 - 0.5
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


def vis_skeletons(joints_3d, kpt_3d_vis, kps_lines, output_path):
    rotation_angle_x = np.pi  # 180 degrees rotation around X-axis
    rotation_matrix_x = get_rotation_matrix_x(rotation_angle_x)
    joints_3d = rotate_points(joints_3d, rotation_matrix_x)

    # Check the dimensions of the joints_data
    if len(joints_3d.shape) == 2 or len(joints_3d.shape) == 3 and joints_3d.shape[0] == 1:
        visualize_3d_skeleton(joints_3d, kpt_3d_vis, kps_lines, output_path=output_path)
    elif len(joints_3d.shape) == 3:
        visualize_skeleton_sequence(joints_3d, kpt_3d_vis, kps_lines, output_path)


vis_body_skeletons = partial(vis_skeletons, kpt_3d_vis=np.ones((22, 1)), kps_lines=get_smpl_skeleton())


def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0] + 1) + '/' + str(f[i][0] + 1) + ' ' + str(f[i][1] + 1) + '/' + str(
            f[i][1] + 1) + ' ' + str(f[i][2] + 1) + '/' + str(f[i][2] + 1) + '\n')
    obj_file.close()


def render_mesh(img, mesh, face, cam_param, view='random', distance=7.0):
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

    mesh.vertices[:, 2] -= distance
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


def multiple_render(samples, denormalize_fn, model, target_path, img_name, part='body', convert=True, distance=7.0,
                    idx_map=None, faster=True, device=None, bg_img=None, focal=None, princpt=None, view='front'):
    """
    General function to render multiple body or hand samples.

    :param samples: The pose samples to render.
    :param denormalize_fn: The function for denormalizing the samples.
    :param model: The model to use (body / hand / face model).
    :param target_path: Path to save rendered images.
    :param img_name: Image name pattern.
    :param part: Specify 'body' or 'hand' for rendering.
    :param convert: If True, convert the samples using the normalizer.
    :param idx_map: Optional index mapping.
    :param faster: Use faster rendering if True.
    :param device: Device to use for faster rendering.
    :param bg_img: Background image for rendering.
    :param focal: Focal length for camera.
    :param princpt: Principal point for camera.
    :param view: Viewpoint for rendering.
    """
    os.makedirs(target_path, exist_ok=True)
    if isinstance(samples, dict):
        sample_num = None
        for value in samples.values():
            assert len(value.shape) == 2, "All values in the dictionary should be 2-dimensional, [batch, pose_dim]"
            if sample_num is None:
                sample_num = value.shape[0]
            else:
                assert value.shape[0] == sample_num, "All values in the dictionary should have the same batch size"
    else:
        assert len(samples.shape) == 2, "The shape of samples should be 2-dimensional, [batch, pose_dim]"
        sample_num = samples.shape[0]

    if convert:
        samples = denormalize_fn(samples, to_axis=True)

    if part == 'body':
        output = model(body_pose=samples)
    elif part == 'hand':
        output = model(hand_pose=samples)
    elif part == 'face':
        if isinstance(samples, dict):
            output = model(jaw_pose=samples['jaw_pose'], expression=samples['expression'])
        else:
            output = model(face_params=samples)
    elif part == 'whole-body':
        if isinstance(samples, dict):
            output = model(body_pose=samples['body_pose'], left_hand_pose=samples['left_hand_pose'],
                           right_hand_pose=samples['right_hand_pose'], jaw_pose=samples['jaw_pose'],
                           expression=samples['expression'])
        else:
            output = model(whole_body_params=samples)
    else:
        raise ValueError("Invalid part specified. Choose 'body' or 'hand'.")

    if faster:
        assert device is not None
        faster_render(output.v, output.f, target_path, img_name, device, idx_map)
    else:
        meshes = output.v.detach().cpu().numpy()
        faces = output.f.cpu().numpy()
        for idx in range(sample_num):
            mesh = meshes[idx]
            rendered_img = render_mesh(bg_img, mesh, faces, {'focal': focal, 'princpt': princpt},
                                       view=view, distance=distance)
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

    # [cv2.imwrite(os.path.join(target_path, img_name.format(b_id+1)),
    #              cv2.cvtColor(images[b_id, ..., :3].detach().cpu().numpy() * 255, cv2.COLOR_RGB2BGR))
    #  for b_id in range(len(vertices))]
    for idx in range(len(vertices)):
        save_idx = idx if idx_map is None else idx_map[idx]
        cv2.imwrite(os.path.join(target_path, img_name.format(save_idx + 1)),
                    cv2.cvtColor(images[idx, ..., :3].detach().cpu().numpy() * 255, cv2.COLOR_RGB2BGR))


# from CLIFF
class Renderer(object):

    def __init__(self, focal_length=600, img_w=512, img_h=512,
                 camera_center=None, faces=None, same_mesh_color=False):
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w,
                                                   viewport_height=img_h,
                                                   point_size=1.0)
        if camera_center is None:
            self.camera_center = [torch.div(img_w, 2, rounding_mode='trunc'),
                                  torch.div(img_h, 2, rounding_mode='trunc')]
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
            # bg_img_rgb[mask] = color_rgb[mask] * 0.7 + bg_img_rgb[mask] * 0.3
            return bg_img_rgb

    def render_multiple_front_view(self, verts_list, bg_img_rgb_list=None, bg_color=(0, 0, 0, 0)):
        assert len(verts_list) == len(bg_img_rgb_list)
        render_img_list = []
        for verts, bg_img_rgb in zip(verts_list, bg_img_rgb_list):
            # one person each image
            render_img_list.append(self.render_front_view([verts], bg_img_rgb, bg_color))
        return render_img_list

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


# from HaMeR
def create_raymond_lights() -> List[pyrender.Node]:
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes


def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses


def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)


def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))


def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
            .reshape(*(1,) * len(dims), 1, 4)
            .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)


def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )


class HandRenderer:

    def __init__(self, config: ConfigDict, faces: np.array):
        """
        Wrapper around the pyrender renderer to render MANO meshes.
        Args:
            cfg (ConfigDict): Camera config file.
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
        """
        self.config = config
        self.focal_length = config.focal_length
        self.img_res = config.image_size

        # add faces that make the hand mesh watertight
        faces_new = np.array([[92, 38, 234],
                              [234, 38, 239],
                              [38, 122, 239],
                              [239, 122, 279],
                              [122, 118, 279],
                              [279, 118, 215],
                              [118, 117, 215],
                              [215, 117, 214],
                              [117, 119, 214],
                              [214, 119, 121],
                              [119, 120, 121],
                              [121, 120, 78],
                              [120, 108, 78],
                              [78, 108, 79]])
        faces = np.concatenate([faces, faces_new], axis=0)

        self.camera_center = [self.img_res // 2, self.img_res // 2]
        self.faces = faces
        self.faces_left = self.faces[:, [0, 2, 1]]

    def __call__(self,
                 vertices: np.array,
                 camera_translation: np.array,
                 image: torch.Tensor,
                 full_frame: bool = False,
                 imgname: Optional[str] = None,
                 side_view=False, rot_angle=90,
                 mesh_base_color=(1.0, 1.0, 0.9),
                 scene_bg_color=(0, 0, 0),
                 return_rgba=False,
                 ) -> np.array:
        """
        Render meshes on input image
        Args:
            vertices (np.array): Array of shape (V, 3) containing the mesh vertices.
            camera_translation (np.array): Array of shape (3,) with the camera translation.
            image (torch.Tensor): Tensor of shape (3, H, W) containing the image crop with normalized pixel values.
            full_frame (bool): If True, then render on the full image.
            imgname (Optional[str]): Contains the original image filenamee. Used only if full_frame == True.
        """

        if full_frame:
            image = cv2.imread(imgname).astype(np.float32)[:, :, ::-1] / 255.
        else:
            image = image.clone() * torch.tensor(self.cfg.MODEL.IMAGE_STD, device=image.device).reshape(3, 1, 1)
            image = image + torch.tensor(self.cfg.MODEL.IMAGE_MEAN, device=image.device).reshape(3, 1, 1)
            image = image.permute(1, 2, 0).cpu().numpy()

        renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1],
                                              viewport_height=image.shape[0],
                                              point_size=1.0)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(*mesh_base_color, 1.0))

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        if side_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [0, 1, 0])
            mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)
        scene.add(camera, pose=camera_pose)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        if return_rgba:
            return color

        valid_mask = (color[:, :, -1])[:, :, np.newaxis]
        if not side_view:
            output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)
        else:
            output_img = color[:, :, :3]

        output_img = output_img.astype(np.float32)
        return output_img

    def vertices_to_trimesh(self, vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9),
                            rot_axis=[1, 0, 0], rot_angle=0, is_right=1):
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        if is_right:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces.copy(), vertex_colors=vertex_colors)
        else:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces_left.copy(),
                                   vertex_colors=vertex_colors)
        # mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())

        rot = trimesh.transformations.rotation_matrix(
            np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    def render_rgba(
            self,
            vertices: np.array,
            cam_t=None,
            rot=None,
            rot_axis=[1, 0, 0],
            rot_angle=0,
            camera_z=3,
            # camera_translation: np.array,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0, 0, 0),
            render_res=[256, 256],
            focal_length=None,
            is_right=None,
    ):

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))

        focal_length = focal_length if focal_length is not None else self.focal_length

        if cam_t is not None:
            camera_translation = cam_t.copy()
            camera_translation[0] *= -1.
        else:
            camera_translation = np.array([0, 0, camera_z * focal_length / render_res[1]])

        mesh = self.vertices_to_trimesh(vertices, np.array([0, 0, 0]), mesh_base_color, rot_axis, rot_angle,
                                        is_right=is_right)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        # mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color

    def render_rgba_multiple(
            self,
            vertices: List[np.array],
            cam_t: List[np.array],
            rot_axis=[1, 0, 0],
            rot_angle=0,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0, 0, 0),
            render_res=[256, 256],
            focal_length=None,
            is_right=None,
    ):

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))

        if is_right is None:
            is_right = [1 for _ in range(len(vertices))]

        mesh_list = [pyrender.Mesh.from_trimesh(
            self.vertices_to_trimesh(vvv, ttt.copy(), mesh_base_color, rot_axis, rot_angle, is_right=sss)) for
            vvv, ttt, sss in zip(vertices, cam_t, is_right)]

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        for i, mesh in enumerate(mesh_list):
            scene.add(mesh, f'mesh_{i}')

        camera_pose = np.eye(4)
        # camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        focal_length = focal_length if focal_length is not None else self.focal_length
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color

    def add_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    def add_point_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses(dist=0.5)
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            # node = pyrender.Node(
            #     name=f"light-{i:02d}",
            #     light=pyrender.DirectionalLight(color=color, intensity=intensity),
            #     matrix=matrix,
            # )
            node = pyrender.Node(
                name=f"plight-{i:02d}",
                light=pyrender.PointLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)
