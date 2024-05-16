
import torch

import numpy as np

import torch as th
import pytorch3d.renderer
import pytorch3d.structures


class Renderer(torch.nn.Module):
    def __init__(self, silhouette_renderer, depth_renderer, max_depth=5, image_size=256, batch_size=1, device=torch.device('cuda:0')):
        super().__init__()
        self.silhouette_renderer = silhouette_renderer
        self.depth_renderer = depth_renderer

        self.max_depth = max_depth
        self.device = device
        self.batch_size = batch_size

        # Pixel coordinates
        self.X, self.Y = torch.meshgrid(torch.arange(0, image_size), torch.arange(0, image_size))
        self.X = (2*(0.5 + self.X.unsqueeze(0).unsqueeze(-1))/image_size - 1).float().to(self.device) 
        self.Y = (2*(0.5 + self.Y.unsqueeze(0).unsqueeze(-1))/image_size - 1).float().to(self.device) 

        self.X = self.X.repeat(batch_size,1,1,1)
        self.Y = self.Y.repeat(batch_size,1,1,1)

    def depth_2_normal(self, depth, depth_unvalid, cameras):

        B, H, W, C = depth.shape

        grad_out = torch.zeros(B, H, W, 3).to(self.device) 
        # Pixel coordinates
        xy_depth = torch.cat([self.X, self.Y, depth], 3).to(self.device).reshape(B,-1, 3) 
        xyz_unproj = cameras.unproject_points(xy_depth, world_coordinates=False)

        # compute tangent vectors
        XYZ_camera = xyz_unproj.reshape(B, H, W, 3)
        vx = XYZ_camera[:,1:-1,2:,:]-XYZ_camera[:,1:-1,1:-1,:]
        vy = XYZ_camera[:,2:,1:-1,:]-XYZ_camera[:,1:-1,1:-1,:]

        # finally compute cross product
        normal = torch.cross(vx.reshape(-1, 3),vy.reshape(-1, 3))
        normal_norm = normal.norm(p=2, dim=1, keepdim=True)

        normal_normalized = normal.div(normal_norm)
        # reshape to image
        normal_out = normal_normalized.reshape(B, H-2, W-2, 3)
        grad_out[:,1:-1,1:-1,:] = (0.5 - 0.5*normal_out)

        # zero out +Inf
        grad_out[depth_unvalid] = 0.0

        return grad_out

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        # take care of soft silhouette
        silhouette_ref = self.silhouette_renderer(meshes_world=meshes_world, **kwargs)
        silhouette_out = silhouette_ref[..., 3]

        # now get depth out
        depth_ref = self.depth_renderer(meshes_world=meshes_world, **kwargs)
        depth_ref = depth_ref.zbuf[...,0].unsqueeze(-1)
        depth_unvalid = depth_ref<0
        depth_ref[depth_unvalid] = self.max_depth
        depth_out = depth_ref[..., 0]

        # post process depth to get normals, contours
        normals_out = self.depth_2_normal(depth_ref, depth_unvalid.squeeze(-1), kwargs['cameras'])

        return normals_out, silhouette_out


def get_renderer(batch_size=1, img_size=512, cam_distance=2.4, device=torch.device('cuda:0')):

    R, T = pytorch3d.renderer.look_at_view_transform(cam_distance, 0, 0)
    R = R.repeat(batch_size, 1, 1) # B x 3 x 3
    T = T.repeat(batch_size, 1) # B x 3
    
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(device=device, R=R, T=T)

    lights = pytorch3d.renderer.PointLights(device=device, location=[[0.0, 0.0, 3.0]])
    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=img_size,
        blur_radius=0.000001,
        faces_per_pixel=1,
    )
    raster_settings_soft = pytorch3d.renderer.RasterizationSettings(
        image_size=img_size,
        blur_radius=np.log(1. / 1e-4 - 1.)*1e-5,
        faces_per_pixel=25,
    )

    # instantiate renderers
    silhouette_renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings_soft
        ),
        shader=pytorch3d.renderer.SoftSilhouetteShader()
    )
    depth_renderer = pytorch3d.renderer.MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    renderer_class = Renderer(silhouette_renderer, depth_renderer, image_size=img_size, device=device, batch_size=batch_size)

    renderer = lambda meshes : renderer_class(meshes, cameras=cameras, lights=lights)

    return renderer