


from pytorch3d.renderer import PointsRenderer, PointsRasterizer, PointsRasterizationSettings, AlphaCompositor, FoVPerspectiveCameras, look_at_view_transform

from PIL import Image
import numpy as np


def getPointCloudRenders_Pytorch3D(pc, center=((0, 0, 0),), distance=2.5, elevation=0.0, maxAzimuth=360, rangeAzimuth=2, device='cuda:0'):
    frames = []
    
    for azimuth in range(0, maxAzimuth, rangeAzimuth):
        R, T = look_at_view_transform(distance, elevation, azimuth)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        raster_settings = PointsRasterizationSettings(image_size=1024, radius = 0.003, points_per_pixel = 10)
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())
    
        frames.append(renderer(pc)[0, ..., :].cpu().numpy())
    return frames


def saveGif(frames, path):
    imgFrames = [Image.fromarray((image * 255).astype(np.uint8)) for image in frames]
    meshGif = imgFrames[0]
    meshGif.save(path, format='GIF', append_images=imgFrames, save_all=True, duration=100, loop=0)


if __name__ == '__main__':
    import pytorch3d.structures as py3d_struct
    import torch
    from IPython.display import Image as IPyImage

    points = torch.rand(size=(10000, 3), device='cuda:0')
    rgb = torch.ones(size=(points.shape[0], 3), device='cuda:0')
    point_cloud = py3d_struct.Pointclouds(points=[points], features=[rgb])

    frames = getPointCloudRenders_Pytorch3D(point_cloud, center=((0, 0, 0),), distance=2.5, elevation=0.0, rangeAzimuth=4)

    meshGifPath = f'./Pytorch3D. Point Cloud Rendering (Geralt).gif'
    saveGif(frames, meshGifPath)

    IPyImage(filename=meshGifPath)