import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import open3d as o3d
from PIL import Image as im
import math
from scipy import ndimage
import json
from multiprocessing import Pool


DOWNSAMPLING = 10
MPIXEL = 0.05
PADDING = 50
HEIGHT_THRESHOLD = 1.5
CROP_RESOLUTION = 3.2

BASE_DIR = "/scratch/scannet_data"
RESULTS_DIR = "./Results"
LOCAL_DIR = "./Results/local"
GLOBAL_DIR = "./Results/global"
PROPERTIES_DIR = "./Results/properties"

def generate_map(scene):
    camera_intrinsics = np.loadtxt(BASE_DIR+'/intrinsics/'+scene+'/intrinsic_depth.txt')
    #print(camera_intrinsics)
    fx = camera_intrinsics[0][0]
    fy = camera_intrinsics[1][1]
    S = camera_intrinsics[0][1]
    cx = camera_intrinsics[0][2]
    cy = camera_intrinsics[1][2]
    # print(camera_intrinsics)
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, fx=fx,fy=fy,cx=cx,cy=cy)  
    
    # POINT CLOUD REGISTRATION
    
    scans_list = os.listdir(BASE_DIR+'/depths/'+scene)
    scans_list = list(map(lambda x: x.split('.')[0],scans_list))

    pcd_list = []
    for i,scan in enumerate(scans_list):
        depth_map = o3d.geometry.Image(np.ascontiguousarray(np.load(BASE_DIR+'/depths/'+scene+"/"+scan+'.npy')).astype(np.float32))
        mask = cv2.resize(cv2.imread(BASE_DIR+'/masks/'+scene+'/'+scans_list[i]+'.png'),(640,480))
        floor_segmentation = o3d.geometry.Image(np.asarray(np.isin(mask, [5])*254,dtype=np.uint8))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(floor_segmentation), o3d.geometry.Image(depth_map),convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
        pose = np.loadtxt(BASE_DIR+'/poses/'+scene+'/'+scan+'.txt',dtype=np.float32)
        pcd = pcd.transform(pose)
        pcd_list.append(pcd.uniform_down_sample(DOWNSAMPLING))
        

    # COMBINIG THE POINT CLOUD
    
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcd_list)):
            pcd_combined += pcd_list[point_id]
#     o3d.io.write_point_cloud(BASE_DIR+"/pcds/"+scene+".pcd", pcd_combined)
    
    # print(f"Registered point cloud  - total points :{len(pcd_combined.points)} for scene : {scene} ... ")    
    
    # READING THE POINT CLOUD
#     pcd = o3d.io.read_point_cloud(BASE_DIR+"/pcds/"+scene+".pcd")
    points = np.asarray(pcd_combined.points)
    colors = np.asarray(pcd_combined.colors)
    
    max_coordinates = np.nanmax(points,axis=0)
    min_coordinates = np.nanmin(points,axis=0)
    #print(max_coordinates,min_coordinates)
    
    bound = max(abs(max_coordinates[0]),abs(min_coordinates[0]),abs(max_coordinates[1]),abs(min_coordinates[1]))
    W = 2*math.ceil(bound/MPIXEL) + PADDING

    
    # GLOBAL OCCUPANCY MAP GENERATION
    p = np.copy(points)
    col = p[:,0]/MPIXEL + W/2
    row = W/2 - p[:,1]/MPIXEL 
    idx = row.astype(int)*W+col.astype(int)
    valid_idx = p[:,2]<HEIGHT_THRESHOLD
    p = p[valid_idx]
    idx = idx[valid_idx]
    c = (colors[valid_idx,0]*255).astype(int)
    # print(np.sum(c==254))
    rank = np.argsort(idx)
    p = p[rank]
    c = c[rank]
    idx = idx[rank]
    keep = np.ones_like(idx).astype(int)
    keep[:-1] = idx[1:]!=idx[:-1]
    occ = np.cumsum(c==0)[keep==1]
    occ[1:] = occ[1:]-occ[:-1]
    ground = np.logical_and(c==254,p[:,2]<0.2)
    free = np.cumsum(ground)[keep==1]
    free[1:] = free[1:]-free[:-1]
    idx = idx[keep==1]
    occupancy_grid = np.zeros((W*W),dtype=np.float64)
    occupancy_grid[idx] = (occ/(occ+free+1e-6) > 0.5)+1
    occupancy_grid = occupancy_grid.reshape((W,W))
    

    cv2.imwrite(GLOBAL_DIR+scene+'.png',occupancy_grid*127)
    properties = {
        "bounds":{
            "max_x":max_coordinates[0],
            "max_y":max_coordinates[1],
            "max_z":max_coordinates[2],
        },
        "padding":PADDING,
        "resolution":MPIXEL,
        "downsampling":DOWNSAMPLING,
        "dimension": W,
        "height_thresholds": HEIGHT_THRESHOLD
    }

    f = open(PROPERTIES_DIR+scene+'.json', "w")
    json.dump(properties, f, indent = 6)
    f.close()

    if not os.path.exists(LOCAL_DIR+'/'+scene):
        os.makedirs(LOCAL_DIR+'/'+scene, exist_ok=False)
    
    def rotateImage(img, angle, pivot):
        padX = [img.shape[1] - pivot[0], pivot[0]]
        padY = [img.shape[0] - pivot[1], pivot[1]]
        imgP = np.pad(img, [padY, padX,[0,0]], 'constant')
        imgR = ndimage.rotate(imgP, angle, reshape=False,order=0)
        return imgR,padY[0],padX[0]

    def Rotation_to_Euler(R):
        yaw = np.arctan2(R[2][0],R[2][1])
        pitch = np.arccos(R[2][2])
        roll = -np.arctan2(R[0][2],R[1][2])
        return roll
    
    poses = os.listdir(BASE_DIR+'/poses/'+scene)
    img = cv2.imread(GLOBAL_DIR+scene+'.png') 

    for pose in poses:
        p = np.loadtxt(BASE_DIR+'/poses/'+scene+'/'+pose)
        x_disp = (p[0][3]//MPIXEL + img.shape[0]//2).astype(int)
        y_disp = (img.shape[0]//2 - p[1][3]//MPIXEL ).astype(int)
        roll = Rotation_to_Euler(p)
        roll = roll*(180/math.pi)
        res_img,pad_y,pad_x =rotateImage(img,-roll,[x_disp.astype(int),y_disp.astype(int)])
    #     print(np.unique(res_img))
    #     plt.figure(figsize=(10,15))
    #     plt.subplot(1,2,1)
    #     plt.imshow(res_img)
    #     plt.plot(x_disp+pad_x,y_disp+pad_y,'o')
        # Cropping the occupancy map
        start_x = x_disp+pad_x-int((CROP_RESOLUTION//MPIXEL)//2)
        end_x =  start_x + int(CROP_RESOLUTION//MPIXEL)
        start_y = y_disp+pad_y - int(CROP_RESOLUTION//MPIXEL)
        end_y = y_disp+pad_y
        local_occupancy_map = res_img[start_y:end_y,start_x:end_x,:]
    #     plt.subplot(1,2,2)
    #     print(local_occupancy_map.shape)
    #     plt.imshow(local_occupancy_map)
    #     occ_img = im.fromarray(local_occupancy_map)
    #     plt.imshow(local_occupancy_map)
    #     if occ_img.mode != 'RGB':
    #         occ_img = occ_img.convert('RGB')
        cv2.imwrite(LOCAL_DIR+scene+'/'+pose.split('.')[0]+'.png',local_occupancy_map)
    #     occ_img.save("./Results/local/"+scene+'/'+pose.split('.')[0]+'.png')

if __name__ == "__main__":
    scenes = os.listdir(BASE_DIR+'/depths')

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=False)
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR, exist_ok=False)
    if not os.path.exists(GLOBAL_DIR):
        os.makedirs(GLOBAL_DIR, exist_ok=False)
    if not os.path.exists(PROPERTIES_DIR):
        os.makedirs(PROPERTIES_DIR, exist_ok=False)
    
    with Pool(processes=18) as pool:
        pool.map(generate_map,scenes)
