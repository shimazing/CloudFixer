## Plot ##
import open3d as o3d
import numpy as np
import os

dataset = 'modelnet'
class_idx = 3
path = f"./visualize/"
file_list = ['dgcnn_background_5/vis/pc_1_final.ply'] #os.listdir(path)
corruptions = ['occlusion', 'lidar', 'density_inc', 'density', 'cutout', # density
               'uniform', 'gaussian', 'impulse', 'upsampling', 'background', # noise
               'rotation', 'shear', 'distortion', 'distortion_rbf', 'distortion_rbf_inv', # transformation
              ]
data_idx = 27

for corruption in corruptions:
    file_list = [
        f'dgcnn_{corruption}_5/vis/pc_{data_idx}_final.ply',
        f'dgcnn_{corruption}_5_lr0.2_ef0.05_matching_t0.8_0.02_400iters_betas_0.9_0.999_wd_0_pow1weightedTrue_l1_10_1_cosaneal_l2_0_wMask_wd0_adamax_schedule_t_tlb_lr_linearLR_sub700_wRotation0.2_denoisingThrs100_transFalse_seed2/vis/pc_{data_idx}_final.ply',

    ]

    for file in file_list:
        if 'fps' in file:
            continue
        print("Load a ply point cloud, print it, and render it")
        pcd = o3d.io.read_point_cloud(file)
        R = pcd.get_rotation_matrix_from_xyz((-np.pi/2-1/8*np.pi, 0, -3/4*np.pi))
        pcd = pcd.rotate(R, center=(0,0,0))
        print(pcd)
        print(np.asarray(pcd.points))
        #pcd.paint_uniform_color([0.8, 0.4,0.9])
        #pcd.paint_uniform_color([0, 0.,1.])
        o3d.visualization.draw_geometries([pcd])
