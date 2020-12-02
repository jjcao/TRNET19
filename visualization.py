import open3d as o3d
import scipy.io as scio
import numpy as np
import json
from skimage.io import imsave
import h5py


def vis_nor(cc, normals, points, name):
    h,w = np.shape(points)
    colors = np.random.rand(h,w)
    label = []
    dis_matrix = np.random.rand(h, 40)
    for i in range(h):
        for j in range(40):
            dis_matrix[i,j] = np.arccos(sum(normals[i,:]*cc[j,:])/(np.linalg.norm(normals[i,:])*np.linalg.norm(cc[j,:])))
        label.append(np.argmin(dis_matrix[i,:]))
        colors[i, :] = cc[label[i], :]
    norimg = colors.reshape(480,640,3).transpose(1,0,2)
    imsave('./vis_results/nyu_k128_s007_nostd_sumd_pt32_pl32_num/'+name+'nyunormals.png', norimg)

def normal_correct(normals_gt, normals):
    h,_ = np.shape(normals_gt)
    for i in range(h):
        ang = np.arccos(sum(normals_gt[i, :]*normals[i,:])/(np.linalg.norm(normals_gt[i,:])*np.linalg.norm(normals[i,:])))
        if ang > np.pi/2:
            normals[i,:] = -normals[i,:]
    return normals

cluster_path = '/data/nyuv2_surfacenormal_metadata/surfacenormal_metadata/vocab40.mat'
vis_name = '/data/nyuv2_surfacenormal_metadata/pcloud/vis_name.txt'
depimage_name = '/data/nyu_depth_v2_labeled.mat'
cluster = scio.loadmat(cluster_path)
cc = cluster['vocabs'][0][0][0][0][0]
list_vis = []
vis_index = []
with open(vis_name, 'r') as f:
    for line in f:
        list_vis.append(line.rstrip('\n'))
        vis_index.append(int(line.rstrip('\n'))-1)

with h5py.File(depimage_name, 'r') as f:
    keys = list(f.keys())
depth = h5py.File(depimage_name, 'r')['depths'][()]
img = h5py.File(depimage_name, 'r')['images'][()]

for i in range(0, len(list_vis)):  #len(list_vis)
    points = np.loadtxt('/data/nyuv2_surfacenormal_metadata/pcloud/'+list_vis[i]+'.xyz')
    normals_gt = np.loadtxt('/data/nyuv2_surfacenormal_metadata/pcloud/'+list_vis[i]+'.normals')
    normals_pcp = np.loadtxt('/data/lkwang/pcpnet/results_vis/single_scale_normal/'+list_vis[i]+'.normals')
    normals_drne = np.loadtxt('/home/lkwang/DRNE19/data/results_vis/nyu_k128_s007_nostd_sumd_pt32_pl32_num/'+list_vis[i]+'.normals')
    #depimg = (depth[vis_index[i],:,:] * 255).astype(np.uint8)
    #rgbimg = img[vis_index[i],:, :,:].transpose(1,2,0)
    #imsave('./vis_results/single/'+list_vis[i]+'depth.png', depimg)
    #imsave('./vis_results/single/'+list_vis[i]+ 'img.png',rgbimg)
    # normals_nesti = np.loadtxt('/data/hrzhu/Nesti-Net/experts/nesti_nyuv2_results/'+ordered_name[i]+'.normals')
    #normals_pcp = normal_correct(normals_gt, normals_pcp)
    normals_drne = normal_correct(normals_gt, normals_drne)
    # normals_nesti = normal_correct(normals_gt,normals_nesti)
    #vis_nor(cc, normals_gt, points, list_vis[i]+'gt')
    #vis_nor(cc, normals_pcp, points, list_vis[i]+'pcp')
    vis_nor(cc, normals_drne, points, list_vis[i]+'drne')
    # vis_nor(cc, normals_nesti, points)


mesh = o3d.geometry.PointCloud()
mesh.points = o3d.utility.Vector3dVector(cc)
o3d.visualization.draw_geometries([mesh])

