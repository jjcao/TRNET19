import open3d as o3d
import json
import pickle
import gzip
import h5py
import os
import numpy as np
import random

cx = 325.5
cy = 253.5
fx = 518.0
fy = 519.0

def _get_open_lambda(filename, is_save):
    '''
    with file extension pklz, we compress the pickle using gzip.
    '''
    extension = os.path.splitext(filename)[1]
    compress = extension == '.pklz'
    if compress:
        open_mode = 'w' if is_save else 'r'
        # note: I tried using bz2 instead of gzip. It compresses better
        # but it is slower.

        def open_lambda(x):
            return gzip.GzipFile(x, open_mode)

    else:
        open_mode = 'wb' if is_save else 'rb'

        def open_lambda(x):
            return open(x, open_mode)

    return open_lambda

def load_pickle(filename):
    '''
    load a pickle file
    '''
    open_lambda = _get_open_lambda(filename, is_save=False)
    with open_lambda(filename) as file:
        data = pickle.load(file)
    return data

def depth2pcd(depi):
    depi = depi.T
    h, w = np.shape(depi)
    xx, yy = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    X = ((xx - cx) * depi / fx).reshape((h*w))
    Y = ((yy - cy) * depi / fy).reshape((h*w))
    Z = depi.reshape((h*w))

    pcloud = np.array([X, Y, Z]).T
    return pcloud

def write_txt(data, name):
    h = np.shape(data)[0]
    with open(name, 'w') as f:
        for i in range(h):
            f.write('%.6f %.6f %.6f\n' % (data[i,0], data[i, 1], data[i, 2]))
    print('write file %s' % name)

def write_pidx(data, name):
    h = np.shape(data)[0]
    with open(name, 'w') as f:
        for i in range(h):
            f.write('%d\n' % data[i])
    print('write file %s' % name)

def pcd_sample(pcd, normals):
    h = np.shape(pcd)[0]
    sample = random.sample(list(range(h)), 5000)
    pcd = pcd[sample,:]
    normals = normals[sample,:]
    return pcd, normals, sample

data_dir = '/data/nyuv2_surfacenormal_metadata/surfacenormal_metadata/'
data_file = data_dir + 'all_normals.pklz'
data = load_pickle(data_file)

root = '/data/nyuv2_surfacenormal_metadata/pcloud_test_5000/'
txt_file = root + 'testset_all.txt'
train_sn40 = data_dir + 'train_SN40.json'
test_sn40 = data_dir + 'test_SN40.json'
train_name = json.load(open(train_sn40, 'r'))
test_name = json.load(open(test_sn40, 'r'))
depth_data_name = '/data/nyu_depth_v2_labeled.mat'
with h5py.File(depth_data_name, 'r') as f:
    keys = list(f.keys())
depth = h5py.File(depth_data_name, 'r')['depths'][()]
index = []

for j in range(len(test_name)):
    index.append(int(test_name[j]['img'].rstrip('_rgb.jpg'))-1)

with open(txt_file, 'w') as f:
    for i in range(len(index)):
        f.write(data['all_filenames'][index[i]]+'\n')

    
for i in range(len(index)):

    pcd_name = '/data/nyuv2_surfacenormal_metadata/pcloud_test_5000/' + data['all_filenames'][index[i]] +'.xyz'
    normal_name = '/data/nyuv2_surfacenormal_metadata/pcloud_test_5000/' + data['all_filenames'][index[i]] +'.normals'
    pidx_name = '/data/nyuv2_surfacenormal_metadata/pcloud_test_5000/' + data['all_filenames'][index[i]] +'.pidx'
    pcd = depth2pcd(depth[index[i], :, :])
    h,w = np.shape(pcd)
    normals = data['all_normals'][index[i], :, :, :].reshape((h,w))
    pcd, normals, sample = pcd_sample(pcd, normals)
    '''
    pcloud = o3d.geometry.PointCloud()
    pcloud.points = o3d.utility.Vector3dVector(pcd)
    pcloud.normals = o3d.utility.Vector3dVector(normals)
    #pcloud.paint_uniform_color([0.0, 0.75, 1.0])
    o3d.visualization.draw_geometries([pcloud])
    '''

    write_txt(pcd, pcd_name)
    write_txt(normals, normal_name)
    write_pidx(sample, pidx_name)

print('')
