from __future__ import print_function
import os
import os.path
import sys
import torch
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial

# do NOT modify the returned points! kdtree uses a reference, not a copy of these points,
# so modifying the points would make the kdtree give incorrect results
def load_shape(point_filename, normals_filename,  pidx_filename, dim_pts=3):
    pts = np.load(point_filename+'.npy')

    if normals_filename != None:
        normals = np.load(normals_filename+'.npy')
        if dim_pts==6:
            innormals = np.load(normals_filename+'-in.npy',allow_pickle=True)
        else:
            innormals = None
    else:
        normals = None
        innormals = None


    if pidx_filename != None:
        patch_indices = np.load(pidx_filename+'.npy')
    else:
        patch_indices = None

    sys.setrecursionlimit(int(max(1000, round(pts.shape[0]/10)))) # otherwise KDTree construction may run out of recursions
    kdtree = spatial.cKDTree(pts, 10)

    return Shape(pts=pts, kdtree=kdtree, normals=normals,  innormals=innormals, pidx=patch_indices)

class SequentialPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.total_patch_count = None

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + self.data_source.shape_patch_count[shape_ind]

    def __iter__(self):
        return iter(range(self.total_patch_count))

    def __len__(self):
        return self.total_patch_count


class SequentialShapeRandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, sequential_shapes=False, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.sequential_shapes = sequential_shapes
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None
        self.shape_patch_inds = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        # global point index offset for each shape
        shape_patch_offset = list(np.cumsum(self.data_source.shape_patch_count))
        shape_patch_offset.insert(0, 0)
        shape_patch_offset.pop()

        shape_inds = range(len(self.data_source.shape_names))

        if not self.sequential_shapes:
            shape_inds = self.rng.permutation(shape_inds)

        # return a permutation of the points in the dataset where all points in the same shape are adjacent (for performance reasons):
        # first permute shapes, then concatenate a list of permuted points in each shape
        self.shape_patch_inds = [[]]*len(self.data_source.shape_names)
        point_permutation = []
        for shape_ind in shape_inds:
            start = shape_patch_offset[shape_ind]
            end = shape_patch_offset[shape_ind]+self.data_source.shape_patch_count[shape_ind]

            global_patch_inds = self.rng.choice(range(start, end), size=min(self.patches_per_shape, end-start), replace=False)
            point_permutation.extend(global_patch_inds)

            # save indices of shape point subset
            self.shape_patch_inds[shape_ind] = global_patch_inds - start

        return iter(point_permutation)

    def __len__(self):
        return self.total_patch_count

class RandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(self.rng.choice(sum(self.data_source.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count


class Shape():
    def __init__(self, pts, kdtree, normals=None, innormals=None, curv=None, pidx=None):
        self.pts = pts
        self.kdtree = kdtree
        self.normals = normals
        self.innormals = innormals
        self.curv = curv
        self.pidx = pidx # patch center points indices (None means all points are potential patch centers)


class Cache():
    def __init__(self, capacity, loader, loadfunc):
        self.elements = {}
        self.used_at = {}
        self.capacity = capacity
        self.loader = loader
        self.loadfunc = loadfunc
        self.counter = 0

    def get(self, element_id):
        if element_id not in self.elements:
            # cache miss

            # if at capacity, throw out least recently used item
            if len(self.elements) >= self.capacity:
                remove_id = min(self.used_at, key=self.used_at.get)
                del self.elements[remove_id]
                del self.used_at[remove_id]

            # load element
            self.elements[element_id] = self.loadfunc(self.loader, element_id)

        self.used_at[element_id] = self.counter
        self.counter += 1

        return self.elements[element_id]

def gauss_fcn(x, mu = 0, sigma2=0.12):
    tmp = -(x - mu)**2 / (2 * sigma2)
    return torch.exp(tmp)

class PointcloudPatchDataset(data.Dataset):

    # patch radius as fraction of the bounding box diagonal of a shape
    def __init__(self, root, root_in, shape_list_filename, patch_radius, points_per_patch, dim_pts=3, knn=False, 
                 seed=None, identical_epochs=False, 
                 cache_capacity=1, point_count_std=0, sparse_patches=False):

        # initialize parameters
        self.root = root
        self.root_innormals = root_in
        self.shape_list_filename = shape_list_filename
        
        self.patch_radius = patch_radius
        self.points_per_patch = points_per_patch
        self.dim_pts = dim_pts
        self.knn = knn
        self.identical_epochs = identical_epochs
        
        self.sparse_patches = sparse_patches
        self.point_count_std = point_count_std
        self.seed = seed


        # self.loaded_shape = None
        self.load_iteration = 0
        self.shape_cache = Cache(cache_capacity, self, PointcloudPatchDataset.load_shape_by_index)

        # get all shape names in the dataset
        self.shape_names = []
        with open(os.path.join(root, self.shape_list_filename)) as f:
            self.shape_names = f.readlines()
        self.shape_names = [x.strip() for x in self.shape_names]
        self.shape_names = list(filter(None, self.shape_names))

        # initialize rng for picking points in a patch
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        # get basic information for each shape in the dataset
        self.shape_patch_count = []
        self.patch_radius_absolute = []
        for shape_ind, shape_name in enumerate(self.shape_names):
            print('getting information for shape %s' % (shape_name))

            # load from text file and save in more efficient numpy format
            point_filename = os.path.join(self.root, shape_name+'.xyz')
            if not os.path.exists(point_filename+'.npy'):
                pts = np.loadtxt(point_filename).astype('float32')
                np.save(point_filename+'.npy', pts)


            normals_filename = os.path.join(self.root, shape_name+'.normals')
            if not os.path.exists(normals_filename+'.npy'):
                normals = np.loadtxt(normals_filename).astype('float32')
                np.save(normals_filename+'.npy', normals)

            if self.dim_pts == 6:
                innormals_filename = os.path.join(self.root_innormals, shape_name+'.normals')
                #if not os.path.exists(normals_filename+'-in.npy'):
                normals = np.loadtxt(innormals_filename).astype('float32')
                np.save(normals_filename+'-in.npy', normals)

            if self.sparse_patches:
                pidx_filename = os.path.join(self.root, shape_name+'.pidx')
                patch_indices = np.loadtxt(pidx_filename).astype('int')
                np.save(pidx_filename+'.npy', patch_indices)
            else:
                pidx_filename = None

            shape = self.shape_cache.get(shape_ind)

            if shape.pidx is None:
                self.shape_patch_count.append(shape.pts.shape[0])
            else:
                self.shape_patch_count.append(len(shape.pidx))

            bbdiag = float(np.linalg.norm(shape.pts.max(0) - shape.pts.min(0), 2))
            self.patch_radius_absolute.append([bbdiag * rad for rad in self.patch_radius])

    # returns a patch centered at the point with the given global index
    # and the ground truth normal the the patch center
    def __getitem__(self, index):

        # find shape that contains the point with given global index
        shape_ind, patch_ind = self.shape_index(index)

        shape = self.shape_cache.get(shape_ind)
        if shape.pidx is None:
            center_point_ind = patch_ind
        else:
            center_point_ind = shape.pidx[patch_ind]

        # get neighboring points (within euclidean distance patch_radius)
        patch_pts = torch.zeros(sum(self.points_per_patch), 3, dtype=torch.float)
        point_normals = torch.zeros(sum(self.points_per_patch), 3, dtype=torch.float)
        patch_pts_dist = torch.ones(sum(self.points_per_patch), 1, dtype=torch.float)
        patch_pts_valid = []
        
        #scale_ind_range = np.zeros([len(self.patch_radius_absolute[shape_ind]), 2], dtype='int')
        start = 0
        for s, rad in enumerate(self.patch_radius_absolute[shape_ind]):
            if self.knn:
                dist, inds = np.array(shape.kdtree.query(shape.pts[center_point_ind, :], self.points_per_patch[s]))
                rad = dist[-1]
                patch_point_inds = inds.astype(np.int)               
            else:
                patch_point_inds = np.array(shape.kdtree.query_ball_point(shape.pts[center_point_ind, :], rad))

            # optionally always pick the same points for a given patch index (mainly for debugging)
            if self.identical_epochs:
                self.rng.seed((self.seed + index) % (2**32))

            point_count = min(self.points_per_patch[s], len(patch_point_inds))

            # randomly decrease the number of points to get patches with different point densities
            if self.point_count_std > 0:
                point_count = max(5, round(point_count * self.rng.uniform(1.0-self.point_count_std*2)))
                point_count = min(point_count, len(patch_point_inds))            

            # if there are too many neighbors, pick a random subset
            if point_count < len(patch_point_inds):
                patch_point_inds = patch_point_inds[self.rng.choice(len(patch_point_inds), point_count, replace=False)]

            end = start+point_count
            #scale_ind_range[s, :] = [start, end]

            patch_pts_valid += list(range(start, end))

            # convert points to torch tensors
            patch_pts[start:end, :] = torch.from_numpy(shape.pts[patch_point_inds, :])
            
            # todo: gt normals are used, should use pca normal, refer to PCPnet
            if shape.innormals is not None:
                point_normals[start:end, :]=torch.from_numpy(shape.innormals[patch_point_inds, :])
            else:
                point_normals[start:end, :]=torch.from_numpy(shape.normals[patch_point_inds, :])

            # center patch (central point at origin - but avoid changing padded zeros)
            patch_pts[start:end, :] = patch_pts[start:end, :] - torch.from_numpy(shape.pts[center_point_ind, :])          

            # normalize size of patch (scale with 1 / patch radius)
            patch_pts[start:end, :] = patch_pts[start:end, :] / rad
            tmp = patch_pts[start:end, :]
            patch_pts_dist[start:end, 0] = torch.sqrt( torch.sum(tmp ** 2, 1) )

            start = start + self.points_per_patch[s]

        
        patch_normal = torch.from_numpy(shape.normals[center_point_ind, :])
        patch_normal = patch_normal.unsqueeze(0)

        mask = abs((point_normals* patch_normal).sum(1))
        mask[mask>1] = 1 # for RuntimeWarning: invalid value encountered in arccos
        mask = np.rad2deg(np.arccos(mask))

        seg = torch.zeros(sum(self.points_per_patch), 1, dtype=torch.long)
        seg[mask<25] = 0
        seg[mask>=25] = 1
        #seg[end:self.points_per_patch] = 2
        # seg[end:self.points_per_patch,0] = 0
        # seg[end:self.points_per_patch,1] = 0
        # seg[end:self.points_per_patch,2] = 1

        if self.dim_pts==6:
            patch_pts = torch.cat((patch_pts, point_normals), 1)

        patch_pts_dist = gauss_fcn(patch_pts_dist)

        return (patch_pts,) + (patch_normal,) + (seg,) + (patch_pts_dist,)


    def __len__(self):
        return sum(self.shape_patch_count)


    # translate global (dataset-wide) point index to shape index & local (shape-wide) point index
    def shape_index(self, index):
        shape_patch_offset = 0
        shape_ind = None
        for shape_ind, shape_patch_count in enumerate(self.shape_patch_count):
            if index >= shape_patch_offset and index < shape_patch_offset + shape_patch_count:
                shape_patch_ind = index - shape_patch_offset
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count

        return shape_ind, shape_patch_ind

    # load shape from a given shape index
    def load_shape_by_index(self, shape_ind):
        point_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.xyz')
        normals_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.normals') 

        pidx_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.pidx') if self.sparse_patches else None
        return load_shape(point_filename, normals_filename, pidx_filename, self.dim_pts)



import argparse
def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--indir', type=str, default='../../data/pclouds', help='input folder (point clouds)')
    parser.add_argument('--trainset', type=str, default='validationset_no_noise.txt', help='training set file name')
    parser.add_argument('--testset', type=str, default='validationset_no_noise.txt', help='test set file name')

    parser.add_argument('--points_per_patch', type=int, default=[512, 256], help='max. number of points per patch')
    parser.add_argument('--knn', type=int, default=True, help='k nearest neighbors.')
    parser.add_argument('--patch_radius', type=float, default=[0.05, 0.03], nargs='+', help='patch radius in multiples of the shape\'s bounding box diagonal, multiple values for multi-scale.')   
    parser.add_argument('--patch_point_count_std', type=float, default=0.1, help='standard deviation of the number of points in a patch')
    parser.add_argument('--patches_per_shape', type=int, default=100, help='number of patches sampled from each shape in an epoch')
    
    parser.add_argument('--batchSize', type=int, default=512, help='input batch size')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=10, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')

    parser.add_argument('--seed', type=int, default=3627474, help='manual seed')
    parser.add_argument('--identical_epochs', type=int, default=False, help='use same patches in each epoch, mainly for debugging')

    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_arguments()
    train_dataset = PointcloudPatchDataset(
        root=opt.indir,
        #shape_list_filename=opt.trainset,
        shape_list_filename=opt.testset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        dim_pts=6,
        knn = opt.knn,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        cache_capacity=opt.cache_capacity)

    train_datasampler = SequentialShapeRandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    train_enum = enumerate(train_dataloader, 0)
    for train_batchind, data in train_enum:
        points = data[0]#这时的point是64*512*3的类型
        points = points.transpose(2, 1)
        target = data[1]
        mask = data[2]
    