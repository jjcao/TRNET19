from torch.utils.data.distributed import DistributedSampler

import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from dataset import PointcloudPatchDataset, SequentialPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler
from pcpnet import PCPNet, MSPCPNet

from dsac import DSAC, WDSAC, MSDSAC, MoEDSAC

# todo: how to load a checkpoint saved in a distributed training!!!!!
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 calculate.py --models s003_nostd_sumd_pt32_pl32_num_6dsac_ddp --distributed 1
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 calculate.py --models s003_nostd_sumd_pt32_pl32_num_6dsac --distributed 1
# python calculate.py --gpu_idx 0 --models moe_532_512_512_512_pred_sumd_num --distributed 0

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)

    # naming / file handling
    parser.add_argument('--indir', type=str, default='../data/pclouds', help='input folder (point clouds)')
    #parser.add_argument('--indir', type=str, default='/data/nyuv2_surfacenormal_metadata/pcloud_test_5000', help='input folder (point clouds)')
    parser.add_argument('--indir2', type=str, default='', help='input folder (point clouds)')
    parser.add_argument('--outdir', type=str, default='../results/pclouds/', help='output folder (estimated point cloud properties)')
    # parser.add_argument('--dataset', type=str, default='trainingset_whitenoise.txt', help='shape set file name')
    # parser.add_argument('--dataset', type=str, default='validationset_whitenoise.txt', help='shape set file name')
    parser.add_argument('--dataset', type=str, default='testset_all.txt', help='shape set file name')
    
    parser.add_argument('--modeldir', type=str, default='models', help='model folder')
    parser.add_argument('--models', type=str, default='k256_s007_nostd_sumd_pt32_pl32_num', help='names of trained models, can evaluate multiple models') 
    parser.add_argument('--distributed', type=int, default=False, help='.')
    parser.add_argument('--modelpostfix', type=str, default='_model.pth', help='model file postfix')
    parser.add_argument('--parmpostfix', type=str, default='_params.pth', help='parameter file postfix')
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')

    parser.add_argument('--sparse_patches', type=int, default=False, help='evaluate on a sparse set of patches, given by a .pidx file containing the patch center point indices.')
    parser.add_argument('--sampling', type=str, default='full', help='sampling strategy, any of:\n'
                        'full: evaluate all points in the dataset\n'
                        'sequential_shapes_random_patches: pick n random points from each shape as patch centers, shape order is not randomized')
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='number of patches evaluated in each shape (only for sequential_shapes_random_patches)')
    parser.add_argument('--seed', type=int, default=40938661, help='manual seed')
    parser.add_argument('--batchSize', type=int, default=0, help='batch size, if 0 the training batch size is used')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')

    return parser.parse_args()

def eval_pcpnet(opt):

    if opt.distributed:
        torch.distributed.init_process_group(backend="nccl")
        # 2） 配置每个进程的gpu
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        print('stage 2 passed')
    else:
        device = torch.device("cpu" if opt.gpu_idx < 0 else "cuda:%d" % opt.gpu_idx)

    opt.models = opt.models.split()
    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)

    for model_name in opt.models:

        print("Random Seed: %d" % (opt.seed))
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)

        model_filename = os.path.join(opt.modeldir, model_name+opt.modelpostfix)
        param_filename = os.path.join(opt.modeldir, model_name+opt.parmpostfix)

        # load model and training parameters
        trainopt = torch.load(param_filename)
        trainopt.indir2 = '../data2'
        if opt.distributed and local_rank == 0:
            print(trainopt)
            
        if opt.batchSize == 0:
            model_batchSize = trainopt.batchSize
        else:
            model_batchSize = opt.batchSize

        # get indices in targets and predictions corresponding to each output
        pred_dim = 0
        output_pred_ind = []
        # trainopt.outputs = {'unoriented_normals'}
        for o in trainopt.outputs:
            if o == 'unoriented_normals' or o == 'oriented_normals':
                output_pred_ind.append(pred_dim)
                pred_dim += 3
            
            else:
                raise ValueError('Unknown output: %s' % (o))

        if len(trainopt.points_per_patch) == 1:
            if trainopt.generate_points_dim == 1:
                regressor = WDSAC(
                    hyps = trainopt.hypotheses + 10,
                    inlier_params = trainopt.inlier_params,                
                    patch_radius = trainopt.patch_radius,
                    decoder = trainopt.decoder,
                    use_mask=trainopt.use_mask,
                    dim_pts = trainopt.in_points_dim,
                    num_gpts = trainopt.generate_points_num,
                    dim_gpts = trainopt.generate_points_dim, 
                    points_per_patch=trainopt.points_per_patch[0],
                    sym_op=trainopt.sym_op,
                    normal_loss = trainopt.normal_loss, seed = trainopt.seed, device = device,
                    use_point_stn=trainopt.use_point_stn, use_feat_stn=trainopt.use_feat_stn
                )
            else:
                regressor = DSAC(
                    hyps = trainopt.hypotheses + 10,
                    inlier_params = trainopt.inlier_params,                
                    patch_radius = trainopt.patch_radius,
                    decoder = trainopt.decoder,
                    use_mask=trainopt.use_mask,
                    dim_pts = trainopt.in_points_dim,
                    num_gpts = trainopt.generate_points_num,
                    dim_gpts = trainopt.generate_points_dim, 
                    points_per_patch=trainopt.points_per_patch[0],
                    sym_op=trainopt.sym_op,
                    normal_loss = trainopt.normal_loss, seed = trainopt.seed, device = device,
                    use_point_stn=trainopt.use_point_stn, use_feat_stn=trainopt.use_feat_stn
                )                
        else:
            regressor = MoEDSAC(
                hyps = trainopt.hypotheses + 10,
                inlier_params = trainopt.inlier_params,
                patch_radius = trainopt.patch_radius,
                share_pts_stn = trainopt.share_pts_stn,
                decoder = trainopt.decoder,
                use_mask=trainopt.use_mask,
                dim_pts = trainopt.in_points_dim,
                num_gpts = trainopt.generate_points_num,
                points_per_patch=trainopt.points_per_patch,
                sym_op=trainopt.sym_op,
                normal_loss = trainopt.normal_loss, seed = trainopt.seed, device = device,
                use_point_stn=trainopt.use_point_stn, use_feat_stn=trainopt.use_feat_stn
            )
            # if len(opt.expert_refine)>1:
            #     dsac.refine(opt.expert_refine)

        regressor.to(device)

        if opt.distributed:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
            regressor = torch.nn.parallel.DistributedDataParallel(regressor,
                                                        device_ids=[local_rank],
                                                        output_device=local_rank)
            #print(local_rank)
        
        state_dict = torch.load(model_filename, map_location=device)
        # print(state_dict.keys())
        # print(regressor.state_dict().keys())
        regressor.load_state_dict(state_dict)       
        print('stage 3 passed')
        
        regressor.eval()

        dataset = PointcloudPatchDataset(
            root=opt.indir, 
            root_in=trainopt.indir2,
            shape_list_filename=opt.dataset,
            patch_radius=trainopt.patch_radius,
            points_per_patch=trainopt.points_per_patch,
            dim_pts = trainopt.in_points_dim,
            knn = trainopt.knn,
            #patch_features=[],
            seed=opt.seed,
            cache_capacity=opt.cache_capacity
            )
        if opt.sampling == 'full':
            datasampler = SequentialPointcloudPatchSampler(dataset)
        elif opt.sampling == 'sequential_shapes_random_patches':
            datasampler = SequentialShapeRandomPointcloudPatchSampler(
                dataset,
                patches_per_shape=opt.patches_per_shape,
                seed=opt.seed,
                sequential_shapes=True,
                identical_epochs=False)
        else:
            raise ValueError('Unknown sampling strategy: %s' % opt.sampling)

        if opt.distributed:           
            dataloader = torch.utils.data.DataLoader(
                dataset,
                sampler=DistributedSampler(datasampler),
                batch_size=model_batchSize,
                num_workers=int(opt.workers))
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                sampler=datasampler,
                batch_size=model_batchSize,
                num_workers=int(opt.workers))


        shape_ind = 0
        shape_patch_offset = 0
        if opt.sampling == 'full':
            shape_patch_count = dataset.shape_patch_count[shape_ind]
        elif opt.sampling == 'sequential_shapes_random_patches':
            shape_patch_count = min(opt.patches_per_shape, dataset.shape_patch_count[shape_ind])
        else:
            raise ValueError('Unknown sampling strategy: %s' % opt.sampling)
        shape_properties = torch.zeros(shape_patch_count, 3, dtype=torch.float, device=device)

        # append model name to output directory and create directory if necessary
        model_outdir = os.path.join(opt.outdir, model_name)
        if not os.path.exists(model_outdir):
            os.makedirs(model_outdir)

        num_batch = len(dataloader)
        batch_enum = enumerate(dataloader, 0)
        for batchind, data in batch_enum:

            # get batch and upload to GPU
            points, target, _, dist = data
            points = points.transpose(2, 1)
            points = points.to(device)
            target = target.to(device)
            dist = dist.to(device)

            with torch.no_grad():
                exp_loss, top_loss, pred, pts, _ , _, _ = regressor(points, target, dist)

            print('[%s %d/%d] shape %s' % (model_name, batchind, num_batch-1, dataset.shape_names[shape_ind]))
            

            batch_offset = 0
            while batch_offset < pred.size(0):

                shape_patches_remaining = shape_patch_count-shape_patch_offset
                batch_patches_remaining = pred.size(0)-batch_offset

                # append estimated patch properties batch to properties for the current shape
                shape_properties[shape_patch_offset:shape_patch_offset+min(shape_patches_remaining, batch_patches_remaining), :] = pred[
                    batch_offset:batch_offset+min(shape_patches_remaining, batch_patches_remaining), :]

                batch_offset = batch_offset + min(shape_patches_remaining, batch_patches_remaining)
                shape_patch_offset = shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining)

                if shape_patches_remaining <= batch_patches_remaining:

                    # save shape properties to disk
                    prop_saved = [False]*len(trainopt.outputs)

                    # save normals
                    oi = [i for i, o in enumerate(trainopt.outputs) if o in ['unoriented_normals', 'oriented_normals']]
                    if len(oi) > 1:
                        raise ValueError('Duplicate normal output.')
                    elif len(oi) == 1:
                        oi = oi[0]
                        normal_prop = shape_properties[:, output_pred_ind[oi]:output_pred_ind[oi]+3]
                        
                        np.savetxt(os.path.join(model_outdir, dataset.shape_names[shape_ind]+'.normals'), normal_prop.cpu().numpy())
                        print('saved normals for ' + dataset.shape_names[shape_ind])
                        prop_saved[oi] = True

                    # save curvatures
                    

                    if not all(prop_saved):
                        raise ValueError('Not all shape properties were saved, some of them seem to be unsupported.')

                    # save point indices
                    if opt.sampling != 'full':
                        np.savetxt(os.path.join(model_outdir, dataset.shape_names[shape_ind]+'.idx'), datasampler.shape_patch_inds[shape_ind], fmt='%d')

                    # start new shape
                    if shape_ind + 1 < len(dataset.shape_names):
                        shape_patch_offset = 0
                        shape_ind = shape_ind + 1
                        if opt.sampling == 'full':
                            shape_patch_count = dataset.shape_patch_count[shape_ind]
                        elif opt.sampling == 'sequential_shapes_random_patches':
                            # shape_patch_count = min(opt.patches_per_shape, dataset.shape_patch_count[shape_ind])
                            shape_patch_count = len(datasampler.shape_patch_inds[shape_ind])
                        else:
                            raise ValueError('Unknown sampling strategy: %s' % opt.sampling)
                        shape_properties = shape_properties.new_zeros(shape_patch_count, pred_dim)


if __name__ == '__main__':
    eval_opt = parse_arguments()
    eval_pcpnet(eval_opt)
