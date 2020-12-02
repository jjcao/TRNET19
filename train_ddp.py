from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import argparse
import os
import sys
import random
import math
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter # https://github.com/lanpa/tensorboard-pytorch

from dataset import PointcloudPatchDataset, RandomPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler
from dsac import DSAC, WDSAC, MSDSAC, MoEDSAC

# CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py
# nproc_per_node 参数指定为当前主机创建的进程数。一般设定为当前主机的 GPU 数量

def parse_arguments():
    parser = argparse.ArgumentParser()
    # 注意这个参数，必须要以这种形式指定，即使代码中不使用。因为 launch 工具默认传递该参数
    parser.add_argument("--local_rank", type=int, default=0)

    # naming / file handling
    parser.add_argument('--name', type=str, default='k512_s007_std5_sumd_pt32_pl32_dist_c_ddp', help='training run name')
    parser.add_argument('--patch_radius', type=float, default=[0.07], nargs='+', help='patch radius in multiples of the shape\'s bounding box diagonal, multiple values for multi-scale.')
    #parser.add_argument('--patch_radius', type=float, default=[0.05, 0.03, 0.02], nargs='+', help='patch radius in multiples of the shape\'s bounding box diagonal, multiple values for multi-scale.')
    parser.add_argument('--knn', type=int, default=True, help='k nearest neighbors.')
    parser.add_argument('--decoder', type=str, default='PointPredNet', help='PointPredNet, PointGenNet')
    parser.add_argument('--use_mask', type=int, default=False, help='use point mask')
    parser.add_argument('--share_pts_stn', type=int, default=True, help='')

    #parser.add_argument('--gpu_idx', type=str, default='1,2,3', help='set < 0 to use CPU')
    parser.add_argument('--gpu_idx', type=int, default=1, help='set < 0 to use CPU')
    #parser.add_argument('--refine', type=str, default='', help='refine model at this patch')
    parser.add_argument('--refine', type=str, default='../../data/dsacmodels/k512_s007_nostd_sumd_pt32_pl32_num_model.pth', help='refine model at this patch')
    #parser.add_argument('--expert_refine', type=str, default=[''], help='refine model at this patch')

    parser.add_argument('--batchSize', type=int, default=256, help='input batch size: 64*|gpu| for 512 points')
    #parser.add_argument('--point_tuple', type=int, default=1, help='use n-tuples of points as input instead of single points')
    parser.add_argument('--points_per_patch', type=int, default=[512], nargs='+', help='max. number of points per patch')
    #parser.add_argument('--points_per_patch', type=int, default=[512, 512, 512], nargs='+', help='max. number of points per patch')
    parser.add_argument('--in_points_dim', '-ipdim', type=int, default=3, help='3 for position, 6 for position + normal, ')

    parser.add_argument('--desc', type=str, default='training for single-scale normal estimation.', help='description')
    parser.add_argument('--indir', type=str, default='/data/pclouds', help='input folder (point clouds)')
    parser.add_argument('--indir2', type=str, default='../../data/results/s003_nostd_sumd_pt32_pl32_dist_c', help='input folder (point clouds)')
    parser.add_argument('--outdir', type=str, default='../../data/dsacmodels', help='output folder (trained models)')
    parser.add_argument('--logdir', type=str, default='../../data/dsaclogs', help='training log folder')
    parser.add_argument('--trainset', type=str, default='trainingset_whitenoise.txt', help='training set file name')
    parser.add_argument('--testset', type=str, default='validationset_whitenoise.txt', help='test set file name')
    # parser.add_argument('--trainset', type=str, default='validationset_no_noise.txt', help='training set file name')
    # parser.add_argument('--testset', type=str, default='validationset_no_noise.txt', help='test set file name')
    parser.add_argument('--saveinterval', type=int, default='10', help='save model each n epochs')

    # training parameters 
    parser.add_argument('--patch_point_count_std', type=float, default=0.5, help='standard deviation of the number of points in a patch')
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--seed', type=int, default=3627474, help='manual seed')
    parser.add_argument('--identical_epochs', type=int, default=False, help='use same patches in each epoch, mainly for debugging')


    parser.add_argument('--opti', type=str, default='SGD', help='optimizer, SGD or Adam')
    # lr = 0.0001 & momentum = 0.9 for SGD in PCPNet; lr = 0.001 for Adam in 3dcoded,
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')

    
    #parser.add_argument('--use_pca', type=int, default=False, help='Give both inputs and ground truth in local PCA coordinate frame')
    parser.add_argument('--normal_loss', type=str, default='ms_euclidean', help='Normal loss type:\n'
                        'ms_euclidean: mean square euclidean distance\n'
                        'ms_oneminuscos: mean square 1-cos(angle error)')
    # model hyperparameters
    parser.add_argument('--outputs', type=str, nargs='+', default=['unoriented_normals'], help='outputs of the network, a list with elements of:\n'
                        'unoriented_normals: unoriented (flip-invariant) point normals\n'
                        'oriented_normals: oriented point normals\n'
                        'max_curvature: maximum curvature\n'
                        'min_curvature: mininum curvature')
    parser.add_argument('--use_point_stn', type=int, default=True, help='use point spatial transformer')  
    parser.add_argument('--use_feat_stn', type=int, default=True, help='use feature spatial transformer')
    parser.add_argument('--sym_op', type=str, default='sumd', help='symmetry operation: max, sum, sumd (sum_dist), aved (ave_dist) ')

    ##### RANSAC hyperparameters
    parser.add_argument('--generate_points_num', '-gpnum', type=int, default=32, help='number of points output form the net')
    parser.add_argument('--generate_points_dim', '-gpdim', type=int, default=3, help='dim of points output form the net: 4 for pts + weight')
    parser.add_argument('--hypotheses', '-hyps', type=int, default=32, help='number of planes hypotheses sampled for each patch')
    
    # two parameters if score with sum of error distance: 
    #   p[0] for sigma^2 of gaussian used in the soft inlier count. 
    # three parameters if socre with inliner count:
    #   p[0] for threshold used in the soft inlier count, 
    #   p[1] for scaling factor within the sigmoid of the soft inlier count'
    # common parameter: p[end] for scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution
    parser.add_argument('--inlier_params', '-ip', type=float, default=[0.01, 0.5], help='RANSAC scorer with inlier distance') 
    #parser.add_argument('--inlier_params', '-ip', type=float, default=[0.1, 100, 0.5], help='RANSAC scorer with inlier count') 

    return parser.parse_args()

def train(opt):
    # 1) 配置每个进程的gpu
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print('stage 1 passed')
    
    # 2）初始化
    # colored console output
    green = lambda x: '\033[92m' + x + '\033[0m'
    blue = lambda x: '\033[94m' + x + '\033[0m'

    log_dirname = os.path.join(opt.logdir, opt.name)
    params_filename = os.path.join(opt.outdir, '%s_params.pth' % (opt.name))
    model_filename = os.path.join(opt.outdir, '%s_model.pth' % (opt.name))
    desc_filename = os.path.join(opt.outdir, '%s_description.txt' % (opt.name))

    if local_rank == 0:
        if os.path.exists(log_dirname) or os.path.exists(model_filename):
            if os.path.exists(log_dirname):
                shutil.rmtree(os.path.join(opt.logdir, opt.name))

    # if os.path.exists(log_dirname) or os.path.exists(model_filename):
    #     if os.path.exists(log_dirname):
    #         shutil.rmtree(os.path.join(opt.logdir, opt.name))
    print('stage 2 passed')

    # 3) 封装之前要把模型移到对应的gpu
    if len(opt.points_per_patch) == 1:
        if opt.generate_points_dim == 1:
            dsac = WDSAC(
                hyps = opt.hypotheses,
                inlier_params = opt.inlier_params,
                patch_radius = opt.patch_radius, 
                decoder = opt.decoder,
                use_mask=opt.use_mask,
                dim_pts = opt.in_points_dim,
                num_gpts = opt.generate_points_num,
                dim_gpts = opt.generate_points_dim, 
                points_per_patch=opt.points_per_patch[0],
                sym_op=opt.sym_op,
                normal_loss = opt.normal_loss, seed = opt.seed, device = device,
                use_point_stn=opt.use_point_stn, use_feat_stn=opt.use_feat_stn
                )
        else:
            dsac = DSAC(
                hyps = opt.hypotheses,
                inlier_params = opt.inlier_params,
                patch_radius = opt.patch_radius, 
                decoder = opt.decoder,
                use_mask=opt.use_mask,
                dim_pts = opt.in_points_dim,
                num_gpts = opt.generate_points_num,
                dim_gpts = opt.generate_points_dim, 
                points_per_patch=opt.points_per_patch[0],
                sym_op=opt.sym_op,
                normal_loss = opt.normal_loss, seed = opt.seed, device = device,
                use_point_stn=opt.use_point_stn, use_feat_stn=opt.use_feat_stn
                )
    else:
        dsac = MoEDSAC(
            hyps = opt.hypotheses, 
            inlier_params = opt.inlier_params,     
            patch_radius = opt.patch_radius,     
            share_pts_stn = opt.share_pts_stn,           
            decoder = opt.decoder,
            use_mask=opt.use_mask,
            dim_pts = opt.in_points_dim,
            num_gpts = opt.generate_points_num,
            points_per_patch=opt.points_per_patch,
            sym_op=opt.sym_op,
            normal_loss = opt.normal_loss, seed = opt.seed, device = device,
            use_point_stn=opt.use_point_stn, use_feat_stn=opt.use_feat_stn
            )

    dsac.to(device)

    if torch.cuda.device_count() > 1:
        if opt.refine != '': # if the refine model is not from ddp
            dsac.load_state_dict( torch.load(opt.refine, map_location=device) )  
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 3) 封装
        dsac = torch.nn.parallel.DistributedDataParallel(dsac,
                                                        device_ids=[local_rank],
                                                        output_device=local_rank)
        # if opt.refine != '': # if the refine model is not from ddp?
        #     dsac.load_state_dict( torch.load(opt.refine, map_location=device) )                                                       
    else:
        if opt.refine != '':
            dsac.load_state_dict(torch.load(opt.refine))    

    print('stage 3 passed')

    # 4）使用DistributedSampler
    train_dataset = PointcloudPatchDataset(
        root=opt.indir,
        root_in=opt.indir2,
        shape_list_filename=opt.trainset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        dim_pts = opt.in_points_dim,
        knn = opt.knn,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        cache_capacity=opt.cache_capacity)

    train_datasampler = RandomPointcloudPatchSampler(
        train_dataset,
        patches_per_shape=opt.patches_per_shape,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=DistributedSampler(train_datasampler),
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    test_dataset = PointcloudPatchDataset(
        root=opt.indir,
        root_in=opt.indir2,
        shape_list_filename=opt.testset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        dim_pts = opt.in_points_dim,
        knn = opt.knn,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        cache_capacity=opt.cache_capacity)

    test_datasampler = RandomPointcloudPatchSampler(
        test_dataset,
        patches_per_shape=opt.patches_per_shape,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs)

    test_dataloader = DataLoader(
        test_dataset,
        sampler=DistributedSampler(test_datasampler),
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))    

    # keep the exact training shape names for later reference
    opt.train_shapes = train_dataset.shape_names
    opt.test_shapes = test_dataset.shape_names

    print('training set: %d patches (in %d batches) - test set: %d patches (in %d batches)' %
          (len(train_datasampler), len(train_dataloader), len(test_datasampler), len(test_dataloader)))

    try:
        os.makedirs(opt.outdir)
    except OSError:
        pass

    if local_rank == 0:
        train_writer = SummaryWriter(os.path.join(log_dirname, 'train'))
        test_writer = SummaryWriter(os.path.join(log_dirname, 'test'))

    lrate = opt.lr
    if opt.opti == 'SGD':
        optimizer = optim.SGD(dsac.parameters(), lr=lrate, momentum=opt.momentum)
    else:
        optimizer = optim.Adam(dsac.parameters(), lr=lrate)  
    
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.1) # milestones in number of optimizer iterations


    train_num_batch = len(train_dataloader)
    test_num_batch = len(test_dataloader)
    # save parameters
    torch.save(opt, params_filename)

    # save description
    with open(desc_filename, 'w+') as text_file:
        print(opt.desc, file=text_file)


    for epoch in range(opt.nepoch):
        train_batchind = -1
        train_fraction_done = 0.0
        train_enum = enumerate(train_dataloader, 0)

        test_batchind = -1
        test_fraction_done = 0.0
        test_enum = enumerate(test_dataloader, 0)

        for train_batchind, data in train_enum:
            # set to training mode
            dsac.train()

            points = data[0]#这时的point是64*512*3的类型
            target = data[1]
            mask = data[2]
            dist = data[3]

            points = points.transpose(2, 1)
            points = points.to(device)          
            target = target.to(device)
            mask = mask.to(device)
            dist = dist.to(device)
            
            # zero gradients
            #optimizer.zero_grad()

            exp_loss, top_loss, pred, pts, mask_p, patch_rot, _ = dsac(points, target, dist)
            loss = exp_loss.mean()

            # backpropagate through entire network to compute gradients of loss w.r.t. parameters
            loss.backward()

            # parameter optimization step
            optimizer.step()

            train_fraction_done = (train_batchind+1) / train_num_batch

            # print info and update log file
            print('[%s %d: %d/%d] %s tloss: %f loss: %f Top Loss:%f' % (opt.name, epoch, train_batchind, train_num_batch-1, green('train'),loss, exp_loss.mean().item(),top_loss.mean()))
            if local_rank == 0:
                x1 = (epoch + train_fraction_done) * train_num_batch * opt.batchSize
                train_writer.add_scalars('loss', {'meanLoss':exp_loss.mean().item(),
                                            'topLoss':top_loss.mean().item()}, x1)

            while test_fraction_done <= train_fraction_done and test_batchind+1 < test_num_batch:

                # set to evaluation mode
                dsac.eval()

                test_batchind, data = next(test_enum)

                points = data[0]#这时的point是64*512*3的类型
                target = data[1]
                mask = data[2]
                dist = data[3]

                points = points.transpose(2, 1)
                points = points.to(device)          
                target = target.to(device)
                mask = mask.to(device)
                dist = dist.to(device)

                # forward pass
                with torch.no_grad():
                    exp_loss, top_loss, pred, pts, mask_p, patch_rot, _ = dsac(points, target, dist)

                loss = exp_loss.mean()

                test_fraction_done = (test_batchind+1) / test_num_batch

                # print info and update log file
                print('[%s %d: %d/%d] %s tloss: %f loss: %f Top Loss:%f' 
                    % (opt.name, epoch, train_batchind, train_num_batch-1, 
                    blue('test'),loss, exp_loss.mean().item(),top_loss.mean()))
                if local_rank == 0:
                    x1 = (epoch + test_fraction_done) * train_num_batch * opt.batchSize              
                    # test_writer.add_scalar('loss', exp_loss.mean().item(), x1)   
                    test_writer.add_scalars('loss', {'meanLoss':exp_loss.mean().item(),
                                                'topLoss':top_loss.mean().item()}, x1)

        # update learning rate
        scheduler.step()
        # save model, overwriting the old model
        if local_rank == 0:
            if epoch % opt.saveinterval == 0 or epoch == opt.nepoch-1:
                print('model saved')
                torch.save(dsac.state_dict(), model_filename)

        # save model in a separate file in epochs 0,5,10,50,100,500,1000, ...
        if epoch % (5 * 10**math.floor(math.log10(max(2, epoch-1)))) == 0 or epoch % 100 == 0 or epoch == opt.nepoch-1:
            torch.save(dsac.state_dict(), os.path.join(opt.outdir, '%s_model_%d.pth' % (opt.name, epoch)))

if __name__ == '__main__':
    torch.distributed.init_process_group(backend="nccl") #NCCL 是目前最快的后端，且对多进程分布式（Multi-Process Single-GPU）支持极好，可用于单节点以及多节点的分布式训练。

    opt = parse_arguments()
    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)
    print("Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    train(opt)