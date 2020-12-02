import torch
import torch.nn.functional as F
import numpy as np
import random
import utils
import math
import torch
import torch.nn as nn
#import torchsnooper

from pcpnet import PCPNet, MSPCPNet
import os

def compute_loss(pred, target, normal_loss='ms_euclidean'):

    # if patch_rot is not None:
    #     # transform predictions with inverse transform
    #     # since we know the transform to be a rotation (QSTN), the transpose is the inverse
    #     target = torch.bmm(target, patch_rot.transpose(2, 1))#.squeeze(1)

    if normal_loss == 'ms_euclidean':
        loss = torch.min((pred-target).pow(2).sum(2), (pred+target).pow(2).sum(2))#* output_loss_weight
    elif normal_loss == 'ms_oneminuscos':
        loss = (1-torch.abs(utils.cos_angle(pred, target))).pow(2)#* output_loss_weight
    else:
        raise ValueError('Unsupported loss type: %s' % (normal_loss))

    return loss

def sample_hyp(pts, hyps, idx_combi, device):
    '''
    Calculate a plane hypothesis from 3 random points.
    hyps: plane num
    pt_weight: [B, |pts|]
    '''
    batchsize = pts.size(0)
    tries = 10
    while tries:
        # select three*plane_num random points   
        tmp = torch.randint(0, idx_combi.size(0), (hyps*batchsize,))
        #plane_weight = torch.ones(self.idx_combi.size(0), dtype=torch.float).to(device)
        #tmp = torch.multinomial(plane_weight, hyps*batchsize, replacement=False) # 1552 slower than randint
        idx = idx_combi[tmp,:].view(batchsize, hyps*3, 1)
        index=torch.cat((idx,idx,idx),2)

        pts_sample = torch.gather(pts, 1, index)
        pts_sample = pts_sample.view(pts.size(0),hyps,3,3)# plane_num*3*3
        pts_sample = pts_sample.transpose(3,2)
        planes_sample = utils.pts_to_plane(pts_sample,hyps)

        nozeros = 4-torch.eq(planes_sample,0).sum(2)

        if (len(torch.nonzero(nozeros)) == (batchsize*hyps)) : break
        elif tries == 1:              #循环10次后还不行
            #找到无效平面的位
            tmp = nozeros - nozeros
            nozeros = torch.where(nozeros==0, torch.full_like(nozeros, 1), tmp)
            idx = torch.nonzero(nozeros, as_tuple=True)
            planes_sample[idx[0], idx[1], :] = 1 #填充为4个1
        
        tries-=1

    return planes_sample 

def sample_hyp_old(pts, hyps, rng, device):
    '''
    Calculate a plane hypothesis  from 3 random points.

    '''
    batchsize = pts.size(0)
    plane_num = hyps
    while 1: # 可能死循环， dead,当生成点云很差的话
        # select three*plane_num random points
        index=torch.stack([torch.from_numpy(np.stack(rng.choice(pts.size(1), 3, replace=False) for _ in range(hyps)).reshape(-1)) for _ in range(batchsize)]).to(device)#.long()

        index = index.view(batchsize,hyps*3,1)
        nindex=torch.cat((index,index,index),2)

        pts_sample=torch.gather(pts, 1, nindex)
        pts_sample = pts_sample.view(pts.size(0),plane_num,3,3)# plane_num*3*3
        pts_sample =pts_sample.transpose(3,2)
        planes_sample = utils.pts_to_plane(pts_sample,plane_num)

        nozeros= 4-torch.eq(planes_sample,0).sum(2)

        if (len(torch.nonzero(nozeros)) == (batchsize*plane_num)) : break

    return planes_sample  # True indicates success

def gauss_fcn(x, mu=0, sigma2=0.12):
    tmp = -(x - mu)**2 / (2 * sigma2)
    return torch.exp(tmp)

class Scorer_dist():
    def __init__(self, inlier_sigma2=0.01):
        self.inlier_sigma2 = inlier_sigma2
        
    def __call__(self, pts, planes):
        '''
            planes: normalized planes
        '''
        # point plane distances
        batchsize=pts.size(0)
        dists = torch.abs(planes[:,:,0].view(batchsize,-1,1)*pts[:,:,0].view(batchsize,1,-1)+planes[:,:,1].view(batchsize,-1,1)*pts[:,:,1].view(batchsize,1,-1)+planes[:,:,2].view(batchsize,-1,1)*pts[:,:,2].view(batchsize,1,-1)+planes[:,:,3].view(batchsize,-1,1))#.cuda()
        # tmp = torch.sqrt(planes[:,:,0].view(batchsize,-1,1)**2+planes[:,:,1].view(batchsize,-1,1)**2+planes[:,:,2].view(batchsize,-1,1)**2)
        # dists = dists / tmp
        score = gauss_fcn(dists, 0, self.inlier_sigma2)      
        score = score.sum(2)

        return score#, dists

class Scorer_inlier_count():
    '''
    Soft inlier count for a given plane and a given set of points.
    '''
    def __init__(self, inlier_thresh=0.1, inlier_beta=100):
        self.inlier_thresh = inlier_thresh
        self.inlier_beta = inlier_beta

    def __call__(self, pts, planes):
        '''
            planes: normalized planes
        '''
        # point plane distances
        batchsize=pts.size(0)

        dists = torch.abs(planes[:,:,0].view(batchsize,-1,1)*pts[:,:,0].view(batchsize,1,-1)+planes[:,:,1].view(batchsize,-1,1)*pts[:,:,1].view(batchsize,1,-1)+planes[:,:,2].view(batchsize,-1,1)*pts[:,:,2].view(batchsize,1,-1)+planes[:,:,3].view(batchsize,-1,1))
        dists = torch.sigmoid( self.inlier_beta * (self.inlier_thresh - dists) )

        score = dists.sum(2)

        return score#, dists

class DSAC(nn.Module):
    '''
    Differentiable RANSAC to robustly fit planes.

    dim_gpts=3: for standard DSAC, which allows differentiation w.r.t. to observations. 
                DSAC does not model observation selection, i.e. not allows differentiation w.r.t. observation selection.
    dim_gpts=4: for NG-RANSAC, learn observation sampling weights. 在用pcpnet生成64点，同时学习每点的采样权重，根据top32权重，选择32点，进入RANSAC过程。
    
    todo: backward mean loss or top loss?
    todo: 使用inliner count做自监督训练？那只能在原始点云上做吧。否则生成的点云只要落在某个平面上就好了。
    todo: 除坐标外，输入距离等信息？

    Refer to:
        DSAC - differentiable RANSAC for camera localization. CVPR, 2017: train scorer in Ransac.
        Learning less is more-6d camera localization via 3d surface regression. CVPR, 2018: use inliner count as scorer.
        Neural-guided ransac: Learningwhere to sample model hypotheses. ICCV, 2019: learn observation sampling weights; inliner count做自监督训练。
    '''

    def __init__(self, hyps=32, inlier_params=[0.01, 0.5], 
                patch_radius=[0.05], decoder='PointPredNet', use_mask=True, 
                dim_pts=3, num_gpts=32, dim_gpts=3, points_per_patch=512,                       
                sym_op='max', ith = 0, 
                use_point_stn=True, use_feat_stn=True,                
                device=0, normal_loss= 'ms_euclidean', seed = 3627474):
        '''
        Constructor.

        hyps -- number of planes hypotheses sampled for each patch
        inlier_thresh -- threshold used in the soft inlier count, 
        inlier_beta -- scaling factor within the sigmoid of the soft inlier count
        inlier_alpha -- scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)

        '''
        super(DSAC, self).__init__()
        self.hyps = hyps

        if len(inlier_params)==2:
            self.scorer = Scorer_dist(inlier_params[0])         
        else:
            self.scorer = Scorer_inlier_count(inlier_params[0], inlier_params[1])
        self.inlier_alpha = inlier_params[-1]
        
        self.normal_loss = normal_loss 
        self.use_point_stn = use_point_stn
        self.device = device

        torch.manual_seed(seed)    
        gpts_idx = torch.tensor([i for i in range(num_gpts)])  
        self.idx_combi = torch.combinations(gpts_idx, 3).to(device)    
        #self.plane_weight = torch.ones(self.idx_combi.size(0), dtype=torch.float).to(device) 

        if decoder == 'PointPredNet':
            if len(patch_radius) == 1:
                self.pcpnet = PCPNet(num_pts=points_per_patch, dim_pts=dim_pts, num_gpts=num_gpts, dim_gpts=dim_gpts,
                                    use_point_stn=use_point_stn, use_feat_stn=use_feat_stn, device=device,
                                    b_pred = True, use_mask=use_mask, sym_op=sym_op, ith = ith)
            else:
                self.pcpnet = MSPCPNet(num_scales=len(patch_radius), 
                                    num_points=points_per_patch, dim_pts=dim_pts, num_gpts=num_gpts, dim_gpts=dim_gpts,
                                    use_point_stn=use_point_stn, use_feat_stn=use_feat_stn, 
                                    use_mask=use_mask,sym_op=sym_op)
        # elif: decoder == 'WeightPredNet':
        #     self.mask_dim = 1
        #     self.pcpnet = PCPNet(num_pts=points_per_patch, dim_pts=dim_pts, num_gpts=num_gpts,
        #                             use_point_stn=use_point_stn, use_feat_stn=use_feat_stn, device=device,
        #                             b_pred = True, use_mask=False, mask_dim = self.mask_dim, sym_op=sym_op, ith = ith)
        else:
            self.pcpnet = PCPNet(num_pts=points_per_patch, dim_pts=dim_pts, num_gpts=num_gpts, dim_gpts=dim_gpts, 
                            use_point_stn=use_point_stn, use_feat_stn=use_feat_stn, device=device,
                            b_pred = False, use_mask=use_mask,sym_op=sym_op, ith = ith)

    def forward(self, x, target, dist=None, gfeat=None, patch_rot=None):
        '''
        Perform robust, differentiable plane fitting according to DSAC.

        Returns the expected loss of choosing a good plane hypothesis which can be used for backprob.

        '''
        batchsize=x.size(0)
        pts, patch_rot, _, mask, glob_feat = self.pcpnet(x, dist, gfeat, patch_rot)
        # if tpts.shape[2] > 3:
        #     pts = tpts[:,:,0:3]
        #     pt_weight = tpts[:,:,3] # [-1, +1] # +1
        #     _, idx = torch.topk(pt_weight, self.num_gpts_used, dim=1, largest=True, sorted=False, out=None) 
        #     nidx = idx.view(batchsize, self.num_gpts_used, 1)
        #     nindex=torch.cat((nidx,nidx,nidx),2)
        #     pts = torch.gather(pts, 1, nindex)
        # else:
        #     pts = tpts

        if patch_rot is not None: # 逆旋转生成点，所以就不用逆旋转GT normal了。实际应用的时候，也方便。
            pts=torch.bmm(pts,patch_rot.transpose(2, 1)) 
     
        # === step 1: select  planes ===========================
        planes = sample_hyp(pts, self.hyps, self.idx_combi, self.device)

        tmp = torch.sqrt(planes[:,:,0]**2+planes[:,:,1]**2+planes[:,:,2]**2).view(batchsize, self.hyps, 1)
        tmp=torch.cat((tmp,tmp,tmp,tmp),2)
        planes = planes/tmp

        # === step 2: score hypothesis using soft inlier count ====
        score = self.scorer(pts, planes)
     
        # === step 3: calculate the loss ===========================
        loss = compute_loss(
            planes[:,:,0:3],
            target.view(batchsize,1,3),
            normal_loss = self.normal_loss
        )
        
        maxindex = torch.argmax(score,1).long().view(batchsize,1)
        top_loss=torch.gather(loss, 1, maxindex).view(batchsize)

        maxindex = maxindex.view(batchsize,1,1)
        nindex=torch.cat((maxindex,maxindex,maxindex),2)
        pred = torch.gather(planes[:,:,0:3], 1, nindex).squeeze() # best/top normal
        
        # === step 4: calculate the expectation ===========================
        #softmax distribution from hypotheses scores            
        score = F.softmax(self.inlier_alpha * score, 1)   
        exp_loss = torch.sum(loss * score,1)
        
        return exp_loss, top_loss, pred, pts, mask, patch_rot, glob_feat

class WDSAC(nn.Module):
    '''
    Differentiable RANSAC to robustly fit planes.

    pick observations from learnt weights.
    todo: default pytorch implementation can not backward, since the operation, 
            selecting input points using topk weights, does not contain gradient info. 
          We have to refer to the paper's c++ implementation.

    Refer to:
        Neural-guided ransac: Learningwhere to sample model hypotheses. ICCV, 2019: learn observation sampling weights; inliner count做自监督训练。
    '''

    def __init__(self, points_per_patch=256, patch_radius = [0.05], dim_pts=3, num_gpts=128, dim_gpts=1, 
                hyps=64, inlier_params=[0.01, 0.5],                          
                use_mask=False,  sym_op='max', ith = 0, 
                use_point_stn=True, use_feat_stn=True, decoder = 'PointPredNet',               
                device=0, normal_loss= 'ms_euclidean', seed = 3627474):
        '''
        Constructor.

        hyps -- number of planes hypotheses sampled for each patch
        inlier_thresh -- threshold used in the soft inlier count, 
        inlier_beta -- scaling factor within the sigmoid of the soft inlier count
        inlier_alpha -- scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)

        '''
        super(WDSAC, self).__init__()
        self.hyps = hyps
        self.num_gpts = num_gpts

        if len(inlier_params)==2:
            self.scorer = Scorer_dist(inlier_params[0])         
        else:
            self.scorer = Scorer_inlier_count(inlier_params[0], inlier_params[1])
        self.inlier_alpha = inlier_params[-1]
        
        self.normal_loss = normal_loss 
        self.use_point_stn = use_point_stn
        self.device = device

        torch.manual_seed(seed)    
        gpts_idx = torch.tensor([i for i in range(num_gpts)])  
        self.idx_combi = torch.combinations(gpts_idx, 3).to(device)    
        #self.plane_weight = torch.ones(self.idx_combi.size(0), dtype=torch.float).to(device) 

        self.wpcp = PCPNet(num_pts=points_per_patch, dim_pts=dim_pts, num_gpts=points_per_patch, dim_gpts=1, 
                            use_point_stn=use_point_stn, 
                            use_feat_stn=use_feat_stn, device=device,
                            b_pred = True, use_mask=False, sym_op=sym_op, ith = 0)

    def forward(self, pts, target, dist=None, gfeat=None, patch_rot=None):
        '''
        Perform robust, differentiable plane fitting according to DSAC.

        Returns the expected loss of choosing a good plane hypothesis which can be used for backprob.

        '''
        batchsize = pts.size(0)
        pt_weight, _, _, _, _ = self.wpcp(pts, dist, gfeat=None, patch_rot=None)
        pts = pts.transpose(2, 1)

        #gradients = torch.zeros(pts.shape[0], pts.shape[1], 1)
        _, idx = torch.topk(pt_weight, self.num_gpts, dim=1, largest=True, sorted=False) 
        #gradients[idx] += 1

        nidx = idx.view(batchsize, self.num_gpts, 1)
        nindex=torch.cat((nidx,nidx,nidx),2)
        gpts = torch.gather(pts, 1, nindex)
     
        # === step 1: select  planes ===========================
        planes = sample_hyp(gpts, self.hyps, self.idx_combi, self.device)

        tmp = torch.sqrt(planes[:,:,0]**2+planes[:,:,1]**2+planes[:,:,2]**2).view(batchsize, self.hyps, 1)
        tmp=torch.cat((tmp,tmp,tmp,tmp),2)
        planes = planes/tmp

        # === step 2: score hypothesis using soft inlier count ====
        score = self.scorer(pts, planes)
     
        # === step 3: calculate the loss ===========================
        loss = compute_loss(
            planes[:,:,0:3],
            target.view(batchsize,1,3),
            normal_loss = self.normal_loss
        )
        
        maxindex = torch.argmax(score,1).long().view(batchsize,1)
        top_loss=torch.gather(loss, 1, maxindex).view(batchsize)

        maxindex = maxindex.view(batchsize,1,1)
        nindex=torch.cat((maxindex,maxindex,maxindex),2)
        pred = torch.gather(planes[:,:,0:3], 1, nindex).squeeze() # best/top normal
        
        # === step 4: calculate the expectation ===========================
        #softmax distribution from hypotheses scores            
        score = F.softmax(self.inlier_alpha * score, 1)
        exp_loss = torch.sum(loss * score,1)
        
        return exp_loss, top_loss, pred, gpts, _, _, _


class MSDSAC(nn.Module):
    def __init__(self, hyps=32, inlier_params=[0.01, 0.5],
            normal_loss= 'ms_euclidean', seed = 3627474, 
            device=0, patch_radius = [0.05, 0.03, 0.02],
            use_point_stn=True, use_feat_stn=True, decoder = 'PointPredNet',
            use_mask=True, num_gpts=32, points_per_patch=[512, 256, 128], sym_op='max'):
        super(MSDSAC, self).__init__() 

        self.use_mask = use_mask
        self.points_per_patch = points_per_patch
        self.dsacs = nn.ModuleList()
        for s, rad in enumerate(patch_radius): 
            if s > 0:
                use_point_stn = False

            dsac = DSAC(        
                hyps = hyps, 
                inlier_params = inlier_params,
                normal_loss = normal_loss,
                seed = seed, 
                device= device,
                patch_radius = [patch_radius[s]],
                use_point_stn = use_point_stn, 
                use_feat_stn = use_feat_stn,
                decoder = decoder,
                use_mask = use_mask,
                num_gpts = num_gpts,
                points_per_patch = points_per_patch[s],
                sym_op = sym_op,
                ith = s
                )
            self.dsacs.append(dsac)         

    def forward(self, x, target, dist=None):
        exp_loss = torch.zeros( len(self.dsacs) )

        start_idx = 0
        end_idx = self.points_per_patch[0]
        y = x[:, :, start_idx:end_idx]
        if self.use_mask:
            mask_p = torch.zeros(x.shape[0], sum(points_per_patch), 3)
            exp_loss[0], top_loss, pred, pts, mask_p[:, start_idx:end_idx, :], patch_rot, gfeat = \
                                                            self.dsacs[0](y, target, dist[:,start_idx:end_idx,:])
            for i in range(1, len(self.dsacs)): 
                start_idx = start_idx + self.points_per_patch[i-1]
                end_idx = start_idx + self.points_per_patch[i]

                y = x[:, :, start_idx:end_idx]
                exp_loss[i], top_loss, pred, pts, mask_p[:, start_idx:end_idx, :], _, gfeat =\
                                            self.dsacs[i](y, target, dist[:,start_idx:end_idx,:], gfeat, patch_rot)
        else:
            exp_loss[0], top_loss, pred, pts, mask_p, patch_rot, gfeat = self.dsacs[0](y, target, dist[:,start_idx:end_idx,:])
            for i in range(1, len(self.dsacs)): 
                start_idx = start_idx + self.points_per_patch[i-1]
                end_idx = start_idx + self.points_per_patch[i]

                y = x[:, :, start_idx:end_idx]
                exp_loss[i], top_loss, pred, pts, mask_p, _, gfeat = \
                                            self.dsacs[i](y, target, dist[:,start_idx:end_idx,:], gfeat, patch_rot)
        
        #exp_loss = torch.sum(exp_loss).mean()
        return exp_loss, top_loss, pred, pts, mask_p, patch_rot, None

class MoEDSAC(nn.Module):
    ''' 
        否定：方案1: 为了能refine 三个尺度的网络，不从manager取global feature，这样也不能用manager旋转stn1，进一步的拓展不方便呀。
        方案1，从显存、速度和精度上都不如方案2。 方案3也不如2，每个expert不应该受到干扰。

        方案1: manger用所有尺度的input，独立的stn1和stn2，预测权重；expert各自完全独立，average normal loss with learnt weights.
        方案2: manger用所有尺度的input，使用manager提供的stn1，独立RANSAC，average normal loss with learnt weights
        方案3: 在2的基础上，每个expert同时接收manager的global feature作为输入。

    Refer to:
        Nesti-Net: Normal Estimation for Unstructured 3D Point Clouds using Convolutional Neural Networks, cvpr19: .
    '''
    def __init__(self, hyps=32, inlier_params=[0.01, 0.5],
            normal_loss= 'ms_euclidean', seed = 3627474, 
            device=0, patch_radius = [0.05, 0.03, 0.02],
            use_point_stn=True, share_pts_stn = False, use_feat_stn=True, decoder = 'PointPredNet',
            use_mask=True, dim_pts=3, num_gpts=32, dim_gpts=3, points_per_patch=[512, 256, 128], sym_op='max'): #, expert_refine=['']):
        super(MoEDSAC, self).__init__() 
        self.use_mask = use_mask
        self.points_per_patch = points_per_patch
        self.normal_loss = normal_loss
        self.device = device
        self.share_pts_stn = share_pts_stn

        #self.manager = PCPNet(num_pts=sum(points_per_patch), dim_pts=dim_pts, num_gpts=len(points_per_patch), dim_gpts=1, 
        self.manager = PCPNet(num_pts=points_per_patch[0], dim_pts=dim_pts, num_gpts=len(points_per_patch), dim_gpts=1, 
                            use_point_stn=use_point_stn, 
                            use_feat_stn=use_feat_stn, device=device,
                            b_pred = True, use_mask=False, sym_op='max', ith = 0)

        # if share_pts_stn:
        #     use_point_stn = False

        self.dsacs = nn.ModuleList()
        for s, rad in enumerate(patch_radius): 
            dsac = DSAC(        
                hyps = hyps, 
                inlier_params = inlier_params,
                normal_loss = normal_loss,
                seed = seed, 
                device= device,
                patch_radius = [patch_radius[s]],
                points_per_patch = points_per_patch[s],
                use_point_stn = use_point_stn, 
                use_feat_stn = use_feat_stn,
                decoder = decoder,
                use_mask = use_mask,
                dim_pts = dim_pts, 
                num_gpts = num_gpts,  
                dim_gpts = dim_gpts,             
                sym_op = sym_op,
                ith = s
                )

            if self.share_pts_stn:
                dsac.pcpnet.feat.use_point_stn = False

            self.dsacs.append(dsac) 
            
    def refine(self, expert_models=[]):  
        i = 0
        for name in expert_models:
            try:
                self.dsacs[i].load_state_dict(torch.load(name))
            except Exception as e: 
                print(e)
                #print("maybe Unexpected key(s) in state_dict for STN1, it is ok.")  

            i+=1

    def forward(self, x, target, dist=None):
        batchsize = x.size(0)
        num_experts = len(self.dsacs)
        
        start_idx = 0
        end_idx = start_idx + self.points_per_patch[0]
        y = x[:, :, start_idx:end_idx]
        weights, patch_rot, _, _, gfeat = self.manager(y, dist, gfeat=None, patch_rot=None)

        loss = torch.zeros(batchsize, num_experts).to(self.device)
        #top_loss = torch.zeros(batchsize).to(self.device)
        top_loss = torch.zeros(batchsize, num_experts).to(self.device)
        pred = torch.zeros(batchsize, num_experts, 3).to(self.device)       
        
        if self.use_mask:
            mask_p = torch.zeros(x.shape[0], sum(points_per_patch), 3)
            for i in range(0, num_experts): 
                end_idx = start_idx + self.points_per_patch[i]

                y = x[:, :, start_idx:end_idx]
                loss[:,i], top_loss[:,i], pred[:,i,:], pts, mask_p[:, start_idx:end_idx, :], _, _ =\
                                            self.dsacs[i](y, target, dist[:,start_idx:end_idx,:], gfeat=None, patch_rot=patch_rot)
                start_idx = end_idx
        else:
            for i in range(0, num_experts): 
                end_idx = start_idx + self.points_per_patch[i]

                y = x[:, :, start_idx:end_idx]
                loss[:,i], top_loss[:,i], pred[:,i,:], pts, mask_p, _, _ = \
                                            self.dsacs[i](y, target, dist[:,start_idx:end_idx,:], gfeat=None, patch_rot=patch_rot)

                start_idx = end_idx
        
        maxidx = torch.argmax(weights,1).long().view(batchsize,1,1)
        maxidx = torch.cat((maxidx, maxidx, maxidx), 2)
        pred_normal = torch.gather(pred, 1, maxidx).view(batchsize, 3)

        # weights: (b, num_scales) 
        exp_loss = torch.sum(loss * weights,1).mean()
        top_loss = torch.sum(top_loss * weights,1).mean() 

        return exp_loss, top_loss, pred_normal, pts, mask_p, patch_rot, None

class ESAC(nn.Module):
    ''' 
        todo: wait success of WDSAC.

        方案4: 在MoEDSAC的方案2的基础上，根据manager学到的权重，分配expert的Ransac采样，剩下操作和DSAC相同。这样可以把生成平面的数量扩展为：32*|experts|?
    Refer to:
        Expert Sample Consensus Applied to Camera Re-Localization. ICCV, 2019: .
    '''
    def __init__(self, hyps=32, inlier_params=[0.01, 0.5], use_mask=False, num_gpts=32, dim_gpts=3, 
            points_per_patch=[512, 256, 128], patch_radius = [0.05, 0.03, 0.02], sym_op='max',
            use_point_stn=True, share_pts_stn = False, use_feat_stn=True, decoder = 'PointPredNet',
            normal_loss= 'ms_euclidean', seed = 3627474, device=0): #, expert_refine=['']):
        super(ESAC, self).__init__() 

        self.points_per_patch = points_per_patch
        self.normal_loss = normal_loss
        self.device = device

        self.hyps = hyps
        self.dim_gpts = dim_gpts
        if self.dim_gpts == 4:
            raise ValueError('Unsupported dimensions: %d' % (self.dim_gpts)) # self.num_gpts_used = num_gpts // 2
        else:
            self.num_gpts_used = num_gpts

        if len(inlier_params)==2:
            self.scorer = Scorer_dist(inlier_params[0])         
        else:
            self.scorer = Scorer_inlier_count(inlier_params[0], inlier_params[1])
        self.inlier_alpha = inlier_params[-1]


        torch.manual_seed(seed)    
        gpts_idx = torch.tensor([i for i in range(self.num_gpts_used)])  
        self.idx_combi = torch.combinations(gpts_idx, 3).to(device)    
        self.plane_weight = torch.ones(self.idx_combi.size(0), dtype=torch.float).to(device) 

        self.manager = PCPNet(num_pts=sum(points_per_patch), dim_pts=3, num_gpts=len(points_per_patch), dim_gpts=1, 
                            use_point_stn=use_point_stn, 
                            use_feat_stn=use_feat_stn, device=device,
                            b_pred = True, use_mask=False, sym_op='sum', ith = 0)

        if share_pts_stn:
            use_point_stn = False

        self.pcpnets = nn.ModuleList()
        for s, rad in enumerate(patch_radius): 
            pn = PCPNet(num_pts=points_per_patch[s], dim_pts=3, num_gpts=num_gpts, dim_gpts=dim_gpts,
                                    use_point_stn=use_point_stn, use_feat_stn=use_feat_stn, device=device,
                                    b_pred = True, use_mask=use_mask, sym_op=sym_op, ith = s)
            self.pcpnets.append(pn) 

    def forward(self, x, target, dist=None):
        batchsize=x.size(0)
        num_experts = len(self.pcpnets)
        weights, patch_rot, _, _, gfeat = self.manager(x, dist, gfeat=None, patch_rot=None)

        start_idx = 0
        tpts = torch.ones(batchsize, num_experts*self.num_gpts_used, 3).to(self.device) 
        for i in range(0, num_experts): 
            end_idx = start_idx + self.points_per_patch[i]

            y = x[:, :, start_idx:end_idx]
            tpts[:,i*self.num_gpts_used:(i+1)*self.num_gpts_used,:], _, _, mask, _ = \
                                self.pcpnets[i](y, dist[:,start_idx:end_idx,:], gfeat=None, patch_rot=patch_rot)
            start_idx = end_idx

        if patch_rot is not None: # 逆旋转生成点，所以就不用逆旋转GT normal了。实际应用的时候，也方便。
            tpts=torch.bmm(tpts, patch_rot.transpose(2, 1)) 

        # === step 1: select  planes ===========================
        num_hyps = torch.round(weights * self.hyps).int() 
        planes = sample_hyp_experts_weighted() 
   
        # === step 2: score hypothesis using soft inlier count ====
        score = self.scorer(pts, planes)
     
        # === step 3: calculate the loss ===========================
        loss = compute_loss(
            planes[:,:,0:3],
            target.view(batchsize,1,3),
            normal_loss = self.normal_loss
        )
        
        maxindex = torch.argmax(score,1).long().view(batchsize,1)
        top_loss=torch.gather(loss, 1, maxindex).view(batchsize)

        maxindex = maxindex.view(batchsize,1,1)
        nindex=torch.cat((maxindex,maxindex,maxindex),2)
        pred = torch.gather(planes[:,:,0:3], 1, nindex).squeeze() # best/top normal
        
        # === step 4: calculate the expectation ===========================
        #softmax distribution from hypotheses scores            
        score = F.softmax(self.inlier_alpha * score, 1)   
        exp_loss = torch.sum(loss * score,1)
        
        return exp_loss, top_loss, pred, pts, mask, patch_rot, glob_feat

def test_sample_hyp(tries_in = 9999):
    gpu_idx = -3
    device = torch.device("cpu" if gpu_idx < 0 else "cuda:%d" % gpu_idx)
    batchsize = 2

    rng = np.random.RandomState(3627474)
    num_gpts = 256
    pts = torch.rand(batchsize, num_gpts, 3) 
    hyps = 32
    gpts_idx = torch.tensor([i for i in range(num_gpts)])   
    combi = torch.combinations(gpts_idx, 3).to(device)
    
    t = time.process_time()
    tries = tries_in
    while tries:   
        tmp = torch.randint(0, combi.size(0), (hyps*batchsize,))
        index = combi[tmp,:].view(batchsize, hyps*3, 1)
        if tries:
            tries -= 1
        else:
            break
    print(time.process_time() - t)

    plane_weight = torch.ones(combi.size(0), dtype=torch.float).to(device)
    t = time.process_time()
    tries = tries_in
    while tries:   
        tmp = torch.multinomial(plane_weight, hyps*batchsize, replacement=False)
        index = combi[tmp,:].view(batchsize, hyps*3, 1)
        if tries:
            tries -= 1
        else:
            break
    print(time.process_time() - t)

    t = time.process_time()
    tries = tries_in
    while tries:
        index=torch.stack([torch.from_numpy(np.stack(rng.choice(pts.size(1), 3, replace=False) for _ in range(hyps)).reshape(-1)) for _ in range(batchsize)])
        index = index.view(batchsize,hyps*3,1)
        if tries:
            tries -= 1
        else:
            break
    print(time.process_time() - t)
def test_DSAC():
    gpu_idx = -3
    device = torch.device("cpu" if gpu_idx < 0 else "cuda:%d" % gpu_idx)
    batchsize = 4
    dim_pts = 6
    points_per_patch = [256]
    patch_radius = [0.07]

    sim_pts = torch.rand(batchsize, 3, sum(points_per_patch))  
    sim_normals = torch.rand(batchsize, 3, sum(points_per_patch))   
    sim_dists = torch.rand(batchsize, sum(points_per_patch), 1)  
    patch_normal = torch.rand(batchsize, 1, 3)
    if dim_pts == 6:
        sim_pts = torch.cat((sim_pts, sim_normals), 1)

    dsac = DSAC(        
        hyps = 64, 
        inlier_params= [0.01, 0.5], 
        normal_loss = 'ms_euclidean',
        seed = 3627474, 
        device= device,
        patch_radius = patch_radius,
        decoder = 'PointPredNet', # PointPredNet, PointGenNet
        use_mask=True,
        dim_pts = dim_pts, 
        num_gpts = 128,
        dim_gpts = 3,
        points_per_patch=points_per_patch[0],
        sym_op='sum'
        )
    # sim_pts = sim_pts.transpose(2, 1)
    t = time.process_time()
    exp_loss, top_loss, _, pts, mask_p, patch_rot, _ = dsac(sim_pts, patch_normal, sim_dists)
    print(time.process_time() - t)

    print(pts.size())    

def test_WDSAC():
    gpu_idx = -3
    device = torch.device("cpu" if gpu_idx < 0 else "cuda:%d" % gpu_idx)
    batchsize = 4
    points_per_patch = [256]
    patch_radius = [0.07]

    sim_pts = torch.rand(batchsize, sum(points_per_patch), 3)   
    sim_dists = torch.rand(batchsize, sum(points_per_patch), 1)  
    sim_normal = torch.rand(batchsize, 1, 3)

    dsac = WDSAC(        
        hyps = 64, 
        inlier_params= [0.01, 0.5], 
        normal_loss = 'ms_euclidean',
        seed = 3627474, 
        device= device,
        patch_radius = patch_radius,
        decoder = 'PointPredNet', # PointPredNet, PointGenNet
        use_mask=False,
        num_gpts = 128,
        dim_gpts = 3,
        points_per_patch=points_per_patch[0],
        sym_op='sum'
        )
    sim_pts = sim_pts.transpose(2, 1)
    t = time.process_time()
    exp_loss, top_loss, _, pts, mask_p, patch_rot, _ = dsac(sim_pts, sim_normal, sim_dists)
    print(time.process_time() - t)

    print(pts.size()) 

def test_MSDSAC():
    gpu_idx = -3
    device = torch.device("cpu" if gpu_idx < 0 else "cuda:%d" % gpu_idx)
    batchsize = 2
    points_per_patch = [512, 256]
    patch_radius = [0.05, 0.03]
    
    sim_pts = torch.rand(batchsize, sum(points_per_patch), 3)   
    sim_dists = torch.rand(batchsize, sum(points_per_patch), 1)  
    sim_normal = torch.rand(batchsize, 1, 3)

    dsac = MSDSAC(        
        hyps = 32, 
        inlier_params= [0.01, 0.5], 
        normal_loss = 'ms_euclidean',
        seed = 3627474, 
        device= device,
        patch_radius = patch_radius,
        decoder = 'PointPredNet', # PointPredNet # PointGenNet
        use_mask=False,
        num_gpts = 32,
        points_per_patch=points_per_patch,
        sym_op='sum'
        )

    sim_pts = sim_pts.transpose(2, 1)
    exp_loss, top_loss, _, pts, mask_p, patch_rot, _ = dsac(sim_pts, sim_normal, sim_dists)
    
    print(pts.size())   

def test_MoEDSAC():

    gpu_idx = 2
    device = torch.device("cpu" if gpu_idx < 0 else "cuda:%d" % gpu_idx)
    batchsize = 4
    points_per_patch = [256, 256]
    patch_radius = [0.05, 0.03]
    expert_models = ['../../data/dsacmodels/k256_s007_nostd_sumd_pt32_pl32_num_model.pth',
    '../../data/dsacmodels/k256_s007_nostd_sumd_pt32_pl32_num_model.pth']
    
    sim_pts = torch.rand(batchsize, sum(points_per_patch), 3).to(device) 
    sim_dists = torch.rand(batchsize, sum(points_per_patch), 1).to(device)
    sim_normal = torch.rand(batchsize, 1, 3).to(device)

    dsac = MoEDSAC(        
        hyps = 32, 
        inlier_params= [0.01, 0.5], 
        normal_loss = 'ms_euclidean',
        seed = 3627474, 
        device= device,
        patch_radius = patch_radius,
        decoder = 'PointPredNet', # PointPredNet # PointGenNet
        use_mask=False,
        num_gpts = 32,
        points_per_patch=points_per_patch,
        sym_op='sum'
        )
    dsac.to(device)
    dsac.refine(expert_models)

    sim_pts = sim_pts.transpose(2, 1)
    exp_loss, top_loss, _, pts, mask_p, patch_rot, _ = dsac(sim_pts, sim_normal, sim_dists)
    
    print(pts.size())    

def test_ESAC():

    gpu_idx = -3
    device = torch.device("cpu" if gpu_idx < 0 else "cuda:%d" % gpu_idx)
    batchsize = 4
    points_per_patch = [512, 256]
    patch_radius = [0.05, 0.03]
    
    sim_pts = torch.rand(batchsize, sum(points_per_patch), 3)   
    sim_dists = torch.rand(batchsize, sum(points_per_patch), 1)  
    sim_normal = torch.rand(batchsize, 1, 3)

    dsac = ESAC(        
        hyps = 32*len(patch_radius), 
        inlier_params= [0.01, 0.5], 
        normal_loss = 'ms_euclidean',
        seed = 3627474, 
        device= device,
        patch_radius = patch_radius,
        decoder = 'PointPredNet', # PointPredNet # PointGenNet
        use_mask=False,
        num_gpts = 32,
        points_per_patch=points_per_patch,
        sym_op='sum'
        )

    sim_pts = sim_pts.transpose(2, 1)
    exp_loss, top_loss, _, pts, mask_p, patch_rot, _ = dsac(sim_pts, sim_normal, sim_dists)
    
    print(pts.size())    


if __name__ == '__main__':
    import time
    #test_sample_hyp(tries_in = 99)
    #test_DSAC()
    #test_WDSAC()
    #test_MSDSAC()
    test_MoEDSAC()
    #test_ESAC()

    