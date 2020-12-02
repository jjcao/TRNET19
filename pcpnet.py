# https://github.com/bbaaii/DRNE19

from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import utils


class STN(nn.Module):
    def __init__(self, num_scales=1, num_points=512, dim=3, sym_op='max'):
        super(STN, self).__init__()

        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.dim*self.dim)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        if self.num_scales > 1:
            self.fc0 = nn.Linear(1024*self.num_scales, 1024)
            self.bn0 = nn.BatchNorm1d(1024)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            x_scales = x.new_empty(x.size(0), 1024*self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*1024:(s+1)*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, 1024*self.num_scales)

        if self.num_scales > 1:
            x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.dim, dtype=x.dtype, device=x.device).view(1, self.dim*self.dim).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.dim, self.dim)
        return x
class MASK(nn.Module):
    def __init__(self, num_points=512, dim=3 ):
        super(MASK, self).__init__()
        self.dim = dim
            #self.num_scales = num_scales
    
        self.num_points = num_points
    
        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.conva = torch.nn.Conv1d(576, 256, 1)
        #self.convb = torch.nn.Conv1d(512, 256, 1)
        self.convc = torch.nn.Conv1d(256, 64, 1)
        self.convd = torch.nn.Conv1d(64, 1, 1)
        self.bna = nn.BatchNorm1d(256)
        #self.bnb = nn.BatchNorm1d(256)
        self.bnc = nn.BatchNorm1d(64)
    
        #self.fc1 = nn.Linear(1024, 512)
        #self.fc2 = nn.Linear(512,num_points)
            #self.fc3 = nn.Linear(256, 4)
    
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        #self.bn4 = nn.BatchNorm1d(512)
            #self.bn5 = nn.BatchNorm1d(256)
    
    
    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))#3 to 64
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))# 64 to 128
    
        x = F.relu(self.bn3(self.conv3(x)))# 128 to 512
    
        x = self.mp1(x)
    
        x = x.view(-1, 512)
        x = x.view(-1, 512, 1).repeat(1, 1, n_pts)
        x = torch.cat([x, pointfeat], 1)
        x = F.relu(self.bna(self.conva(x)))
        #x = F.relu(self.bnb(self.convb(x)))
        x = F.relu(self.bnc(self.convc(x)))
        x = F.relu(self.convd(x))

        #x = F.relu(self.bn4(self.fc1(x)))
    
            #x = F.relu(self.bn5(self.fc2(x)))
    
        #x = F.relu(self.fc2(x))
        x=x.view(batchsize,1,self.num_points)
        return x#,x2
class QSTN(nn.Module):
    def __init__(self, num_scales=1, num_points=512, dim=3, sym_op='max'):
        super(QSTN, self).__init__()

        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        if self.num_scales > 1:
            self.fc0 = nn.Linear(1024*self.num_scales, 1024)
            self.bn0 = nn.BatchNorm1d(1024)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            x_scales = x.new_empty(x.size(0), 1024*self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*1024:(s+1)*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, 1024*self.num_scales)

        if self.num_scales > 1:
            x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # add identity quaternion (so the network can output 0 to leave the point cloud identical)
        iden = x.new_tensor([1, 0, 0, 0])
        x = x + iden

        # convert quaternion to rotation matrix
        x = utils.batch_quat_to_rotmat(x)

        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_scales=1, num_points=512, dim_pts=3, use_point_stn=True, use_feat_stn=True, 
                    sym_op='max', use_pointfeat=False):
        super(PointNetfeat, self).__init__()
        self.num_points = num_points
        self.dim_pts = dim_pts
        self.num_scales = num_scales
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.sym_op = sym_op
        self.use_pointfeat = use_pointfeat
        self.bottleneck_size = 1024

        if self.use_point_stn:
            self.stn1 = QSTN(num_scales=self.num_scales, num_points=num_points, dim=dim_pts, sym_op=self.sym_op)

        if self.use_feat_stn:
            self.stn2 = STN(num_scales=self.num_scales, num_points=num_points, dim=64, sym_op=self.sym_op)

        self.conv0a = torch.nn.Conv1d(dim_pts, 64, 1)
        self.conv0b = torch.nn.Conv1d(64, 64, 1)
        self.bn0a = nn.BatchNorm1d(64)
        self.bn0b = nn.BatchNorm1d(64)
        self.conv1 = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.bottleneck_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.bottleneck_size)

        if self.num_scales > 1:
            self.conv4 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size*self.num_scales, 1)
            self.bn4 = nn.BatchNorm1d(self.bottleneck_size*self.num_scales)

        if self.sym_op == 'max':
            self.mp1 = torch.nn.MaxPool1d(num_points)
        else: # various sum or ave
            self.mp1 = None
        # else:
        #     raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))

        # self.conv_1 = torch.nn.Conv1d(self.bottleneck_size*2, self.bottleneck_size, 1)
        # self.conv_2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        # self.bn_1 = nn.BatchNorm1d(self.bottleneck_size)
        # self.bn_2 = nn.BatchNorm1d(self.bottleneck_size)

        # if self.sym_op == 'max':
        #     self.mp2 = torch.nn.MaxPool1d(num_points)
        # elif self.sym_op == 'sum':
        #     self.mp2 = None
        # else:
        #     raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))

    def forward(self, pts, patch_rot=None, dist=None):
        '''
            pts(B, 3, 512) or (B, 6, 512)
        '''
        x = pts
        # input transform
        trans = patch_rot
        # if trans is not None or self.use_point_stn:
        #     # from tuples to list of single points
        #     x = x.view(x.size(0), self.dim_pts, -1)
        if self.use_point_stn:
            trans = self.stn1(x)
            
        if trans is not None:
            if self.dim_pts == 6:
                x = torch.cat((x[:,0:3,:],x[:,3:6,:]), 2)
                x = x.transpose(2, 1)
                x = torch.bmm(x, trans)
                x = x.transpose(2, 1)  
                x = torch.cat((x[:,:,0:self.num_points],x[:,:,self.num_points:(self.num_points+self.num_points)]), 1) 
            else:            
                x = x.transpose(2, 1)
                x = torch.bmm(x, trans)
                x = x.transpose(2, 1)

            x = x.contiguous().view(x.size(0), self.dim_pts, -1) # self.dim_pts = 3

        # mlp (64,64)    
        x = F.relu(self.bn0a(self.conv0a(x)))
        x = F.relu(self.bn0b(self.conv0b(x)))

        # feature transform
        if self.use_feat_stn:
            trans2 = self.stn2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans2)
            x = x.transpose(2, 1)
        else:
            trans2 = None
             
        # mlp (64,128,1024)
        x = F.relu(self.bn1(self.conv1(x)))
        if self.use_pointfeat:
            pointfeat = x
        else:
            pointfeat = None
            
        x = F.relu(self.bn2(self.conv2(x)))
        
        x = self.bn3(self.conv3(x))
        
        # mlp (1024,1024*num_scales)
        if self.num_scales > 1:
            x = self.bn4(self.conv4(F.relu(x)))

        # symmetric max operation over all points
        # if self.num_scales == 1:
        if self.sym_op == 'max':
            x = self.mp1(x)
        elif self.sym_op.startswith('sum'):
            if self.sym_op.endswith('d'):
                x = x * dist.transpose(1,2)
            x = torch.sum(x, 2, keepdim=True)   
        elif self.sym_op.startswith('ave'):
            if self.sym_op.endswith('d'):
                x = x * dist.transpose(1,2)
            x = torch.mean(x, 2, keepdim=True) 
        else:
            raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))

        # else:
        #     x_scales = x.new_empty(x.size(0), self.bottleneck_size*self.num_scales, 1)#x_scales = x.new_empty(x.size(0), 1024*self.num_scales**2, 1)
        #     if self.sym_op == 'max':
        #         for s in range(self.num_scales):
        #             x_scales[:, s*self.bottleneck_size:(s+1)*self.bottleneck_size, :] = \
        #                 self.mp1(x[:, s*self.bottleneck_size:(s+1)*self.bottleneck_size, s*self.num_points:(s+1)*self.num_points])
        #     elif self.sym_op == 'sum':
        #         for s in range(self.num_scales):
        #             x_scales[:, s*self.bottleneck_size:(s+1)*self.bottleneck_size, :] = \
        #                 torch.sum(x[:, s*self.bottleneck_size:(s+1)*self.bottleneck_size, s*self.num_points:(s+1)*self.num_points], 2, keepdim=True)
        #     elif self.sym_op == 'ave':
        #         for s in range(self.num_scales):
        #             x_scales[:, s*self.bottleneck_size:(s+1)*self.bottleneck_size, :] = \
        #                 torch.mean(x[:, s*self.bottleneck_size:(s+1)*self.bottleneck_size, s*self.num_points:(s+1)*self.num_points], 2, keepdim=True)
        #     else:
        #         raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))
        #     x = x_scales
        

        gfeat = x.contiguous().view(-1, self.bottleneck_size*self.num_scales) #  x = x.contiguous().view(-1, 1024*self.num_scales**2)

        ########################
        # symmetric max operation over all points: local + global feature 
        # no obvious effect, close it for running time, temporary
        ########################
        # x = x.view(-1, self.bottleneck_size*self.num_scales, 1).repeat(1, 1, self.num_points*self.num_scales)
        # x = torch.cat([pointfvals, x], 1)
        # x = F.relu(self.bn_1(self.conv_1(x)))
        # x = F.relu(self.bn_2(self.conv_2(x)))
        # if self.sym_op == 'max':
        #     x = self.mp2(x)
        # elif self.sym_op == 'sum':
        #     if dist is not None:
        #         x = x * dist.transpose(1,2)
        #     x = torch.mean(x, 2, keepdim=True)
        # else:
        #     raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))
        # x = x.contiguous().view(-1, self.bottleneck_size*self.num_scales) 

        return gfeat, trans, trans2, pointfeat

class PointGenNet(nn.Module):
    def __init__(self, bottleneck_size = 1024):
        self.bottleneck_size = bottleneck_size
        super(PointGenNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)

    def forward(self, x):
        '''
            x (B, bottleneck_size, N)
        '''
        batchsize = x.size()[0]
        # print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x.transpose(1,2).contiguous()

class PointPredNet(nn.Module):
    def __init__(self, num_scales=1, num_gpts = 32, dim_gpts = 3): 
        super(PointPredNet, self).__init__()
        self.num_gpts = num_gpts
        self.dim_gpts = dim_gpts
        self.num_scales = num_scales

        # if num_scales > 1:
        #     self.fc0 = nn.Linear(num_scales, 1)
        #     self.bn0 = nn.BatchNorm1d(1024)
        #     self.do0 = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(1024*num_scales, 512)
        #self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_gpts*self.dim_gpts)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.do1 = nn.Dropout(p=0.3)
        self.do2 = nn.Dropout(p=0.3) 
        self.th = nn.Tanh()  

    def forward(self, x):
        # if self.num_scales > 1:
        #     x = x.transpose(1,2)
        #     x = self.fc0(x)
        #     x= x.view(x.size(0),-1)
        #     x = F.relu(self.bn0(x))
        #     x = self.do0(x)
        #     #x = x.transpose(1,2)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.do1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.do2(x)
        x = self.fc3(x)
        x = x.contiguous().view(-1, self.num_gpts, self.dim_gpts)    
        # x = torch.sigmoid(x)*2-1.0 # from 白 [-1, 1]
        x = self.th(x) # [-1, 1]
        # x = torch.sigmoid(x) # we need 0--1, since the 4 dimension may be a weight. 
        
        return x

class MaskPredNet(nn.Module):
    ''' 
    '''
    def __init__(self, num_pts=256, num_scales=1, pointfeat_dim=64, gfeat_dim=1024): 
        super(MaskPredNet, self).__init__()

        self.mask_dim = 2
        self.num_pts = num_pts
        self.num_scales = num_scales

        self.conva = torch.nn.Conv1d(pointfeat_dim + gfeat_dim*num_scales, gfeat_dim//2, 1) # local feature size 64
        self.convb = torch.nn.Conv1d(gfeat_dim//2, gfeat_dim//4, 1)
        self.convc = torch.nn.Conv1d(gfeat_dim//4, gfeat_dim//8, 1)
        self.convd = torch.nn.Conv1d(gfeat_dim//8, self.mask_dim, 1) # 2 is num_classes
        self.bna = nn.BatchNorm1d(gfeat_dim//2)
        self.bnb = nn.BatchNorm1d(gfeat_dim//4)
        self.bnc = nn.BatchNorm1d(gfeat_dim//8)

    def forward(self, pointfeat, gfeat):
        gfeat_dim = gfeat.size(1)

        gfeat = gfeat.view(-1, gfeat_dim*self.num_scales, 1).repeat(1, 1, self.num_pts*self.num_scales)
        x2 = torch.cat([pointfeat, gfeat], 1)
        x2 = F.relu(self.bna(self.conva(x2)))
        x2 = F.relu(self.bnb(self.convb(x2)))
        x2 = F.relu(self.bnc(self.convc(x2)))
        x2 = self.convd(x2)

        x2 = x2.transpose(2,1).contiguous()
        mask = F.log_softmax(x2, dim=-1)
        
        return mask

class WeightPredNet(nn.Module):
    ''' 
        learn weight from multiple globa features of different scale
    '''
    def __init__(self, num_weight): 
        super(WeightPredNet, self).__init__()

        self.fc1 = nn.Linear(1024, 512)
        #self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_weight)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.do1 = nn.Dropout(p=0.3)
        self.do2 = nn.Dropout(p=0.3) 
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.do1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.do2(x)
        x = self.fc3(x)
        x = self.sm(x)
        return x
       
class PCPNet(nn.Module):
    '''
        be_pre, True for several layers of FC, like pointnet, Faslse for using folding-net-like structure
    '''
    def __init__(self, num_pts=512, dim_pts=3, num_gpts=32, dim_gpts=3, 
                use_point_stn=True, use_feat_stn=True, device = '1',
                b_pred=True, use_mask=True,
                sym_op='max', ith = 0):
        super(PCPNet, self).__init__()
        self.ith = ith
        self.b_pred = b_pred
        self.use_mask = use_mask
        # if ith > 0:
        #     num_scales = 2 # ith + 1
        # else:
        num_scales = 1 
        
        self.use_pointfeat = False
        if self.b_pred:
            # if mask_dim == 1:
            #     self.decoder = MaskPredNet(mask_dim = mask_dim)
            #     self.use_pointfeat = True
            if dim_gpts == 1:
                self.decoder = WeightPredNet(num_weight = num_gpts)
            else: # dim_gpts == 3 or 4
                self.decoder = PointPredNet(num_scales = num_scales, num_gpts = num_gpts, dim_gpts= dim_gpts )
        else:
            self.decoder = PointGenNet(bottleneck_size = 2+1024*(ith+1))
            nx, ny = (4, 8) # 4,8; 8,8 
            xv, yv = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing='ij')
            grid = np.append(xv.reshape(-1,1),yv.reshape(-1,1),axis=1)
            self.grid = torch.tensor(grid).float()            
            #if torch.cuda.is_available(): self.grid = self.grid.to(device)
            self.grid = self.grid.to(device)

        if self.use_mask:
            self.mask_decoder = MaskPredNet(num_pts=num_pts, num_scales=num_scales)
            self.use_pointfeat = True

        self.feat = PointNetfeat(
            num_points=num_pts,
            dim_pts=dim_pts,
            num_scales=1,
            use_point_stn=use_point_stn,
            use_feat_stn=use_feat_stn,
            sym_op=sym_op,
            use_pointfeat=self.use_pointfeat)


    def forward(self, pts, dist=None, gfeat=None, patch_rot=None):
        '''
            pts(B, 3, 512)
        '''

        glob_feat, trans, trans2, point_feat = self.feat(pts, patch_rot, dist)
        if gfeat is not None:
            gfeat = torch.cat( (gfeat, glob_feat), 1).contiguous() # [Bx1024] + [Bx1024] => [Bx2048]
            # [Bxithx1024] + [Bx1024] => [Bx(ith+1)*1024]
            # x = torch.cat( (gfeat.view(x.size(0),self.ith,-1), x.view(x.size(0),1, -1)), 1).contiguous()
        else:
            gfeat = glob_feat
       
        if self.b_pred:
            gpts = self.decoder(gfeat)
        else:
            grid = self.grid.transpose(0,1).contiguous().unsqueeze(0).expand(gfeat.size(0), 2, -1)
            y = gfeat.unsqueeze(2).expand(gfeat.size(0), gfeat.size(1), grid.size(2)).contiguous()
            y = torch.cat( (grid, y), 1).contiguous() # y (B, 1024+2, 64)
            gpts = self.decoder(y)

        mask = None
        if self.use_mask:
            mask = self.mask_decoder(point_feat, glob_feat)
        return gpts, trans, trans2, mask, glob_feat

class MSPCPNet(nn.Module):
    def __init__(self, num_scales=2, num_points=512, num_gpts=32, 
                    use_point_stn=True, use_feat_stn=True, 
                    use_mask = False, 
                    sym_op='max'):
        super(MSPCPNet, self).__init__()
        self.num_points = num_points
        self.num_gpts = num_gpts

        self.feat = PointNetfeat(
            num_points=num_points,
            num_scales=num_scales,
            use_point_stn=use_point_stn,
            use_feat_stn=use_feat_stn,
            sym_op=sym_op,
            )

        self.decoder = PointPredNet(num_scales=num_scales, num_gpts = num_gpts)

    def forward(self, x):
        x, trans, trans2, mask = self.feat(x)
        glob_feat = x
        x = self.decoder(x)
        return x, trans, trans2, mask, glob_feat

if __name__ == '__main__':

    batch_size = 2
    dim_pts = 6
    points_per_patch = 4
    patch_radius = [0.05]
    patch_rot = torch.eye(3).view(1, 3*3).repeat(batch_size,1)
    patch_rot = patch_rot.contiguous().view(-1, 3, 3) 
    
    # patch_rot = torch.cat((ident,ident), dim=0)
    # patch_rot = torch.repeat_interleave(ident, repeats=batch_size, dim=0)

    if len(patch_radius) == 1:
        pcpnet = PCPNet(num_pts=points_per_patch, dim_pts=dim_pts, num_gpts=64, 
                        use_point_stn=True, use_feat_stn=True, 
                        b_pred=True, use_mask=False, sym_op='sum')
    else:
        pcpnet = MSPCPNet(num_scales=len(patch_radius), 
                    num_points=points_per_patch, num_gpts=64, 
                    use_point_stn=True, use_feat_stn=True, 
                    use_mask=True, sym_op='sum')
 
    sim_pts = torch.ones(batch_size, 3, points_per_patch*len(patch_radius)) 
    sim_normals = torch.zeros(batch_size, 3, points_per_patch*len(patch_radius))   
    sim_dists = torch.rand(batch_size, points_per_patch*len(patch_radius), 1)  

    if dim_pts == 6:
        sim_pts = torch.cat((sim_pts, sim_normals), 1)
    pts, patch_rot, _, mask, glob_feat = pcpnet(sim_pts, dist=sim_dists, patch_rot=patch_rot)
    print(pts.shape)