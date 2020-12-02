import torch

# quaternion a + bi + cj + dk should be given in the form [a,b,c,d]
def batch_quat_to_rotmat(q, out=None):

    batchsize = q.size(0)

    if out is None:
        out = q.new_empty(batchsize, 3, 3)

    # 2 / squared quaternion 2-norm
    s = 2/torch.sum(q.pow(2), 1)

    # coefficients of the Hamilton product of the quaternion with itself
    h = torch.bmm(q.unsqueeze(2), q.unsqueeze(1))

    out[:, 0, 0] = 1 - (h[:, 2, 2] + h[:, 3, 3]).mul(s)
    out[:, 0, 1] = (h[:, 1, 2] - h[:, 3, 0]).mul(s)
    out[:, 0, 2] = (h[:, 1, 3] + h[:, 2, 0]).mul(s)

    out[:, 1, 0] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
    out[:, 1, 1] = 1 - (h[:, 1, 1] + h[:, 3, 3]).mul(s)
    out[:, 1, 2] = (h[:, 2, 3] - h[:, 1, 0]).mul(s)

    out[:, 2, 0] = (h[:, 1, 3] - h[:, 2, 0]).mul(s)
    out[:, 2, 1] = (h[:, 2, 3] + h[:, 1, 0]).mul(s)
    out[:, 2, 2] = 1 - (h[:, 1, 1] + h[:, 2, 2]).mul(s)

    return out

def cos_angle(v1, v2):

    return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)

def pts_to_plane(q,hps):

    batchsize = q.size(0)

    #if out is None:
    # out = q.new_empty(batchsize,hps,4)
    
    out0=(q[:,:,2,2]-q[:,:,2,0])*(q[:,:,1,1]-q[:,:,1,0]) -(q[:,:,1,2]-q[:,:,1,0])*(q[:,:,2,1]-q[:,:,2,0])
    out1=(q[:,:,0,2]-q[:,:,0,0])*(q[:,:,2,1]-q[:,:,2,0]) -(q[:,:,2,2]-q[:,:,2,0])*(q[:,:,0,1]-q[:,:,0,0])
    out2=(q[:,:,1,2]-q[:,:,1,0])*(q[:,:,0,1]-q[:,:,0,0]) -(q[:,:,0,2]-q[:,:,0,0])*(q[:,:,1,1]-q[:,:,1,0])
    out3=-out0*q[:,:,0,0]-out1*q[:,:,1,0]-out2*q[:,:,2,0]
    out = torch.cat(( out0.view(batchsize,hps,1) , out1.view(batchsize,hps,1) , out2.view(batchsize,hps,1), out3.view(batchsize,hps,1)),2)
    # out=torch.stack(out)
    # out=out.transpose(1,0)

    return out

