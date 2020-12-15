#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
import numpy as np
from dep2def import depth2defocus
from functools import partial
# from awnet import pwc_5x5_sigmoid_bilinear   # cm:import AWnet model
# import pytorch_ssim
import pixel_estimator_with_weights
import cv2

# # In[2]:

# AWnet = pwc_5x5_sigmoid_bilinear.pwc_residual().cuda()
# AWnet.load_state_dict(torch.load('awnet/fs0_61_294481_0.00919393_dict.pkl'))

width = 1080 # img.shape[1]
f = 25
fn = 4
FoV_h = 10 * np.pi / 180
pp = 2 * f * np.tan(FoV_h / 2) / width  # pixel pitch in mm
gamma = 2.4
# use partial is recommended to set lens parameter
myd2d = partial(depth2defocus, f=f, fn=fn, pp=pp, r_step=1, inpaint_occlusion=False)  # this would fix f, fn, pp, and r_step

def t2n(tensor, isImage = True):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    narray = tensor.numpy()
    if isImage:
        if len(narray.shape) == 4:
            narray = narray.transpose(0, 2, 3, 1)
        elif len(narray.shape) == 3:
            narray = narray.transpose(1, 2, 0)
        else:
            #raise Exception("convertion error!")
            pass
    return narray

def n2t(narray, isImage = True, device = "cuda:0"):
    if isImage:
        if len(narray.shape) == 4:
            narray = narray.transpose(0, 3, 1, 2)
        elif len(narray.shape) == 3:
            narray = narray.transpose(2, 0, 1)  
        else:
            #raise Exception("convertion error!")
            pass
    tensor = torch.from_numpy(narray).float().to(device)
    return tensor


def get_parameter_number(net):
    '''
    print total and trainable number of params 
    '''
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def dfs_freeze(model):
    '''
    freeze the network
    '''
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


# In[3]:


def reconsLoss(J_est, J_gt):   
    '''
    Calculate loss (neg reward) of Reinforcement learning
    
    input: 
        J_est: (B, Seq, C, H, W) predicted image sequences
        J_gt: (B, Seq, C, H, W) ground truth image sequence

    output: 
        lossTensor: (B, 1)
            mse value for each sequence of images in minibatch.
    '''
    lossList = []

    for i in range(J_gt.size(0)):
#         lossList.append(2.5-torch.log10(4 / ((J_gt[i] - J_est[i])**2).mean()))
        lossList.append(F.mse_loss(J_gt[i], J_est[i]))
    lossTensor = torch.stack(lossList)
    #lossTensor = pytorch_ssim.ssim(J_gt/2+0.5, J_est/2+0.5) #torch.stack(lossList)
    return lossTensor

def depth_from_region(depthmap, loc):
    
    assert len(depthmap.shape) == 2
    assert len(loc.shape) == 1
    
    if loc.shape[0] == 2:
        H, W = depthmap.shape
        window_size =  min(H, W)//4

        x_l = int((loc[0]+1) * (H - window_size) / 2)
        y_l = int((loc[1]+1) * (W - window_size) / 2)
        x_r = int(min(H, x_l + window_size))
        y_r = int(min(W, y_l + window_size))

            #print("fun_depth_from_region: ({}, {})".format(x_l, y_l))

        if type(loc) == torch.Tensor:
            value = torch.mean(depthmap[x_l:x_r, y_l:y_r])
        else:
            value = np.mean(depthmap[x_l:x_r, y_l:y_r])
    else:
        value = np.clip(loc[0]*6000+4000, depthmap.min(), depthmap.max())
    return value

def color_region(tensors, locs):
    
    S, C, H, W = tensors.size()
    tensors_copy = tensors.clone()
    assert S == locs.size(0)
    if locs.size(1) == 2:
        for i in range(S):
            loc = locs[i]
            window_size =  min(H, W)//4
            x_l = int((loc[0]+1) * (H - window_size) / 2)
            y_l = int((loc[1]+1) * (W - window_size) / 2)
            x_r = int(min(H, x_l + window_size))
            y_r = int(min(W, y_l + window_size))
            tensors_copy[i][1:, x_l:x_r, y_l:y_r] = -1
            tensors_copy[i][0, x_l:x_r, y_l:y_r] = 1
    
    return tensors_copy


def getDefocuesImage(focusPos, J, dpt, threshold = 5e-2):
    '''
    Camera model. 
    Input: 
        focusPos Tensor(B, 1): current timestep focus position [-1, 1]
        J  Tensor (B, C, H, W): next time gt image [0, 1]
        dpt  Tensor (B, 1, H, W): J corresponding depth map [???]
    Output: 
        imageTensor (B, C, H, W): current timestep captured minibatch [0 1]
    '''

    imageTensor = []
    simAutofocusTensor = []
    is_cuda_tensor = focusPos.is_cuda
    if is_cuda_tensor:
        focusPos, J, dpt = focusPos.cpu(), J.cpu(), dpt.cpu()

    for i in range(J.size()[0]):
        J_np = t2n(J[i])
        J_np = ((J_np+1)*127.5).astype(np.uint8) # uint8
        dpt_np = dpt[i].squeeze().numpy()*1000.0
        focusPos_np = focusPos[i].detach().numpy()
        focusPos_np = depth_from_region(dpt_np, focusPos_np)
        focal_img = myd2d(J_np, dpt_np, focusPos_np, inpaint_occlusion=False)
        focal_img = focal_img/127.5-1
        focal_img = n2t(focal_img)
        dpt_np = cv2.resize(dpt_np, (3072, 1536))
        sim_autofocus_map = np.abs((dpt_np - focusPos_np)[..., np.newaxis])/6000.0
        ############# 05/07/2020
        sim_autofocus_map = (sim_autofocus_map - sim_autofocus_map.min()) / (sim_autofocus_map.max() - sim_autofocus_map.min())*2.0-1.0
        
        assert (sim_autofocus_map.max() <= 1) and (sim_autofocus_map.min() >= -1)
        sim_autofocus_map = n2t(sim_autofocus_map)
        simAutofocusTensor.append(sim_autofocus_map)
        imageTensor.append(focal_img)
        
    imageTensor = torch.stack(imageTensor)
    simAutofocusTensor = torch.stack(simAutofocusTensor)

    if is_cuda_tensor:
        imageTensor = imageTensor.cuda()
        simAutofocusTensor = simAutofocusTensor.cuda()
    
    return imageTensor, simAutofocusTensor, (torch.abs(simAutofocusTensor) < threshold).float()

def getAFs(dpts, locs):
    batch_size, C, H, W = dpts.size()

    afs = []
    
    for i in range(batch_size):
        dpt = dpts[i]
        loc = locs[i]
        window_size = 512
        x_l = int((loc[0]+1) * (H - window_size) / 2)
        y_l = int((loc[1]+1) * (W - window_size) / 2)
        x_r = int(min(H, x_l + window_size))
        y_r = int(min(W, y_l + window_size))
        af = torch.abs(dpt - dpt[:, x_l:x_r, y_l:y_r].mean())
        af = ((af - af.min())*255.0/(af.max() - af.min())).int().float()/127.5 - 1.0
        afs.append(af)
    
    afs = torch.stack(afs)
    
    return afs

def greedyReward(input_t, locs):
    batch_size, C, H, W = input_t.size()

    rewards = []
    
    for i in range(batch_size):
        loc = locs[i]
        window_size = min(H, W)//4
        x_l = int((loc[0]+1) * (H - window_size) / 2)
        y_l = int((loc[1]+1) * (W - window_size) / 2)
        x_r = int(min(H, x_l + window_size))
        y_r = int(min(W, y_l + window_size))
        if torch.mean(input_t[i][:, x_l:x_r, y_l:y_r]) > 0.0:
            r = 1
        else:
            r = 0
        rewards.append(r)
    
    rewards = torch.FloatTensor(rewards)
    if input_t.is_cuda:
        rewards = rewards.cuda()
    
    return rewards
    

def fuseTwoImages(I, J_hat):
    '''
    AWnet fusion algorithm. 
    Input:
        I Tensor (B, C, H, W): current timestep captured minibatch
        J Tensor (B, C, H, W): last timestep fused minibatch
    Output:
        fusedTensor (B, C, H, W): current timestep fused minibatch
    '''

    with torch.no_grad():
        fusedTensor,_ ,_ = AWnet(J_hat/2+0.5,I/2+0.5)
    
    return torch.clamp(fusedTensor*2-1, -1, 1) 
