#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import warnings
warnings.simplefilter("ignore", UserWarning)
import torchvision
import utils
from model import *
from tensorboardX import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import shutil

# In[2]:


class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class easyAppendDict(object):
    
    def __init__(self, *argv):
        self.dict = {}
        for arg in argv:
            self.dict[arg] = []
    def __getitem__(self, key):
        if key in self.dict.keys():
            return self.dict[key]
        else: 
            raise KeyError("no such key.")
    def append(self, *argv):
        for i, (k, arg) in enumerate(zip(self.dict.keys(), argv)):
            (self.dict[k]).append(arg)
            
    def toTensor(self):
        for i, (k, v) in enumerate(self.dict.items()):
            self.dict[k] = torch.stack(v, dim = 1)
            
class prettyfloat(float):
    def __repr__(self):
        return "%0.3f" % self    

# In[3]:


class Trainer(object):
    
    def __init__(self, config, data_loader):
        
        self.config = config  
        self.is_train = config.is_train
        
        if self.is_train: #
            self.train_loader = data_loader[0]
            self.num_train = len(self.train_loader.sampler.indices)
            self.valid_loader = data_loader[1]
            self.num_valid = len(self.valid_loader.sampler.indices)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
          
        self.use_cuda = config.use_cuda
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.ckpt_dir = config.ckpt_dir
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.resume = config.resume
        self.model_name = 'rfc'
        self.use_tensorboard = config.use_tensorboard
        self.is_plot = config.is_plot
        self.best_loss = 0
        
        if self.use_tensorboard:
            tensorboard_dir = config.logs_dir
#             if not self.is_train:
#                 tensorboard_dir = tensorboard_dir + '_test'
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            self.writer = SummaryWriter(tensorboard_dir)
        
        self.channel = config.channel
        self.hidden_size = config.hidden_size
        self.out_size = config.out_size
        
        self.std = config.std
        self.model = focusLocNet(self.std, self.channel, self.hidden_size, self.out_size).to(self.device)
        
        self.start_epoch = 0
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.seq = config.seq
        self.lr = config.init_lr
        self.optimizer= optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        self.rl_loss = config.rl_loss
        
        self.use_gan = config.use_gan
                        
    def reset(self):
        h = [torch.zeros(1, self.batch_size, self.hidden_size).to(self.device),
                      torch.zeros(1, self.batch_size, self.hidden_size).to(self.device)]
        l = torch.rand(self.batch_size, self.out_size).to(self.device)*2.0-1.0
        return h, l
      
            
    def train(self):
        """
        Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint()
            
            
        print("\n[*] Train on {} samples.".format(
            self.num_train)
        )           
            
        self.model.train()

        for epoch in range(self.start_epoch, self.epochs):

            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch+1, self.epochs, self.lr)
            )

            # train for 1 epoch
            train_loss, train_rw = self.train_one_epoch(epoch)
            
            # evaluate on validation set
            valid_loss, valid_rw = self.validate(epoch)

            is_best = valid_rw >= self.best_loss
            msg1 = "train loss: {:.3f} - train reward: {:.3f}"
            msg2 = " - val loss: {:.3f} - val reward: {:.3f}"
            if is_best:
                msg2 += " [*]"
            msg = msg1 + msg2
            print(msg.format(train_loss, train_rw, valid_loss, valid_rw))
            
            self.best_loss = max(valid_rw, self.best_loss)
            self.save_checkpoint(
                {'epoch': epoch + 1,
                 'model_state': self.model.state_dict(),
                 'optim_state': self.optimizer.state_dict(),
                 'best_valid_mse': self.best_loss,
                 }, is_best
            )
    
    def train_one_epoch(self, epoch):
    
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = {"t_loss": AverageMeter(), "b_loss": AverageMeter(), "r_loss": AverageMeter()}
        rewards = AverageMeter()
        mselosses = AverageMeter()
        tic = time.time()
        
        with tqdm(total=self.num_train) as pbar:
            for i, (x_train, dpt) in enumerate(self.train_loader):

                x_train = x_train.to(self.device)
                dpt = dpt.to(self.device)

                self.batch_size = x_train.size(0)
                self.seq = x_train.size(1)
                
                h, l = self.reset()
                
                afs = []
                mus = []
                locs = []
                log_pi = []
                baselines = []
                reward = []
                reward_wo_gamma = []
                
                if self.use_gan:
                    ## build a discriminator
                    pass

                last_af = None
                for t in range(self.seq):
                    
                    af = utils.getAFs(dpt[:, t, ...], l)

                    if last_af is None:
                        input_t = af
                    else:
                        input_t = torch.min(af, last_af)
                        
                    afs.append(input_t)
                        
                    last_af = af
                    h, mu, l, b, p = self.model(input_t, l, h)
                    log_pi.append(p)
                    mus.append(mu)
                    locs.append(l)
                    baselines.append(b)
                    
                    if self.use_gan:
                        ## treat the agent as a Generator and update rewards
                        raise NotImplementedError("gan loss has not been implemented")
                    else:
                        r = utils.greedyReward(input_t, l)

                    reward.append(r)
                    reward_wo_gamma.append(r)
                    for tt in range(t):
                        reward[tt] = reward[tt] + (0.9 ** (t - tt)) * r

                locs = torch.stack(locs).squeeze()
                mus = torch.stack(mus).squeeze()
                baselines = torch.stack(baselines).transpose(1, 0)
                log_pi = torch.stack(log_pi).transpose(1, 0)
                R = torch.stack(reward).transpose(1, 0) * 1.0
                R_wo_gamma = torch.stack(reward_wo_gamma).transpose(1, 0)
                
                loss_baseline = F.mse_loss(baselines, R)
            
                adjusted_reward = R - baselines.detach()              

                ## Basic REINFORCE algorithm
                loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
                loss_reinforce = torch.mean(loss_reinforce, dim=0)
                

                loss = loss_reinforce + loss_baseline
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses["t_loss"].update(loss.item(), self.batch_size)
                losses["b_loss"].update(loss_baseline.item(), self.batch_size)
                losses["r_loss"].update(loss_reinforce.item(), self.batch_size)

                rewards.update(torch.mean(torch.sum(R, dim = 1),dim = 0).item(),self.batch_size)
                
                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)

                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.5f}".format(
                            (toc-tic), loss.item()
                        )
                    )
                )
                pbar.update(self.batch_size)

            if self.use_tensorboard:
                iteration = epoch#*len(self.train_loader)# + i
#                 self.writer.add_scalars('Stats/total_loss', {"t_loss":losses["t_loss"].avg, 
#                                                              "b_loss":losses["b_loss"].avg,
#                                                              "r_loss":losses["r_loss"].avg}
#                                                              , iteration)
                self.writer.add_scalar('Stats/total_loss', losses["t_loss"].avg, iteration)
                self.writer.add_scalar('Stats/reinforce_loss', losses["r_loss"].avg, iteration)
                self.writer.add_scalar('Stats/reward', rewards.avg, iteration)
                
                for n, p in self.model.named_parameters():
                    if(p.requires_grad) and ("bias" not in n) and ("bn" not in n):
                        self.writer.add_histogram('hist/'+n, p, iteration)

            return losses["t_loss"].avg, rewards.avg
        
    
    def validate(self, epoch):
        
        losses = AverageMeter()
        rewards = AverageMeter()

        for i, (x_train, dpt) in enumerate(self.valid_loader):

            x_train = x_train.to(self.device)
            dpt = dpt.to(self.device)

            self.batch_size = x_train.size(0)
            self.seq = x_train.size(1)

            h, l = self.reset()

            afs = []
            mus = []
            locs = []
            log_pi = []
            baselines = []
            reward = []
            reward_wo_gamma = []

            if self.use_gan:
                ## build a discriminator
                pass

            last_af = None
            for t in range(self.seq):

                af = utils.getAFs(dpt[:, t, ...], l)

                if last_af is None:
                    input_t = af
                else:
                    input_t = torch.min(af, last_af)

                afs.append(input_t)

                last_af = af

                h, mu, l, b, p = self.model(input_t, l, h)
                log_pi.append(p)
                mus.append(mu)
                locs.append(l)
                baselines.append(b)

                if self.use_gan:
                    ## treat the agent as a Generator and update rewards
                    raise NotImplementedError("gan loss has not been implemented")
                else:
                    r = utils.greedyReward(input_t, l)

                reward.append(r)
                reward_wo_gamma.append(r)
                for tt in range(t):
                    reward[tt] = reward[tt] + (0.9 ** (t - tt)) * r

            locs = torch.stack(locs).squeeze()
            mus = torch.stack(mus).squeeze()
            baselines = torch.stack(baselines).transpose(1, 0)
            log_pi = torch.stack(log_pi).transpose(1, 0)
            R = torch.stack(reward).transpose(1, 0) * 1.0
            R_wo_gamma = torch.stack(reward_wo_gamma).transpose(1, 0)

            loss_baseline = F.mse_loss(baselines, R)

            adjusted_reward = R - baselines.detach()              

            ## Basic REINFORCE algorithm
            loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce, dim=0)


            loss = loss_reinforce + loss_baseline

            losses.update(loss.item(), self.batch_size)

            rewards.update(torch.mean(torch.sum(R, dim = 1),dim = 0).item(),self.batch_size)

        if self.use_tensorboard:
            iteration = epoch
            self.writer.add_scalar('Stats/valid_loss', losses.avg, iteration)
            self.writer.add_scalar('Stats/valid_reward', rewards.avg, iteration)

        return losses.avg, rewards.avg
    
    def save_checkpoint(self, state, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date.
        """
#         print("[*] Saving model to {}".format(self.ckpt_dir))

        filename = self.model_name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        
        
        
        if is_best:
            filename = self.model_name + '_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )
        
#         print("[*] Saved model to {}".format(self.ckpt_dir))

    def load_checkpoint(self):
        
        print("[*] Loading model from {}".format(self.ckpt_dir))
        
        filename = self.model_name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_loss = ckpt['best_valid_mse']
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])   
        
        print("[*] Loaded model from {}".format(self.ckpt_dir))
