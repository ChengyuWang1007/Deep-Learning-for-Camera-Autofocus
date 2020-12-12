#!/usr/bin/env python
# coding: utf-8

# # ROBOTICS FOCUS CONTROL

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# In[4]:


class focusLocNet(nn.Module):
    
    def __init__(self, std, _channel, _hidden_size, _out_size):
        super(focusLocNet, self).__init__()
        
        self.std = std
        self.channel = _channel
        self.hidden_size = _hidden_size
        self.out_size = _out_size
        
        ##---------------0430 conv downward ------------
        self.block0_0 = convBlock(self.channel, 4, 8, 8)
        self.block0_1 = convBlock(4, 8, 3, 3)
        
        self.block1 = convBlock(8, 16, 5, 2)
        self.block2 = convBlock(16, 32, 5, 2)
        self.block3 = convBlock(32, 32, 5, 2)
        self.block4 = convBlock(32, 64, 3, 2, isBn = False)
        self.fc0 = nn.Linear(self.out_size, 128)
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = fcBlock(128+128, 256)
        self.fc3 = fcBlock(256, 256, activation = None)
        self.lstm = nn.LSTM(input_size=256, hidden_size=self.hidden_size, num_layers=1)

        self.fc4 = fcBlock(self.hidden_size, 128)
        self.fc5_0 = fcBlock(128, 128)
        self.fc5 = nn.Linear(128, self.out_size)
        self.fc6_0 = fcBlock(self.hidden_size, 128)
        self.fc6 = nn.Linear(128, 1)

        torch.nn.init.kaiming_normal_(self.fc0.weight)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.lstm.weight_hh_l0)
        torch.nn.init.kaiming_normal_(self.lstm.weight_ih_l0)
        torch.nn.init.kaiming_normal_(self.fc5.weight)
        torch.nn.init.kaiming_normal_(self.fc6.weight)
        
    def forward(self, x, l_prev, h_prev):
        batch_size = x.size(0)
        ##---------------0430 conv downward ------------
        x = self.block0_0(x)
        x = self.block0_1(x)
        
        x = self.block1(x)
        x = self.block2(x) 
        x = self.block3(x) 
        x = self.block4(x) 
        
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        y = F.relu(self.fc0(l_prev))
        x = torch.cat((x, y), dim = 1)
        x = self.fc2(x)
        x = self.fc3(x)
        
        x, hidden = self.lstm(x.view(1, *x.size()), h_prev)

        x = hidden[0].view(batch_size, -1)
        
        b = self.fc6_0(x.detach())
        b = self.fc6(b).squeeze(1)

        x = self.fc4(x)
        x = self.fc5_0(x)
        mu = torch.tanh(self.fc5(x))
        
        noise = torch.zeros_like(mu)
        noise.data.normal_(std=self.std)
        pos = mu + noise

        # bound between [-1, 1]
        pos = torch.clamp(pos, -1, 1)

        log_pi = Normal(mu, self.std).log_prob(pos)
        log_pi = torch.sum(log_pi, dim=1)
         
        return hidden, mu, pos, b, log_pi

class convBlock(nn.Module):
    '''
    Conv+ReLU+BN
    '''

    def __init__(self, in_feature, out_feature, filter_size, stride = 1, activation = F.relu, isBn = False):
        super(convBlock, self).__init__()
        self.isBn = isBn
        self.activation = activation

        self.conv1 = nn.Conv2d(in_feature, out_feature, filter_size, stride=stride)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(out_feature)

    def forward(self, x):
        x = self.conv1(x)
                
        if self.isBn:
            x = self.bn1(x)

        if self.activation is not None:
            x = self.activation(x)

        return x

class fcBlock(nn.Module):
    
    def __init__(self, in_feature, out_feature, activation = F.relu, isBn = True):
        super(fcBlock, self).__init__()
        self.isBn = isBn
        self.activation = activation
        
        self.fc1 = nn.Linear(in_feature, out_feature)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(out_feature)
        
    def forward(self, x):
        
        x = self.fc1(x)
                
        if self.isBn:
            x = self.bn1(x)

        if self.activation is not None:
            x = self.activation(x)

        return x

if __name__ == '__main__':
    model = focusLocNet(0.17, 1, 256, 2)
    for n, p in model.named_parameters():
        if(p.requires_grad) and ("bias" not in n):
            print(n, p.max().item())
            print(n, p.mean().item())
