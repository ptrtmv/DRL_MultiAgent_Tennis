'''
Created on Jun 25, 2019

@author: ptrtmv
'''

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)



class Actor(nn.Module):
    """Actor (Policy) Model."""

    
    def __init__(self, stateSize,actionSize,
                 hiddenLayers=[256,128], 
                 batchNormAfterLayers=None,
                 seed = None):
        """Initialize parameters and build model.
        Params
        ======

        """
        
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
            
        self.hiddenLayers = [stateSize, *hiddenLayers]

        # check if and where batchNorm should be added
        batchCount = 0
        if batchNormAfterLayers!= None :
            nextBatchLayer = batchNormAfterLayers[0]         
        else: 
            nextBatchLayer = None
            
        self.network = nn.ModuleList([])
        i = 0
        for s1,s2 in zip(self.hiddenLayers[:-1],self.hiddenLayers[1:]):
            
            # check if to attach batch-norm here
            if i == nextBatchLayer:
                self.network.append(nn.BatchNorm1d(s1)) # append the batch layer
                batchCount = min( len(batchNormAfterLayers)-1,batchCount+1) # adjust index running over batch-norm layers
                nextBatchLayer = batchNormAfterLayers[batchCount]   # get the position of the next batch-norm layer                 
                
            
            self.network.append(nn.Linear(s1,s2))
            i+=1
        
        self.outLayer = nn.Linear(s2,actionSize)
            
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.network:
            if type(layer) == nn.Linear:
                layer.weight.data.uniform_(*hidden_init(layer))
        self.outLayer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        nextX = state
        
        for layer in self.network:            
            if type(layer) == nn.Linear:
                nextX = F.relu(layer(nextX))
            else: #this should be a batch-layer
                nextX = layer(nextX)
                         
        return torch.tanh(self.outLayer(nextX))
    

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, stateSize,actionSize,
                 hiddenLayers=[256,128], 
                 batchNormAfterLayers=None,
                 attachActionToLayer=1,
                 seed = None):
        """Initialize parameters and build model.
        Params
        ======

        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.attachActionToLayer = attachActionToLayer        
        self.hiddenLayers = [stateSize, *hiddenLayers]

        # check if and where batchNorm should be added
        batchCount = 0
        if batchNormAfterLayers!= None :            
            nextBatchLayer = batchNormAfterLayers[batchCount]         
        else: 
            nextBatchLayer = None
            
        self.network = nn.ModuleList([])
        i = 0
        for s1,s2 in zip(self.hiddenLayers[:-1],self.hiddenLayers[1:]):
            
            # check if the action should be attached here
            # and adjust the size of the layer
            if i == attachActionToLayer:
                s1 += actionSize    
            
            # check if to attach batch-norm here
            if i == nextBatchLayer:
                self.network.append(nn.BatchNorm1d(s1)) # append the batch layer
                batchCount = min( len(batchNormAfterLayers)-1,batchCount+1) # adjust index running over batch-norm layers
                nextBatchLayer = batchNormAfterLayers[batchCount]   # get the position of the next batch-norm layer                 
                # if the action has not been attached yet 
                # we have to shift the index of the layer it should be attached to
                # because an additional batch layer has been added 
                if attachActionToLayer > i:
                    self.attachActionToLayer += 1
            
            self.network.append(nn.Linear(s1,s2))
            i+=1

            
        
        self.outLayer = nn.Linear(s2,1)
            
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.network:
            if type(layer) == nn.Linear:
                layer.weight.data.uniform_(*hidden_init(layer))
        self.outLayer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        nextX = state
        
        i = 0
        for layer in self.network:
            if i == self.attachActionToLayer:
                nextX = torch.cat((nextX, action), dim=1)
            
            if type(layer) == nn.Linear:
                nextX = F.relu(layer(nextX))
            else: #this should be a batch-layer
                nextX = layer(nextX)
            
            i+=1

        return self.outLayer(nextX)










