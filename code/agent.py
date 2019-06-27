'''
Created on Jun 25, 2019
@author: ptrtmv
'''

import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from tree import SumTree



GRADIENT_CLIP = 10


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    
    def __init__(self,brain, seed = None, noiseScale = 0.2):
        self.brain = brain
        self.seed = seed

        self.noise = OUNoise(brain.actionSize, self.seed, sigma = noiseScale)        
#         self.noise = GaussianNoise(brain.actionSize, self.seed, sigma = noiseScale)
        
    def act(self,state,withNoise=True):
        actions = self.brain.react(state)
        
        # check if the noise process is active
        if withNoise:
            actions += self.noise.sample()        

        return np.clip(actions, -1, 1)     
    
    def step(self, state, action, reward, nextState, done):    
        self.brain.experience(state, action, reward, nextState, done)

    def reset(self):
        self.noise.reset()
        self.brain.reset()
        
        

class Brain():
    
    def __init__(self, stateSize, actionSize, 
                 actorHiddenLayers=[256,128], 
                 actorBatchNormAfterLayers=[1],
                 criticHiddenLayers=[256,128], 
                 criticBatchNormAfterLayers=[1],
                 criticAttachActionToLayer = 1,
                 gamma = 0.99,
                 actorLearningRate = 1e-4,   
                 criticLearningRate = 1e-4,  
                 actorSoftHardUpdatePace = 1e-3,  
                 criticSoftHardUpdatePace = 1e-3,           
                 dnnUpdatePace = 4, 
                 bufferSize = int(1e5),
                 batchSize = 64, 
                 batchEpochs = 1,
                 weightDecay = 1e-5,
                 usePrioritizedMemory = False,
                 seed = None):
        '''
        Initialization of the brain object        
        :param stateSize: 
                The dimension of the state space; number of features for the deep Q-network
        :param actionSize: 
                Number of possible actions; size of output layer
        :param hiddenLayers: 
                list with sizes of output layers        
        :param gamma: 
                RL discount factor for future rewards (Bellman's return) 
        :param xxxLearningRate: 
                The learning rate for the gradient descent in the DQN; 
                corresponds more or less to the parameter alpha in
                RL controlling the how much the most recent episodes
                contribute to the update of the Q-Table
        :param dnnUpdatePace:
                Determines after how many state-action steps the local network 
                should  be updated. 
        :param xxxSoftHardUpdatePace: 
                If xxxSoftHardUpdatePace < 1: a soft update is performed at each 
                local network update
                If xxxSoftHardUpdatePace >= 1: the target network is replaced by 
                the local network after targetdnnUpdatePace steps
        :param bufferSize: 
                Size of the memory buffer containing the experiences < s, a, r, s’ >     
        
        '''            
        
        
        self.bufferSize = bufferSize
        self.batchSize = batchSize
        self.batchEpochs = batchEpochs
        
        self.actorSoftHardUpdatePace = actorSoftHardUpdatePace
        self.criticSoftHardUpdatePace = criticSoftHardUpdatePace
        self.dnnUpdatePace = dnnUpdatePace
        self.numberExperiences = 0
        
        self.actorLearningRate = actorLearningRate
        self.criticLearningRate = criticLearningRate
        self.gamma = gamma
        
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.seed = random.seed(seed)

        # Actor Network (w/ Target Network)
        self.actorLocal = Actor(stateSize, actionSize, 
                                hiddenLayers=actorHiddenLayers,
                                batchNormAfterLayers=actorBatchNormAfterLayers,
                                seed = seed).to(device)
        self.actorTarget = Actor(stateSize, actionSize, 
                                hiddenLayers=actorHiddenLayers,
                                batchNormAfterLayers=actorBatchNormAfterLayers,
                                seed = seed).to(device)
        self.actorOptimizer = optim.Adam(self.actorLocal.parameters(), 
                                         lr=actorLearningRate)

        # Critic Network (w/ Target Network)
        self.criticLocal = Critic(stateSize, actionSize,
                                  hiddenLayers=criticHiddenLayers,
                                  batchNormAfterLayers=criticBatchNormAfterLayers,
                                  attachActionToLayer=criticAttachActionToLayer, 
                                  seed=seed).to(device)
        self.criticTarget = Critic(stateSize, actionSize,
                                  hiddenLayers=criticHiddenLayers,
                                  batchNormAfterLayers=criticBatchNormAfterLayers,
                                  attachActionToLayer=criticAttachActionToLayer, 
                                  seed=seed).to(device)
        self.criticOptimizer = optim.Adam(self.criticLocal.parameters(), 
                                          lr=criticLearningRate, 
                                          weight_decay=weightDecay)
        
        # Experience Memory
        if usePrioritizedMemory:
            # Prioritized memory
            self.memory = PrioritizedMemory(bufferSize, batchSize, seed)
        else:
            # Replay memory
            self.memory = ReplayBuffer(bufferSize, batchSize, seed)
        
        self.numberExperiences = 0
 
 
    def react(self, states):  # @DontTrace
        '''
        Get the action for a given state
        :param state: react to agiven state
        '''
        
        state = torch.from_numpy(states).float().unsqueeze(0).to(device)
        self.actorLocal.eval()
        with torch.no_grad():
            actions = self.actorLocal(state).cpu().data.numpy()
        self.actorLocal.train()
        
        return actions
    
    def _evaluateNetworkForInput(self,network,*inputValues):     
        """
        Helper function to directly evaluate the forward propagation
        Args: 
            network: 
                the network
            *args: 
                the input values which should be propagated
        Return: 
            The output values in as torch-tensor
        """
           
        network.eval()
        with torch.no_grad():
            tensorOutput = network(*inputValues).cpu()
        network.train()
        return tensorOutput

    def getPriorityError(self, state, action, reward, nextState, done):
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action = torch.from_numpy(action).float().unsqueeze(0).to(device)
        nextState = torch.from_numpy(nextState).float().unsqueeze(0).to(device)   
        
        # calculate nextAction  
        nextActionTarget = self._evaluateNetworkForInput(self.actorTarget,nextState)
        # calculate q-value for nextState and nextAction
        nextQTarget = self._evaluateNetworkForInput(self.criticTarget,nextState, nextActionTarget)
        targetsQ = reward + (self.gamma * nextQTarget * (1 - done))
        # calculate expected q-value for current state and action
        expectedQ = self._evaluateNetworkForInput(self.criticLocal,state, action)
        # invert tensor to numpy-array
        targetsQ = targetsQ.data.numpy()[0]
        expectedQ = expectedQ.data.numpy()[0]
          
        err = abs(targetsQ - expectedQ)[0]        
        return err 
 

    def experience(self,state, action, reward, nextState, done):
        
        self.numberExperiences += 1        
        
        if type(self.memory) == PrioritizedMemory:
            # Save experience in replay memory        
            err = self.getPriorityError(state, action, reward, nextState, done)
            self.memory.add(state, action, reward, nextState, done, err)
        else: 
            self.memory.add(state, action, reward, nextState, done)    
        
        
        if self.numberExperiences %  self.dnnUpdatePace == 0:
            self.learn()
        
    
    def learn(self):    
        if len(self.memory) <= self.batchSize*( self.batchEpochs ) :
            return
        
        for epoch in range(self.batchEpochs):
            self.trainNetworks() 
        
        # ----------------------- update target networks ----------------------- #
        if self.criticSoftHardUpdatePace < 1:
            self.softTargetUpdate(self.criticTarget,self.criticLocal,self.criticSoftHardUpdatePace)
        elif self.numberExperiences %  ( self.dnnUpdatePace * self.criticSoftHardUpdatePace) == 0: 
            self.targetUpdate(self.criticTarget,self.criticLocal)

            
        if self.actorSoftHardUpdatePace < 1 and \
           self.numberExperiences %  self.dnnUpdatePace == 0: 
            self.softTargetUpdate(self.actorTarget,self.actorLocal,self.actorSoftHardUpdatePace)
        elif self.numberExperiences %  ( self.dnnUpdatePace * self.actorSoftHardUpdatePace) == 0: 
            self.targetUpdate(self.actorTarget,self.actorLocal)
    
    
    def trainNetworks(self):
        
        # sample experience from memory
        if type(self.memory) == PrioritizedMemory:
            experience, idxs, is_weight = self.memory.torchSample()
        else:
            experience = self.memory.torchSample()
        
        states, actions, rewards, nextStates, dones = experience
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        nextActionTarget = self.actorTarget(nextStates)
        nextQTarget = self.criticTarget(nextStates, nextActionTarget)
        # Compute Q targets for current states (y_i)
        targetsQ = rewards + (self.gamma * nextQTarget * (1 - dones))
        # Compute critic loss
        expectedQ = self.criticLocal(states, actions)
        criticLoss = F.mse_loss(expectedQ, targetsQ)
        # Minimize the loss
        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.criticLocal.parameters(), GRADIENT_CLIP)
        self.criticOptimizer.step()
        
        # in case of prioritized learning, adjust the estimation errors       
        if type(self.memory) == PrioritizedMemory:
            errors = torch.abs(expectedQ - targetsQ).data.numpy()        
            # update priority
            for i in range(self.batchSize):
                idx = idxs[i]
                self.memory.update(idx, errors[i])     
 
 
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        predictedAction = self.actorLocal(states)
        # Maximize Q-value via Gradient Ascent 
        actorLoss = -self.criticLocal(states, predictedAction).mean()
        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.actorLocal.parameters(), GRADIENT_CLIP)
        self.actorOptimizer.step()

    
    def softTargetUpdate(self,targetDnn,localDnn,updateRatio):
        for targetParam, localParam in zip(targetDnn.parameters(), localDnn.parameters()):
            targetParam.data.copy_(updateRatio*localParam.data + (1.0-updateRatio)*targetParam.data)

     
    def targetUpdate(self,targetDnn,localDnn):
        
        for targetParam, localParam in zip(targetDnn.parameters(), localDnn.parameters()):
            targetParam.data.copy_(localParam.data)

    
    def reset(self):
        pass
        
 
class PrioritizedMemory():
    
    eps = 1.e-6
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.0001
    abs_err_upper = 1

    def __init__(self,bufferSize, batchSize, seed):
        '''  
        Initialize Memory object
        
        :param bufferSize (int): 
                maximum size of buffer
        :param batchSize (int): 
                size of each training batch
        :param seed (int): 
                random seed
        '''  
        self.tree = SumTree(bufferSize)
        self.capacity = bufferSize
        self.batchSize = batchSize
        self.n = batchSize
        self.seed = random.seed(seed)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "nextState", "done"])
        
    def _get_priority(self, error):
        error += self.eps
        err = np.clip(error, 0, self.abs_err_upper)
        return err ** self.a
    
    def add(self, state, action, reward, nextState, done, error):        
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, nextState, done)
        p = self._get_priority(error)
        self.tree.add(p,e)
     
    def treeSample(self):
        n = self.batchSize
        
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)        
        is_weight /= is_weight.max()      
               
        return batch,idxs, is_weight
    
    def sample(self):
        
        try:
            batch,idxs, is_weight = self.treeSample()        
        except LookupError:
#             print('resampling...')
            batch,idxs, is_weight = self.treeSample() 
        
        states = np.vstack([e.state for e in batch if e is not None])
        actions = np.vstack([e.action for e in batch if e is not None])
        rewards = np.vstack([e.reward for e in batch if e is not None])
        nextStates = np.vstack([e.nextState for e in batch if e is not None])
        dones = np.vstack([e.done for e in batch if e is not None])
        
            
        return (states, actions, rewards, nextStates, dones), idxs, is_weight    
    
    def torchSample(self):
        sample,idxs, is_weight = self.sample()
        states, actions, rewards, nextStates, dones = sample
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        nextStates = torch.from_numpy(nextStates).float().to(device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(device)
        return (states, actions, rewards, nextStates, dones) , idxs, is_weight
    
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
        
            
    def __len__(self):
        """Return the current size of internal memory."""
        return self.tree.n_entries
            
 
class ReplayBuffer():
    '''
    Memory buffer containing the experiences < s, a, r, s’ >
    '''    

    def __init__(self,bufferSize, batchSize, seed):
        '''  
        Initialize Memory object
        
        :param bufferSize (int): 
                maximum size of buffer
        :param batchSize (int): 
                size of each training batch
        :param seed (int): 
                random seed
        '''  
        self.memory = deque(maxlen=bufferSize)  
        self.batchSize = batchSize
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
            
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batchSize)        
            
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        nextStates = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None])
  
        return (states, actions, rewards, nextStates, dones)    
    
    def torchSample(self):
        states, actions, rewards, nextStates, dones = self.sample()
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        nextStates = torch.from_numpy(nextStates).float().to(device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(device)
        return (states, actions, rewards, nextStates, dones) 
        
            
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

            

class GaussianNoise:
    def __init__(self, size, seed = None, sigma = 0.12, mu = 0):        
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu 
        self.sigma = sigma
        self.seed = random.seed(seed)        
        
    def reset(self):
        pass

    def sample(self):
        return np.random.normal(self.mu,self.sigma,self.size)





