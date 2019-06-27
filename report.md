# Project report

## Learning algorithm

We implemented the multi agent deep deterministic policy gradient algorithm (MADDPG), introduced to solve deep reinforcement learning problems with continuous actions for multiple agent environments. For more information, please see the linked [article](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf). 

The algorithm uses two networks: an actor network which takes the current environment state as input and outputs an action and a critic network which takes the current state and action and returns the value of the state action function. Those networks are shared by all agents in the environment and the personalized experiences from each of the agents are saved in a collective replay buffer. 

For the replay buffer we considered the cases whit [prioritized experience replay](https://arxiv.org/abs/1511.05952) and without. We observed that both methods converge to a solution of the environment but rather unstable. I.e. for given fixed hyper parameters the number of episodes needed before solving the environment varies strongly with the initial conditions (for example the choice of the seed for the random number generator). We observed that in most cases the **MADDPG algorithm performs better in combination with prioritized experience replay**. 

The convergence speed was increased when we used a **hard update of the target networks** (i.e. the target network were completely overridden after a given number of update steps and remained otherwise fixed). In contrast, the soft update is commonly used in applications with underlying [DDPG](http://proceedings.mlr.press/v32/silver14.pdf) or [deep Q-learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)) algorithms.  


### Hyper Parameters and Architecture

To solve the environment we used the following network architectures: 

-----------ACTOR NETWORK-----------
Actor(
  (network): ModuleList(
    (0): Linear(in_features=24, out_features=256, bias=True)
    (1): Linear(in_features=256, out_features=64, bias=True)
    (2): Linear(in_features=64, out_features=32, bias=True)
  )
  (outLayer): Linear(in_features=32, out_features=2, bias=True)
)


-----------CRITIC NETWORK-----------
Critic(
  (network): ModuleList(
    (0): Linear(in_features=24, out_features=256, bias=True)
    (1): Linear(in_features=258, out_features=128, bias=True)
    (2): Linear(in_features=128, out_features=16, bias=True)
  )
  (outLayer): Linear(in_features=16, out_features=1, bias=True)
)


with `ReLu` activation function for the hidden layers and a `tanh` output in the case of the actor network accounting for the fact that the actions are bound between -1 and +1. 
Here we should mention that we were surprised to observe that both algorithms (with and without prioritized experience replay) performed worse when using batch normalization after the network layers. 



 
The parameters used in the MADDPG algorithm are as follows:

- **gamma**:
    RL [discount factor](https://en.wikipedia.org/wiki/Q-learning#Discount_factor) for future rewards  
- **learningRate (for actor and for critic)**:
    The learning rate for the gradient descent while training the (local) neural network; 
    This parameter corresponds more or less to the [learning rate](https://en.wikipedia.org/wiki/Q-learning#Learning_Rate) in RL controlling how much the most recent episodes contribute to the update of the Q-Table 
- **dqnUpdatePace**:
    Determines after how many state-action steps the local networks should be updated. 
- **dnnUpdatePace (for actor and for critic)**:
    * If targetDqnUpdatePace < 1: a soft update is performed at each local network update
    * If targetDqnUpdatePace >= 1: the target network is replaced by the local network after targetDqnUpdatePace steps. 
- **bufferSize**:2
    Size of the memory buffer containing the experiences < s, a, r, s’ >
- **batchSize**:
    The batch size used in the gradient descent during learning
- **batchEpochs**:
    The number of epochs when training the network  


## Training and Results

To solve the environment we used following parameters:


|Parameter|Value|
|----------------------|-----|
|gamma|0.99|
|actorLearningRate|5e-4|
|criticLearningRate|5e-4|
|actorDnnUpdatePace|10|
|criticDnnUpdatePace|10|
|dnnUpdatePace|2|
|bufferSize|1e6|
|batchSize|128|
|batchEpochs|2|

With this parameters the environment was solved (i.e. the agents maintained an average score of 0.5 over 100 consecutive episodes) in 452 episodes for the Replay-Buffer without experience prioritization and for 212 episodes when using a prioritized experience replay. 


<p align="center">
  <img width="460" height="300" src="plots/training_plot_wo_PRIO.png">
</p>

<p align="center">
  <img width="460" height="300" src="plots/training_plot_w_PRIO.png">
</p>



## Possible Future Extensions of the Setting

1. It is clear that the parameters play a crucial role and one can always try to optimize these further.

The environment can be solved using a different algorithm and otherwise following the same idea of both agents sharing the underlying network and memory structures: 

2. Truncated Natural Policy Gradient or Trust Region Policy Optimization: [link](https://arxiv.org/abs/1604.06778).

3. Proximal Policy Optimization Algorithms (PPO): [link](https://arxiv.org/pdf/1707.06347.pdf)

4. A3C: [link](https://arxiv.org/pdf/1602.01783.pdf)

5.  Distributed Distributional Deterministic Policy Gradients (D4PG): [link](https://openreview.net/pdf?id=SyZipzbCb)

6. Prioritized Experience Replay: [link](https://arxiv.org/abs/1511.05952)
