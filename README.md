# DRL Multi-Agent Tennis Environment
Deep Reinforcement Learning Multi Agent Environment (Unity Agent) solved using the method of __Multi Agent Deep Deterministic Policy Gradients (MADDPG)__ with __prioritized learning__.

---

## Intorduction

In this Deep Reinforcement Learning project we train two  Unity ML-agents to play tennis  [Unity Tennis environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis). 
. 

<p align="center">
  <img width="460" height="300" src="plots/tennis.png">
</p>

Each of the agents can control a racket and has to hit the ball over the net.

The __state space__ is `24` dimensional. There are `8` parameters corresponding to `position and velocity`  of the ball and racket whereby each agent receives its own, local observation.    
The __actions__ are `2` corresponding to the movement of the racket `up-down` and `right-left`.  Every entry in the action vector must be a number between `-1` and `1`.

A __reward__ of `+0.1` is provided if an agent hits the ball over the net. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a __negative reward__ of `-0.01`. 

The __environment is considered solved__, when an average score of +0.5  (over 100 episodes) is reached.

This is a __cooperative game__. The total reward of a given agent depends also on the skill of the other agent and the rewards of both agents grow the longer the ball is in play. 

## Getting Started

In order to run the environment you need to download it for your  operation system from one of the following links:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
 


Additionally you may need to install following packages: 
* __matplotlib__: used to visualise the training results 
* __numpy__: shouldn't really surprise you...
* __torch__: used for the deep neural networks
* __unityagents__: used to run the downloaded [Unity Tennis environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis). 

The packages can be directly installed while running the __Jupiter__ notebook `Tennis.ipynb`

