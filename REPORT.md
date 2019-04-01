# Technical details pertaining the DDPG algorithm.
The idea is to test multiple hyper parameters for fewer episodes say 100 and the best instance will be used to play the complete game.

## Attempt 1:
The implementation is a pure default implementation from the Udacity DDPG for pong. The Hyper parameters defined are 

* BUFFER_SIZE = int(1e5)  # replay buffer size
* BATCH_SIZE = 128        # minibatch size
* GAMMA = 0.99            # discount factor
* TAU = 1e-3              # for soft update of target parameters
* LR_ACTOR = 1e-4         # learning rate of the actor 
* LR_CRITIC = 1e-3        # learning rate of the critic
* WEIGHT_DECAY = 0        # L2 weight decay


<img src="https://github.com/karthikrajkumar/Continuous-control/blob/master/default%20hp.JPG" data-canonical-src="https://github.com/karthikrajkumar/Continuous-control/blob/master/default%20hp.JPG" width="400" height="300" />

## Attempt 2:
Introducing the Batch normalization, With the reference to the paper [Batch Normalization](https://arxiv.org/pdf/1502.03167v3.pdf)

* BUFFER_SIZE = int(1e5)  # replay buffer size
* BATCH_SIZE = 128        # minibatch size
* GAMMA = 0.99            # discount factor
* TAU = 1e-3              # for soft update of target parameters
* LR_ACTOR = 1e-4         # learning rate of the actor 
* LR_CRITIC = 1e-3        # learning rate of the critic
* WEIGHT_DECAY = 0        # L2 weight decay
* BN_MODE = 2             # BN_MODE = 1 : Batch Normalization before activation & BN_MODE = 2 : Batch Normalization after activation

<img src="https://github.com/karthikrajkumar/Continuous-control/blob/master/BN%20distribution.JPG" data-canonical-src="https://github.com/karthikrajkumar/Continuous-control/blob/master/BN%20distribution.JPG" width="400" height="300" />

## Attempt 3:
Changing the FC units from 400 & 300 to 128 respectively & Changing the Learning rate

* BUFFER_SIZE = int(1e5)  # replay buffer size
* BATCH_SIZE = 128        # minibatch size
* GAMMA = 0.99            # discount factor
* TAU = 1e-3              # for soft update of target parameters
* LR_ACTOR = 5e-4         # learning rate of the actor 
* LR_CRITIC = 5e-4        # learning rate of the critic
* WEIGHT_DECAY = 0        # L2 weight decay
* BN_MODE = 2             # BN_MODE = 1 : Batch Normalization before activation & BN_MODE = 2 : Batch Normalization after activation

<img src="https://github.com/karthikrajkumar/Continuous-control/blob/master/optimal.JPG" data-canonical-src="https://github.com/karthikrajkumar/Continuous-control/blob/master/optimal.JPG" width="400" height="300" />

## Final Execution
With the final attempt (4), I ran the environment and 

<img src="https://github.com/karthikrajkumar/Continuous-control/blob/master/env%20solved.JPG" data-canonical-src="https://github.com/karthikrajkumar/Continuous-control/blob/master/env%20solved.JPG" width="400" height="300" />

## About the Model DDPG (Deep Deterministic Policy Gradients)
This project implements an off-policy method called Deep Deterministic Policy Gradient and described in the paper
[continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
  > We adapt the ideas underlying the success of Deep Q-Learning to the continuous action domain. We present an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces. Using the same learning algorithm, network architecture and hyper-parameters, our algorithm robustly solves more than 20 simulated physics tasks, including classic problems such as cartpole swing-up, dexterous manipulation, legged locomotion and car driving. Our algorithm is able to find policies whose performance is competitive with those found by a planning algorithm with full access to the dynamics of the domain and its derivatives. We further demonstrate that for many of the tasks the algorithm can learn policies end-to-end: directly from raw pixel inputs

Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

**Quick Facts**
* DDPG is an off-policy algorithm.
* DDPG can only be used for environments with continuous action spaces.
* DDPG can be thought of as being deep Q-learning for continuous action spaces.
* The Spinning Up implementation of DDPG does not support parallelization.




# Future Work
Is to change multiple parameter and to find the optimal policy to solve this environment in less number of episodes
