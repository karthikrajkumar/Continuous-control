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

## Batch normalization
With the reference to the paper [Batch Normalization](https://arxiv.org/pdf/1502.03167v3.pdf)
