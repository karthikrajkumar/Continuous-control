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

**DDPG** is a policy gradient algorithm that uses a stochastic behavior policy for good exploration but estimates a deterministic target policy, which is much easier to learn. Policy gradient algorithms utilize a form of policy iteration: they evaluate the policy, and then follow the policy gradient to maximize performance. Since DDPG is off-policy and uses a deterministic target policy, this allows for the use of the Deterministic Policy Gradient theorem (which will be derived shortly). DDPG is an actor-critic algorithm as well; it primarily uses two neural networks, one for the actor and one for the critic. These networks compute action predictions for the current state and generate a temporal-difference (TD) error signal each time step. The input of the actor network is the current state, and the output is a single real value representing an action chosen from a continuous action space (whoa!). The critic’s output is simply the estimated Q-value of the current state and of the action given by the actor. The deterministic policy gradient theorem provides the update rule for the weights of the actor network. The critic network is updated from the gradients obtained from the TD error signal.

For more information please visit [here](https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html) and [spinningup](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

#### Experience Replay
In general, training and evaluating your policy and/or value function with thousands of temporally-correlated simulated trajectories leads to the introduction of enormous amounts of variance in your approximation of the true Q-function (the critic). The TD error signal is excellent at compounding the variance introduced by your bad predictions over time. It is highly suggested to use a replay buffer to store the experiences of the agent during training, and then randomly sample experiences to use for learning in order to break up the temporal correlations within different training episodes. This technique is known as experience replay. DDPG uses this feature
#### OU Noise
DDPG trains a deterministic policy in an off-policy way. Because the policy is deterministic, if the agent were to explore on-policy, in the beginning it would probably not try a wide enough variety of actions to find useful learning signals. To make DDPG policies explore better, we add noise to their actions at training time. The authors of the original DDPG paper recommended time-correlated OU noise, but more recent results suggest that uncorrelated, mean-zero Gaussian noise works perfectly well. Since the latter is simpler, it is preferred. To facilitate getting higher-quality training data, you may reduce the scale of the noise over the course of training. 

please find the pseudocode from [here](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

### Code implementation
The code was implemented as a reference from the [Udacity DDPG Bipedel] (https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal) and has been modified for the Reacher environment

The code consist of 
* `model.py` - Implement the Actor and the Critic classes.
  - The Actor and Critic classes each implements a Target and a Local Neural Networks used for the training.
* `ddpg_agent.py` - Implement the DDPG agent and a Replay Buffer memory used by the DDPG agent.
  - The Actor's Local and Target neural networks, and the Critic's Local and Target neural networks are instanciated by the Agent's constructor
  - The `learn()` method updates the policy and value parameters using given batch of experience tuples.
* `Continuous_Control.ipynb` - This Jupyter notebooks allows to instanciate and train the agent

# DDPG implementation Observations and reasons
As shown in the results the initial values were not great for this environment.
* Reducing the Sigma values used in the Ornstein-Uhlenbeck noise process was another important change for the agent to start learning.
* I usually add Batch Normalization before the Activation layers in Neural Networks. In this project, it looks like adding the batch normalization layer after the activation layer works better.
* The environment action size is simple, reducing the size of the network (less units) from 400 and 300 to 128 helped improve the performance. 
* Changing the learning rates - Having similar and slightly higher learning rate for both the actor and the critic network and it helped solving the environment.

**Actor Neural Network Architecture**
```
Input nodes (33)
  -> Fully connected nodes (128 nodes, Relu activation)
    -> Batch Normalization
      -> Fully Connected Layer (128 nodes, Relu activation)
        -> Ouput nodes (4 nodes, tanh activation)
```

**Critic Neural Network Architecture**
```
Input nodes (33) 
  -> Fully Connected Layer (128 nodes, Relu activation)
    -> Batch Normlization
      -> Include Actions at the second fully connected layer
        -> Fully Connected Layer (128+4 nodes, Relu activation)
          -> Ouput node (1 node, no activation)
```
Both Neural Networks use the Adam optimizer with a learning rate of 2e-4 and are trained using a batch size of 128.
# Future Work
Is to change multiple parameter and to find the optimal policy to solve this environment in less number of episodes
