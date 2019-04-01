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

<img src="https://github.com/karthikrajkumar/Continuous-control/blob/master/env%20solved.JPG" data-canonical-src="https://github.com/karthikrajkumar/Continuous-control/blob/master/env%20solved.JPG" width="500" height="400" />

## About the Model DDPG (Deep Deterministic Policy Gradients)
This project implements an off-policy method called Deep Deterministic Policy Gradient and described in the paper
[continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
  > We adapt the ideas underlying the success of Deep Q-Learning to the continuous action domain. We present an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces. Using the same learning algorithm, network architecture and hyper-parameters, our algorithm robustly solves more than 20 simulated physics tasks, including classic problems such as cartpole swing-up, dexterous manipulation, legged locomotion and car driving. Our algorithm is able to find policies whose performance is competitive with those found by a planning algorithm with full access to the dynamics of the domain and its derivatives. We further demonstrate that for many of the tasks the algorithm can learn policies end-to-end: directly from raw pixel inputs

Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

**DDPG** is a policy gradient algorithm that uses a stochastic behavior policy for good exploration but estimates a deterministic target policy, which is much easier to learn. DDPG also employs **Actor-Critic** model in which the Critic model learns the value function like DQN and uses it to determine how the Actor’s policy based model should change. The Actor brings the advantage of learning in continuous actions space without the need for extra layer of optimization procedures required in a value based function while the Critic supplies the Actor with knowledge of the performance.

For more information please visit [here](https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html) and [spinningup](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

To mitigate the challenge of unstable learning, a number of techniques are applied like Gradient Clipping, Soft Target Update through twin local / target network and Replay Buffer.

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
As discussed in the Udacity instructions, a further evolution to this project would be to train the 20-agents version.

* In which case, it might be better to use other algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.
  > This work adopts the very successful distributional perspective on reinforcement learning and adapts it to the continuous control setting. We combine this within a distributed framework for off-policy learning in order to develop what we call the Distributed Distributional Deep Deterministic Policy Gradient algorithm, D4PG. We also combine this technique with a number of additional, simple improvements such as the use of N-step returns and prioritized experience replay. Experimentally we examine the contribution of each of these individual components, and show how they interact, as well as their combined contributions. Our results show that across a wide variety of simple control tasks, difficult manipulation tasks, and a set of hard obstacle-based locomotion tasks the D4PG algorithm achieves state of the art performance

* Another enhancement would be to replace the Ornstein-Uhlenbeck noise process with parameter noise as described in Open AI's [paper](https://arxiv.org/abs/1706.01905)
  > Deep reinforcement learning (RL) methods generally engage in exploratory behavior through noise injection in the action space. An alternative is to add noise directly to the agent's parameters, which can lead to more consistent exploration and a richer set of behaviors. Methods such as evolutionary strategies use parameter perturbations, but discard all temporal structure in the process and require significantly more samples. Combining parameter noise with traditional RL methods allows to combine the best of both worlds. We demonstrate that both off- and on-policy methods benefit from this approach through experimental comparison of DQN, DDPG, and TRPO on high-dimensional discrete action environments as well as continuous control tasks. Our results show that RL with parameter noise learns more efficiently than traditional RL with action space noise and evolutionary strategies individually
