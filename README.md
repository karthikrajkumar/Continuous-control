# Continuous-control

![Reacher Environment](https://github.com/karthikrajkumar/Continuous-control/blob/master/reacher.gif)

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
## Solution Criteria for the environment

*   Option 1: Solve the First Version
    -   The task is episodic, and in order to solve the environment, our agent must get an average score of +30 over 100 consecutive episodes.

*   Option 2: Solve the Second Version
    -   The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

        -  After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
        - This yields an average score for each episode (where the average is over all 20 agents).
    - The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

**Out implementation considers Option 1**
The task is episodic, and in order to solve the environment, our agent must get an average score of +30 over 100 consecutive episodes

# Real world example
Watch this [YouTube](https://www.youtube.com/watch?v=ZVIxt2rt1_4) video to see how some researchers were able to train a similar task on a real robot! The accompanying research paper can be found [here](https://arxiv.org/pdf/1803.07067.pdf).

# Environment Details
For this use case, we dont have to install Unity environment as it has already been built. please find the details specific to operating systems.

*    [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
*    [Mac OS](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
*    [Win 32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
*    [Win 64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

## Installation details
you would be requiring the following
*    Python 3.6 or above [click here](https://www.python.org/downloads/)
    
*    Jupyter notebook/ lab

    python3 -m pip install --upgrade pip
    python3 -m pip install jupyter
    
    jupyter notebook
    
*    Pytorch [click for installation](https://pytorch.org/)
*    Numpy & matplotlib
     
    pip install numpy
    pip install matplotlib
     

## How to run this ?
Open the Continuous Control python notebook (Continuous_Control.ipynb) and start running cell by cell or run all.
*    Note: The Unity environment needs to be downloaded and in Continuous_Control.ipynb the path to load the environment needs to be changed accordingly.
