## Deep_Reinforcement_Learning_Udacity_Continuous_Control

### Project Details

This project implements DDPG algorithm to train an agent for the Reacher environment with one agent only. In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The agent learns with actor-critic method in which both actor and critic is implemented with Fully Connected Network (FCN). There are two copies of both actor and critic networks called local and target networks. Target network weights are copied from local network weights after each learning step through soft update to blend in the new weights into previous weights.

The actor takes the state of the environment and outputs the action vector with continuous values. The critic takes both state and action vectors and outputs the Q value of the state-action pair.


Following files exist in the project;

- Continuous_Control.ipynb : Python Jupyter notebook that runs the main training loop and then trained agent.
- ddpg_agent.py        	   : DDPG implementation of the agent that interacts with environment. 
- model.py                 : Implementation of the FCN. 
- checkpoint_actor.pth     : Saved model parameters for actor.
- checkpoint_critic.pth    : Saved model parameters for critic.


### Getting Started

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the Udacity Deep Reinforcement Learning repository, and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

     - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)


6. Clone the repository at https://github.com/dhesin/DRL_Reacher_w_DDPG and move to the DRL_Reacher_w_DDPG directory

7. From the command line terminal run ```jupyter notebook``` and select Continuous_Control.ipynb

8. Before running Continuous_Control.ipynb in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

9. Before running Continuous_Control.ipynb in a notebook, change the line in Continuous_Control.ipynb

```env = UnityEnvironment(file_name="./Reacher.app")``` 

by entering the location of the Reacher.app that you have downloaded at step 5.

### Instructions

Follow the instructions on the Continuous_Control.ipynb and run the code cells in the order they are provided. 


