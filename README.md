# Halite-Kaggle-Competition
This repo includes a set of tools that can be used to build and train reinforcement learning agents to play [Halite](https://www.kaggle.com/c/halite). The tools included are:
* A PyTorch neural net used for estimating the value of a given game state
* A set of functions (and a script for testing these functions) for interfacing between the Halite SDK and the PyTorch neural net
* A Docker container that can be used to remotely train agents
* Some sample agents produced using tools in this repository. 

# Sample agents:
This section includes some of the sample agents trained using this repository. Note - these will certainly need some tweaking, but might be able to offer a starting point for further optimization. All agents were trained against [greedy rule-based agents](https://www.kaggle.com/tmbond/halite-example-agents).  For each sample, a gif containing a snippet from a played game that represents behavior specific to the agent, with the agent being represented as the yellow-colored team, starting on the upper-left side of the game grid. 

## Loading agents
The agents are provided in the [SampleAgents](./SampleAgents) directory, and can be imported using PyTorch's ```torch.load``` function. 
Below are the results from training the agents thus far. There will still need to be trained more - work still might need to be done in testing out different paramaters and architectures.

### GreedyTraining
This agent was trained using imitation learning on a greedy rule-based agent for 500 games (i.e. the agent was trained on games of 4 greedy rule-based agents playing against eachother). Though imitation learning is typically used for problems with spare reward signals, and the reward for this environment is not sparse as it is given at the end of each timestep, imitation learning might decrease the training data required to learn complex behavior. Improvements can be made on this imitation strategy by imitating more sophisticated rule-based agents. Though this approach ulitmately limits the agent's performance to the performance of the rule based agents it is trained on, it can be used to train a starting agent that can be further optimized using exploratory learning. 

### ExploratoryTrainingGamma-0.1
This agent used . For some reason, this has learned osme beahvior better than higher levels of gamma- for learning behavior such as 
### ExploratoryTrainingGamma-0.8
An exploratory agent wtih gamma equal to .8
### ExploratoryTrainingGamma-1.0

## Expanding on training
The sample agents can be further optimized by testing different discount factors (```gamma```), reward functions, and implementing (PPO), using Eexperience replay, testing out different reward functions and neural net architectures ([see Neural Net Design section](#neural-net-design)) (learning  hyperparamatesr can be tested for these : Different versions of gamma, different neural net architectures, and different reward functions (currently only a reward function


# Running agents in a Docker conttainer
The  agent can be trained using Jupyter Notebooks hosted on a remote machine using the haliterltraining docker container, which contains an installation of PyTorch, kaggle-environments, and Jupyter Notebooks. The container can be imported by running  ```docker pull rnksngh/haliterltraining``` from the command line. The notebook can be run through the commandline using ```docker run -p 8888:8888 rnksngh/haliterltraining```. Once the notebook is running, the agent can be trained using the ```local_halite_sdk.ipynb``` file. 

# Neural Net Design: 
The input is given to the neural net from a 2x2x21 vector, and is converted using x script . The first channel is x 

## Neural Net Input

## Reward function

## Hyperparamaters
x script gets you to here. 
z
Y script gets you to here. 

z script gets you to here. 

# Saving agents in Pytorch 
