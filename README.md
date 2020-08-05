# Halite-Kaggle-Competition
This repo is a set of tools to build and train reinforcement learning agents that can play [Halite](https://www.kaggle.com/c/halite), and a few weakly trained starting agents in PyTorch. These sample agents still need to be further optimized, but serve as a good starting point for deciding reinforcement learning strategies. 

# Sample agents:
Below are the results from training the agents thus far. There will still need to be trained more - work still might need to be done in testing out different paramaters and architectures. 



# Running agents in a Docker conttainer
The  agent can be trained using Jupyter Notebooks hosted on a remote machine using the haliterltraining docker container, which contains an installation of PyTorch and kaggle-environments . The container can be imported by running  ```docker pull rnksngh/haliterltraining``` from the command line. The notebook can be run through the commandline using ```docker run -p 8888:8888 rnksngh/haliterltraining```. Once the notebook is running, the agent can be trained using the ```local_halite_sdk.ipynb``` file. 

# Neural Net Architecture: 
The input is given to the neural net from a 2x2x21 vector, . The first channel is x 

x script gets you to here. 
z
Y script gets you to here. 

z script gets you to here. 

# Saving agents in Pytorch 
