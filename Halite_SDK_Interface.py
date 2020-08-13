# -*- coding: utf-8 -*-
"""
Interfaces with the Halite SDK and RL_Agent.py to simulate games against the greedy agent, specified in the greedy_agent.py. To animate a simulated game, 
use the Jupyter Notebook from the docker container (see README)
"""

from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from RL_agent import *
from greedy_agent import *
import pickle 

agentname = "Neural Net " #name of file to save the trained reinforcement learning agent to
net = NoCovNet() #Neural net to be trained
#initialize agent
learning_agent = rl_agent()
learning_agent_function = learning_agent.run_learning_agent

agents = {"learning":learning_agent_function, 'greedy':greedy_agent} #later passed into the game environment

# Create a test environment for use later
board_size = 21
env = make("halite", {"agentTimeout":60}) #increase timeout between moves as agent will take longer to return moves while training
env.agents.update(agents)
agent_count = 4
env.reset(agent_count)

#initialize a variable that will store saved Results to train later
result_list = []
for i in range(500): #play 500 games
    result = env.run(["learning", "greedy", "greedy", "greedy" ]) #RL agent is at first player index
    result_list.append(result)
    rule_update_values(learning_agent, result, .1, .8) #train neural net for played game

torch.save(model.state_dict(), agent_name) # save model to output file
