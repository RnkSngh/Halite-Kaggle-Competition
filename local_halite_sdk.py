# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:23:24 2020

@author: User
"""
import gym
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from RL_agent import *

#initialize agent
learning_agent = rl_agent()
learning_agent_function = learning_agent.run_learning_agent

agents = {"learning":learning_agent_function}

# Create a test environment for use later
board_size = 21
env = make("halite", {"agentTimeout":60})
env.agents.update(agents)
agent_count = 4
env.reset(agent_count)
result = env.run(["learning", "random", "random", "random" ])
#env.render(mode="ipython", width=800, height=600)


# =============================================================================
# test_move = learning_agent.run_agent(state.observation, env.configuration)
# #board = Board(state.observation, environment.configuration)
# #board.ships['0-1'].next_action = ShipAction.EAST
# #board.ships['0-2'].next_action = ShipAction.WEST
# #ship_u = board.ships['0-1']
# print(len(env.state))
# result = env.run(['random' , 'random', 'random', 'random'])
# state1 =  env.state[0]
# print(len(env.state))
# print(env)
# print(env)
# 
# =============================================================================
#will need to keep track for every move for i in 
    