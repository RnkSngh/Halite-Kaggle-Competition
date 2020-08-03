# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 11:37:42 2020
Contains the RL agent function that utilizes the neural net training. 


@author: User
"""
from itertools import product
import random
from halite_training_net import *
from kaggle_environments.envs.halite.helpers import *


class rl_agent():
    def __init__(self, net):
        self.learning_moves = [] #potential for not using all 400 but allocate just in case 
        self.net = net
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.epsilon = 30 #percent chance of having a learning move
        return
        
    #iterate through all possible moves, caluclates the value sof all moves, and then either explores or takes the best action. Epsilon is the percent of exploratory moves

#def run_agent_helper(neural_net):
    def run_learning_agent(self, obs , *config): #takes in a gamma which 
        #need the two lists to have a None possibility for ships
        config  = config[0]
        shipyard_actions = ["SPAWN", None]
        ship_actions = ["NORTH", "SOUTH", "EAST", "WEST", "CONVERT", None]
        #player = obs['players'][0]
        board = Board(obs, config)
        best_board = Board(obs, config)
        
        shipyards = board.current_player.shipyards
        ships = board.current_player.ships
        random_int = random.randint(1, 101)
        if random_int>self.epsilon:  #This will be an exploratory session 
            #explore all possible input combinations for ships and shipyards, and combine them
            #shipyard_moves = product(shipyard_actions, repeat = len(shipyards))
           # ship_moves = product(ship_actions, repeat = len(ships))
           # #possible_moves = product(shipyard_moves, ship_moves)
            
            best_action = {}
            best_value = 0
            
            for shipyard in shipyards:
                best_value = 0
                best_move = None
                for shipyard_move in shipyard_actions:
                    shipyard.next_action = ShipyardAction[shipyard_move] if shipyard_move else None
                    next_obs = board.next().observation
                    obs_tensor = convert_inputs(next_obs)
                    current_value = self.net(obs_tensor).item()
                    if current_value> best_value:
                        best_move = shipyard_move
                        best_value = current_value
                shipyard.next_action = ShipyardAction[best_move] if best_move else None
                
                
            for ship in ships: 
                best_value = 0
                best_move = None
                for ship_move in ship_actions:
                    ship.next_action = ShipAction[ship_move] if ship_move else None
                    next_obs = board.next().observation
                    obs_tensor = convert_inputs(next_obs)
                    current_value = self.net(obs_tensor).item()
                    if current_value > best_value:
                        best_move = ship_move
                        best_value = current_value
                ship.next_action = ShipAction[best_move] if best_move else None
                    
            best_action = board.current_player.next_actions
            self.learning_moves.append(0)
            return best_action
        
        else:# we are taking a learnign move, and thus will take random things
            for shipyard in shipyards:
                random_move = random.choice(shipyard_actions)
                shipyard.next_action = [ShipyardAction[random_move] if random_move else None][0]
            for ship in ships:
                random_move = random.choice(ship_actions)
                ship.next_action = [ShipAction[random_move] if random_move else None][0]
            
            self.learning_moves.append(1) #log this as an exploratory move so the value function is not updated
            return board.current_player.next_actions
#return run_agent
# ============================ =================================================
#             
#             for ship_action in ship_actions:    
#                 for shipyard_action in shipyard_actions:
#                     actions = {}
#                     for ship in ships:for shipps in the reange of not `
#                         for shipyard in shipyards:
#                             if ship_action:
#                                 actions[ship] = ship_action
#                             if shipyard_action:
#                                 actions[shipyard] = shipyard_action
#                     print(actions)
# =============================================================================

#lu = rl_agent()
#lu.run_agent(obs, env.configuration, 0)