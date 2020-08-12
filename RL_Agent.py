"""
This file contains the agent class that gives infromation to the Halite environment in the local_halite_sdk.py file during games. This class uses the PyTorch neural net specified in the RL_agent.py file to model the value function of game states, and thus is also accessed when training the RL agent. 
"""

###################### Modules ###########################
import random 
from halite_training_net import * 
from kaggle_environments.envs.halite.helpers import *


class rl_agent():
    """
    A reinforcement learning agent used to give moves to the Halite SDK through the run_learning_agent method. 
    
    self.net holds the PyTorch neural net used to estimate the value of a state. The self.optimizer and self.criterion are used by PyTorch during the gradient descent procses.       self.learning_moves is an array that keeps track track of exploratory moves, and is used decide on what states to train the agent on. 
    """
    
    def __init__(self, net):
        self.learning_moves = [] #keep track of which moves in a played game are learning moves. 1 indicates an exploratory move, 0 indicates an expoitative move
        self.net = net 
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.epsilon = 30 #percent chance of having an exploratory move
        return
        

    def run_learning_agent(self, obs , *config): #takes in a gamma which 
        '''
        A reinforcement learning agent to play in the Halite game
        
        This method takes in the current game state as an input, and returns a dict array to the Halite SDK representing the moves that ships and shipyards will take in the next         turn.  The method iterates through all owned ships and shipyards and finds the next possible moves for each ship and shipyard. The game state for each next possible move         is converted to a PyTorch tensor and fed into a neural net to find the value of each possible next state. The best next state for each ship and shipyard is returned in a         dict array.
        Parameters
        ----------
        obs : dict
            An observation dictionary given by the the Halite SDK corresponding to the state within a Halite game.
        config: dict
            Config paramaters of the game environment in which this function is called; used to construct possible observations for each possible move, which are then inputted               into a neural net
        '''
        config  = config[0]
        #shipyard_actions and ship_actions enumerate all possible moves for ships and shpiyards. These are different than the ones used by the SDK because they also include the possibility of no action (indicated by the None entry)
        shipyard_actions = ["SPAWN", None] 
        ship_actions = ["NORTH", "SOUTH", "EAST", "WEST", "CONVERT", None]
        board = Board(obs, config) #construct a new board from the given observation
        best_board = Board(obs, config) #best board keeps track of the board resulting from the best possible move for each ship/shipyard
        shipyards = board.current_player.shipyards
        ships = board.current_player.ships
        random_int = random.randint(1, 100) #pick a random int to determine if move will be exploratory or expoitative
        if random_int>self.epsilon:  #If this is an exploratory move
            #find best actions for each shipyard and ship and store them in the next_action variable for each ship and shipyard
            best_value = 0 #the best state value for each ship or shipard; is reset for each ship/shipyard
            
            for shipyard in shipyards: 
                best_value = 0
                best_move = None
                for shipyard_move in shipyard_actions:
                    shipyard.next_action = ShipyardAction[shipyard_move] if shipyard_move else None
                    next_obs = board.next().observation #construct a new board for this possible move
                    obs_tensor = convert_inputs(next_obs) #convert possible move observation to a tensor
                    current_value = self.net(obs_tensor).item() #calculate the value of this possible state using the Neural Net
                    if current_value> best_value:
                        best_move = shipyard_move
                        best_value = current_value
                shipyard.next_action = ShipyardAction[best_move] if best_move else None #update shipyard to include the best move as the next_action
                
            for ship in ships: 
                best_value = 0
                best_move = None
                for ship_move in ship_actions:
                    ship.next_action = ShipAction[ship_move] if ship_move else None
                    next_obs = board.next().observation #construct a new board for this possible move
                    obs_tensor = convert_inputs(next_obs)  #convert possible move observation to a tensor
                    current_value = self.net(obs_tensor).item() #calculate the value of this possible state using the Neural Net
                    if current_value > best_value:
                        best_move = ship_move
                        best_value = current_value
                ship.next_action = ShipAction[best_move] if best_move else None #update ship to include the best move as the next_action
                    
            best_action = board.current_player.next_actions #the dict to be returned by this function
            self.learning_moves.append(0) #update the learning_moves array to reflect this was an exploitative move
            return best_action
        
        else:# #If this is an exploitative move
            #pick a random move for each ship and shipyard
            for shipyard in shipyards:
                random_move = random.choice(shipyard_actions)
                shipyard.next_action = [ShipyardAction[random_move] if random_move else None][0]
            for ship in ships:
                random_move = random.choice(ship_actions)
                ship.next_action = [ShipAction[random_move] if random_move else None][0]
            
            self.learning_moves.append(1) #update the learning_moves array to reflect this was an exploratory move
            return board.current_player.next_actions
