"""
This script includes two PyTorch neural nets (one with convolutional layers and one without convolutional layers) that can be used to estimate the value of a given Halite state. The neural net is called by the RL_Agent.py file. In addition to the neural net, this script also contains a function for converting game observations to tensor inputs, and a function for model training after a game is played. 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random 

"""
The neural net used to estimate the value of a given game state, with sigmoid activation functions and a neural net architecture consisting of 2 convolutional layers (and no pooling between convolutions) and 2 linear layers. 

The input is a 21x21x4 tensor, which corresponds to 4 channels of the game grid (see main README for channels). The output is a 1x1 tensor that contains the value of the state. The game state should be passed into the convert_inputs function before being inputted into the neural net. 
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 6, 3) #first convolutional layer
        self.conv2 = nn.Conv2d(6,16,3) #second convolutional layer
        self.fc1 = nn.Linear(16*9, 16) 
        self.fc2 = nn.Linear(16, 1)
        
     '''
    Pytorch forward function used to calculate output value for a given input using sigmoid activation functions

    Parameters
    ----------
    x: Pytorch Tensor
       A 21x12x4 tensor representing the input to be given to the neural net
    '''
    def forward(self, x):
        #calculate 
        x = self.conv1(x)
        x = torch.sigmoid(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        #calculate linear layers
        x = x.view(-1, 16*9) #shape output for linear layers
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
    
    
"""
The neural net used to estimate the value of a given game state, with sigmoid activation functions and 2 linear layers. 
The input is a 21x21x4 tensor, which corresponds to 4 channels of the game grid (see main README for channels). The output is a 1x1 tensor that contains the value of the state. The game state should be passed into the convert_inputs function before being inputted into the neural net. 
"""

class NoCovNet(nn.Module):
    def __init__(self):
        super(NoCovlNet, self).__init__()
        self.fc1 = nn.Linear(16*9, 16)
        self.fc2 = nn.Linear(16, 1)        
     
    '''
    Pytorch forward function used to calculate output value for a given input using sigmoid activation functions

    Parameters
    ----------
    x: Pytorch Tensor
       A 21x12x4 tensor representing the input to be given to the neural net
    '''
    def forward(self, x):
        x = x.view(-1, 16*9)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
    
'''
Trains a given PyTorch neural net for a given set of training inputs and training outputs. 

Parameters
----------
neural_net: Pytorch net
   The neural net to be trained
training_inputs: Pytorch Tensor
   A nx21x21x4 training inputs tensor, where n is the number of training examples
training_outputs: Pytorch Tensor
   A nx1x1 training outputs tensor, where n is the number of training examples
criterion: Pytorch criterion
   The cost function to be minimized through training.  The RL_Agent.py file uses the MSELoss criterion.
optimizer: Pytorch optimizer
   The optimizing method used to minimize the criterion. The RL_Agent.py file uses the omtim.SGD (stochastic gradient descent) optimizer.
'''
def train(neural_net, training_inputs, training_outputs, criterion, optimizer):
    optimizer.zero_grad() #zero gradients before calculating them
    current_outputs = neural_net(training_inputs) 
    loss = criterion(current_outputs, training_outputs) #calculate the criterion to be optimized
    loss.backward() #calculate gradients using SGD
    optimizer.step() #step in the direction of decreasing gradients
    return

'''
Converts 1D coordinates to 2D column and row coordinates; used by the convert_inputs function

Parameters
----------
size: int
   The size of one dimension along a square map. For halite, this size will be fixed at 21, as the map size is 21x21
pos: int
    The 1D coordinate to be converted to 2D. This will be a number between 0 and size^2
'''
def get_col_row(size, pos):
    return (pos % size, pos // size)
    

'''
Converts a game state observation produced by the Halite SDK to a tensor format

Parameters
----------
observation: dict
    A dict given by the Halite SDK that represents the game state 
'''
def convert_inputs(observation):
    input_tensor = torch.zeros(4,21,21) #tensor to be filled out
    halite_tensor = torch.tensor(observation['halite']).view(21,21) #convert halite array to PyTorch format
    input_tensor[0, : , :] = halite_tensor #set the first channel
    players = observation['players']
    for player_index, player in enumerate(players): #iterate through all players to build the 2nd, 3rd, and 4th channels for each existing game piece
        shipyards = player[1]
        ships = player[2]
        for shipyard_key in shipyards:
            #convert 1d to 2d coord
            _1d_coordinate = shipyards[shipyard_key]
            col, row = get_col_row(21, _1d_coordinate)
            input_tensor[1, row, col] = player_index + 1 #add 1 as indexing by 0  
        for ship_key in ships:
            #convert 1d to 2d coord
            _1d_coordinate = ships[ship_key][0]
            ship_halite = ships[ship_key][1]
            col, row = get_col_row(21, _1d_coordinate)
            input_tensor[2, row, col] = player_index + 1
            input_tensor[3, row, col] = ship_halite 
    return input_tensor.view(1, 4, 21, 21) #convert to tensor format


'''
Trains the neural net associated with an exploratory reinforcement learning agent after a training game is played

Parameters
----------
agent: rl_agent
    The reinforcement learning agent to be trained
game_results: dict array
    The result array returned by the Halite SDK
updating_rate: float
    The rate (between 0-1) at which training existing training targets are overwritten; included to help model convergence
gamma: float
    The discounting factor that exponentially discounts optimal future states under the followed policy           
'''
def update_values(agent, game_results, updating_rate, gamma):
    future_reward = 0 #stores reward of future states
    neural_net = agent.net
    exploratory_moves = agent.exploratory_moves
    for index, step in reversed(list(enumerate(game_results))): #iterate backwards through game states
        obs = step[0]['observation']
        if step[0].status != 'ACTIVE': #continue if we don't have any moves for this agent
            continue
        if not( exploratory_moves[index-1]): #only update value function for non-exploratory moves
            max_opponent_score = 0 #latest updated to reflect the highest of the opponents' scores for this state
            agent_score = obs['players'][0][0]
            for player in obs['players'][1:]: #iterate through players to find player with highest score
                if max_opponent_score< player[0]:
                    max_opponent_score = player[0] #update max_opponent score
            
            reward = agent_score - max_opponent_score #reward for this immediate step
            training_inputs = convert_inputs(obs)
            old_value = neural_net(training_inputs) #calculate old value for this state
            new_value = reward + gamma*(future_reward - old_value)
            training_value = (1-updating_rate)*old_value + (updating_rate*new_value)
            train(neural_net, training_inputs, training_value, agent.criterion, agent.optimizer) #train the net for this game
            future_reward = neural_net(training_inputs) #update value for next iteration
    agent.learning_moves = [] #clear all agent moves
    return

'''
Trains the neural net associated with an agent using imitation learning based on an inputted game

Parameters
----------
agent: rl_agent
    The reinforcement learning agent to be trained
game_results: dict array
    The result array returned by the Halite SDK after a game is played by the agents that are being modeled
updating_rate: float
    The rate (between 0-1) at which training existing training targets are overwritten; included to help model convergence
gamma: float
    The discounting factor that exponentially discoutns optimal future states under the followed policy           
'''
def rule_update_values(agent, game_results, updating_rate, gamma):
    future_reward = 0 #stores reward of future states
    neural_net = agent.net
    for index, step in reversed(list(enumerate(game_results))):
        obs = step[0]['observation']
        if step[0].status != 'ACTIVE': #continue if we don't have any moves for this agent
            continue
    #unlike the update_values function, this method does not use the agent.exploratory_moves variable, as all moves are exploitative moves
        max_opponent_score = 0
        agent_score = obs['players'][0][0]
        for player in obs['players'][1:]:
            if max_opponent_score< player[0]:
                max_opponent_score = player[0]
        
        reward = agent_score - max_opponent_score #reward for this immediate step
        training_inputs = convert_inputs(obs)
        old_value = neural_net(training_inputs)
        new_value = reward + gamma*(future_reward - old_value)
        training_value = (1-updating_rate)*old_value + (updating_rate*new_value)
        train(neural_net, training_inputs, training_value, agent.criterion, agent.optimizer)
        
        future_reward = neural_net(training_inputs) #update value for next iteration
    return


