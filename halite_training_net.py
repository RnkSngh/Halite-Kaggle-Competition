# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:51:13 2020
Used in training the RL bot for halite.
Takes in a 21x21 4 channel grid of inputs. 1 corresponds to the values of halite on a board.
The second channel corresponds to the number of ships. Player ships are corresponding to 1 on the board. Enemy ships correspondond to numbers 2-4 based on wwhere they are on the board, and where inputs are on that board. I guess we will only do inputs on where they are. 

Has two functions which help with converting the input of a board, updating the state function of a board

@author: User
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(4, 6, 3)
        self.pool = nn.MaxPool2d(2,2)
        
        self.conv2 = nn.Conv2d(6,16,3)
        self.fc1 = nn.Linear(16*9, 16)
        self.fc2 = nn.Linear(16, 1)
        
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.sigmoid(x)
        x = F.max_pool2d(x, (2,2))
        x = self.conv2(x)
        x = torch.sigmoid(x)
        x = F.max_pool2d(x,2)
        
        x = x.view(-1, 16*9)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
      #  x= F.relu(self.fc2(x))
        return x
    
 
def train(neural_net, training_inputs, training_outputs, criterion, optimizer):
    optimizer.zero_grad()
    current_outputs = neural_net(training_inputs)
    loss = criterion(current_outputs, training_outputs)
    loss.backward()
    optimizer.step()
    return

def get_col_row(size, pos):
    return (pos % size, pos // size)
    

#convert input from a Halite SDK obserivation to a 4x21x21 pytorch tensor
def convert_inputs(observation):
    input_tensor = torch.zeros(4,21,21)
    halite_tensor = torch.tensor(observation['halite']).view(21,21) #convert halite array to pytorch format
    input_tensor[0, : , :] = halite_tensor #get the halite
    
    players = observation['players']
    for player_index, player in enumerate(players):
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
    return input_tensor.view(1, 4, 21, 21)


#updates the values after a given game if it was not an exploratory move. Gamma is the future reduction of moves
def update_values(neural_net, game_results, exploratory_moves, learning_rate, gamma):
    future_reward = 0 #stores reward of future states
    for index, step in reversed(list(enumerate(game_results))):
        if not( exploratory_moves[index]): #only update value function for non-exploratory moves
            obs = step[0]['observation']
            max_opponent_score = 0 #method is player score
            agent_score = obs['players'][0][0]
            for player in obs['players'][1:]:
                if max_opponent_score< player[0]:
                    max_opponent_score = player[0]
            
            reward = agent_score - max_opponent_score #reward for immediate step
            training_inputs = convert_inputs(obs)
            old_value = neural_net(training_inputs)
            new_value = reward + gamma*future_reward
            training_value = (1-learning_rate)*old_value + (learning_rate*new_value)
            train(neural_net, training_inputs, training_value, neural_net.optimizer, neural_net.criterion)
            
            future_reward = neural_net(training_value) #update value for next iteration
    return

# =============================================================================
# net = Net()
# optimizer = optim.SGD(net.parameters(), lr=0.01)
# criterion = nn.MSELoss()
# 
# input_tensor = torch.randn(1,4,21,21)
# output_tensor = torch.randn(1, 1)
# 
# train(net, input_tensor, output_tensor, criterion, optimizer)
# =============================================================================
