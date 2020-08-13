"""
Tests the halite reinforcement learning agent to ensure outputs and future changes do not affect past tests,
and that the libary is compatible with the halite sdk
"""
import unittest
from halite_training_net import *
from kaggle_environments import make
from halite_training_net import *
from kaggle_environments.envs.halite.helpers import *

class TestHaliteBot(unittest.TestCase):
    def test_convert_inputs(self):
        env = make("halite", debug = True)
        agent_count = 4
        env.reset(agent_count)
        obs= env.state[0].observation
        converted_tensor = convert_inputs(obs)
        #test_tensor = torch.tensor(4,21,21)
        ship1_index = obs['players'][0][2]['0-1'][0]
        ship2_index = obs['players'][1][2]['0-2'][0]
        ship3_index = obs['players'][2][2]['0-3'][0]
        ship4_index = obs['players'][3][2]['0-4'][0]
        
        col1, row1 = get_col_row(21, ship1_index)
        col2, row2 = get_col_row(21, ship2_index)
        col3, row3 = get_col_row(21, ship3_index)
        col4, row4 = get_col_row(21, ship4_index)
        
        self.assertEqual(converted_tensor[0, 2, row1, col1].item(), 1)
        self.assertEqual(converted_tensor[0, 2, row2, col2].item(), 2)
        self.assertEqual(converted_tensor[0, 2, row3, col3].item(), 3)
        self.assertEqual(converted_tensor[0, 2, row4, col4].item(), 4)
        
        #no shipyards should exist, and no ships should be holding any halite
        shipyard_equal_to_zero = torch.eq(converted_tensor[0, 1, :, :], torch.zeros(1, 1, 21, 21))
        ship_halite_equal_to_zero = torch.eq(converted_tensor[0, 3, :, :], torch.zeros(1, 1, 21, 21))
        self.assertEqual(torch.all(shipyard_equal_to_zero), True)
        self.assertEqual(torch.all(ship_halite_equal_to_zero), True)
        
        
if __name__ == '__main__':
    unittest.main()

