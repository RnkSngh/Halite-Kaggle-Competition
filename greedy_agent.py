# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:31:05 2020

@author: User
"""

import copy
import math
import pprint
from random import choice, randint, shuffle

def get_col_row(size, pos):
    return (pos % size, pos // size)

def get_to_pos(size, pos, direction):
    col, row = get_col_row(size, pos)
    if direction == "NORTH":
        return pos - size if pos >= size else size ** 2 - size + col
    elif direction == "SOUTH":
        return col if pos + size >= size ** 2 else pos + size
    elif direction == "EAST":
        return pos + 1 if col < size - 1 else row * size
    elif direction == "WEST":
        return pos - 1 if col > 0 else (row + 1) * size - 1
    
class Board:
    def __init__(self, obs, config):
        self.action = {}
        self.obs = obs
        self.config = config
        size = config.size
        
        self.shipyards = [-1] * size ** 2
        self.shipyards_by_uid = {}
        self.ships = [None] * size ** 2
        self.ships_by_uid = {}
        self.possible_ships = [{} for _ in range(size ** 2)]
        
        for index, player in enumerate(obs.players):
            _, shipyards, ships = player
            for uid, pos in shipyards.items():
                details = {"player_index": index, "uid": uid, "pos": pos}
                self.shipyards_by_uid[uid] = details
                self.shipyards[pos] = details
            for uid, ship in ships.items():
                pos, ship_halite = ship
                details = {"halite": ship_halite, "player_index": index, "uid": uid, "pos": pos}
                self.ships[pos] = details
                self.ships_by_uid[uid] = details
                for direction in ["NORTH", "EAST", "SOUTH", "WEST"]:
                    self.possible_ships[get_to_pos(size, pos, direction)][uid] = details
        
        #pprint(self.possible_ships)
    
    def move(self, ship_uid, direction):
        self.action[ship_uid] = direction
        # Update the board.
        self.__remove_possibles(ship_uid)
        ship = self.ships_by_uid[ship_uid]
        pos = ship["pos"]
        to_pos = get_to_pos(self.config.size, pos, direction)
        ship["pos"] = to_pos
        self.ships[pos] = None
        self.ships[to_pos] = ship
    
    def convert(self, ship_uid):
        self.action[ship_uid] = "CONVERT"
        # Update the board.
        self.__remove_possibles(ship_uid)
        pos = self.ships_by_uid[ship_uid]["pos"]
        self.shipyards[pos] = self.obs.player
        self.ships[pos] = None
        del self.ships_by_uid[ship_uid]
    
    def spawn(self, shipyard_uid):
        self.action[shipyard_uid] = "SPAWN"
        # Update the board.
        temp_uid = f"Spawn_Ship_{shipyard_uid}"
        pos = self.shipyards_by_uid[shipyard_uid]["pos"]
        details = {"halite": 0, "player_index": self.obs.player, "uid": temp_uid, "pos": pos}
        self.ships[pos] = details
        self.ships_by_uid = details
    
    def __remove_possibles(self, ship_uid):
        pos = self.ships_by_uid[ship_uid]["pos"]
        intended_deletes = []
        for d in ["NORTH", "EAST", "SOUTH", "WEST"]:
            to_pos = get_to_pos(self.config.size, pos, d)
            intended_deletes.append(to_pos)
        #print('Deleting possible positions:', intended_deletes,'for', self.ships_by_uid[ship_uid])
        for d in ["NORTH", "EAST", "SOUTH", "WEST"]:
            to_pos = get_to_pos(self.config.size, pos, d)
            #print("Deleting to_pos:",to_pos, "for", ship_uid)
            #print(self.possible_ships[to_pos])
            del self.possible_ships[to_pos][ship_uid]
            

import sys
import traceback

DIRS = ["NORTH", "SOUTH", "EAST", "WEST"]

# Each ship id will be assigned a state, one of COLLECT or DEPOSIT, which decides what it will do on a turn.
states = {}

COLLECT = "collect"
DEPOSIT = "deposit"


def argmax(arr, key=None):
    return arr.index(max(arr, key=key)) if key else arr.index(max(arr))


# This function will not hold up in practice
# E.g. cell getAdjacent(224) includes position 0, which is not adjacent
def getAdjacent(pos):return [
    (pos - 15) % 225,
    (pos + 15) % 225,
    (pos +  1) % 225,
    (pos -  1) % 225
  ]

def getDirTo(fromPos, toPos):
    fromY, fromX = divmod(fromPos, 15)
    toY,   toX   = divmod(toPos,   15)

    if fromY < toY: return "SOUTH"
    if fromY > toY: return "NORTH"
    if fromX < toX: return "EAST"
    if fromX > toX: return "WEST"

    
def greedy_agent(obs):
    action = {}
    player_halite, shipyards, ships = obs.players[obs.player]

    for uid, shipyard in shipyards.items():
        # Maintain one ship always
        if len(ships) == 0:
            action[uid] = "SPAWN"

    for uid, ship in ships.items():
        # Maintain one shipyard always
        if len(shipyards) == 0:
            action[uid] = "CONVERT"
            continue

        # If a ship was just made
        if uid not in states: states[uid] = COLLECT

        pos, halite = ship

        if states[uid] == COLLECT:
            if halite > 2500:
                states[uid] = DEPOSIT

            elif obs.halite[pos] < 100:
                best = argmax(getAdjacent(pos), key=obs.halite.__getitem__)
                action[uid] = DIRS[best]

        if states[uid] == DEPOSIT:
            if halite < 200: states[uid] = COLLECT

            direction = getDirTo(pos, list(shipyards.values())[0])
            if direction: action[uid] = direction
            else: states[uid] = COLLECT

    return action