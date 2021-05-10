import os
import numpy as np
from PIL import Image
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_num_regions, calc_certain_tile, run_dikjstra
import random

TILES_MAP = {"g": "door",
             "+": "key",
             "A": "player",
             "1": "bat",
             "2": "spider",
             "3": "scorpion",
             "w": "solid",
             ".": "empty"}

CHAR_MAP = {"door": 'a',
            "key": 'b',
            "player": 'c',
            "bat": 'd',
            "spider": 'e',
            "scorpion": 'f',
            "solid": 'g',
            "empty": 'h'}

"""
Generate a fully connected GVGAI zelda level where the player can reach key then the door.

Args:
    target_enemy_dist: enemies should be at least this far from the player on spawn
"""
class ZeldaProblemHamm(Problem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        self._goal_map_seed = 1 #random.randint(0, 5)
        self._init_map_seed = 1 #random.randint(0, 5)
        self.init_map_path = '../../expert_trajectories_wide/init_maps_lvl{}/starting_map_{}.txt'.format(self._goal_map_seed,
                                                                                                                          self._init_map_seed)
        self.goal_map_path = '../../expert_trajectories_wide/zelda_lvl{}.txt'.format(self._goal_map_seed)

        self._goal_map = self.to_2d_array_level(self.goal_map_path)
        self._init_map = self.to_2d_array_level(self.init_map_path)

        self._hamming_dist = self.hamming_distance_pct(self._goal_map, self._init_map)
        print(f"self._hamming_dist is {self._hamming_dist}")
        super().__init__(start_hamm=self._hamming_dist)
        self._width = 11
        self._height = 7
        self._prob = {"empty": 0.58, "solid":0.3, "player":0.02, "key": 0.02, "door": 0.02, "bat": 0.02, "scorpion": 0.02, "spider": 0.02}
        self._border_tile = "solid"

        self._max_enemies = 5

        self._target_enemy_dist = 4
        self._target_path = 16


        self._rewards = {
            "player": 3,
            "key": 3,
            "door": 3,
            "regions": 5,
            "enemies": 1,
            "nearest-enemy": 2,
            "path-length": 1
        }

    def to_2d_array_level(self, file_name):
        level = []

        with open(file_name, 'r') as f:
            rows = f.readlines()
            for row in rows:
                new_row = []
                for char in row:
                    if char != '\n':
                        new_row.append(TILES_MAP[char])
                level.append(new_row)

        # Remove the border
        truncated_level = level[1: len(level) - 1]
        level = []
        for row in truncated_level:
            new_row = row[1: len(row) - 1]
            level.append(new_row)
        return level

    def hamming_distance_pct(self, map1, map2, onehot=False):
        map1_str = ''
        map2_str = ''
        num_diffs = 0
        num_comps = 0
        if not onehot:
            for row_i in range(len(map1)):
                for col_i in range(len(map1[0])):
                    map1_str += CHAR_MAP[map1[row_i][col_i]]
                    map2_str += CHAR_MAP[map2[row_i][col_i]]
            num_char_diffs = 0
            for idx, char in enumerate(map1_str):
                if char == map2_str[idx]:
                    continue
                else:
                    num_char_diffs += 1

            return round(num_char_diffs / len(map1_str), 6)
        else:
            for row_i in range(len(map1)):
                for col_i in range(len(map1[0])):
                    num_comps += 1
                    map1_idx = np.argmax(np.array(map1[row_i][col_i]))
                    map2_idx = np.argmax(np.array(map2[row_i][col_i]))

                    if map1_idx == map2_idx:
                        continue
                    else:
                        num_diffs += 1
            return round(num_diffs / num_comps, 6)




    """
    Get a list of all the different tile names

    Returns:
        string[]: that contains all the tile names
    """
    def get_tile_types(self):
        return ["empty", "solid", "player", "key", "door", "bat", "scorpion", "spider"]

    """
    Adjust the parameters for the current problem

    Parameters:
        width (int): change the width of the problem level
        height (int): change the height of the problem level
        probs (dict(string, float)): change the probability of each tile
        intiialization, the names are "empty", "solid"
        target_path (int): the current path length that the episode turn when it reaches
        rewards (dict(string,float)): the weights of each reward change between the new_stats and old_stats
    """
    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)

        self._max_enemies = kwargs.get('max_enemies', self._max_enemies)

        self._target_enemy_dist = kwargs.get('target_enemy_dist', self._target_enemy_dist)
        self._target_path = kwargs.get('target_path', self._target_path)

        rewards = kwargs.get('rewards')
        if rewards is not None:
            for t in rewards:
                if t in self._rewards:
                    self._rewards[t] = rewards[t]

    """
    Get the current stats of the map

    Returns:
        dict(string,any): stats of the current map to be used in the reward, episode_over, debug_info calculations.
        The used status are "reigons": number of connected empty tiles, "path-length": the longest path across the map
    """
    def get_stats(self, init_map, goal_map):
        new_hamm_dist = self.hamming_distance_pct(init_map, goal_map, onehot=True)
        return new_hamm_dist


    """
    Get the current game reward between two stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        float: the current reward due to the change between the old map stats and the new map stats
    """
    def get_reward(self, new_stats, old_stats):
        if new_stats < old_stats:
            self._hamming_dist = new_stats
            return 1
        else:
            self._hamming_dist = new_stats
            return -1

    """
    Uses the stats to check if the problem ended (episode_over) which means reached
    a satisfying quality based on the stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        boolean: True if the level reached satisfying quality based on the stats and False otherwise
    """
    def get_episode_over(self):
        return round(self._hamming_dist, 2) == 0.00

    """
    Get any debug information need to be printed

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        dict(any,any): is a debug information that can be used to debug what is
        happening in the problem
    """
    def get_debug_info(self, new_stats, old_stats):
        return {
            "hamm_dist": new_stats
        }

    """
    Get an image on how the map will look like for a specific map

    Parameters:
        map (string[][]): the current game map

    Returns:
        Image: a pillow image on how the map will look like using the binary graphics
    """
    def render(self, map):
        if self._graphics == None:
            self._graphics = {
                "empty": Image.open(os.path.dirname(__file__) + "/zelda/empty.png").convert('RGBA'),
                "solid": Image.open(os.path.dirname(__file__) + "/zelda/solid.png").convert('RGBA'),
                "player": Image.open(os.path.dirname(__file__) + "/zelda/player.png").convert('RGBA'),
                "key": Image.open(os.path.dirname(__file__) + "/zelda/key.png").convert('RGBA'),
                "door": Image.open(os.path.dirname(__file__) + "/zelda/door.png").convert('RGBA'),
                "spider": Image.open(os.path.dirname(__file__) + "/zelda/spider.png").convert('RGBA'),
                "bat": Image.open(os.path.dirname(__file__) + "/zelda/bat.png").convert('RGBA'),
                "scorpion": Image.open(os.path.dirname(__file__) + "/zelda/scorpion.png").convert('RGBA'),
            }
        return super().render(map)
