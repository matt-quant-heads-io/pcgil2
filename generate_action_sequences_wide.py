import sys
import argparse
import copy

from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
from gym.envs.classic_control import rendering
from gym_pcgrl.envs.reps import REPRESENTATIONS
from generate_training_maps import generate_play_trace_wide, to_char_level

from utils import TILES_MAP, int_map_to_onehot
from PIL import Image
from utils import to_2d_array_level, convert_ob_to_int_arr, convert_action_to_npz_format#, to_action_space

import numpy as np
import random
INT_MAP = {
    "empty": 0,
    "solid": 1,
    "player": 2,
    "key": 3,
    "door": 4,
    "bat": 5,
    "scorpion": 6,
    "spider": 7}


# def to_char_level(map, map_path):
#     #add border back
#     map_copy = map.copy()
#     key_list = list(TILES_MAP.keys())
#     val_list = list(TILES_MAP.values())
#
#     for row in range(len(map_copy)):
#         for col in range(len(map_copy[row])):
#             pos = val_list.index(map_copy[row][col])
#             map_copy[row][col] = key_list[pos]
#     arr = np.array(map_copy)
#
#     num_cols = arr.shape[1]
#     border_rows = []
#     border_cols = []
#     for i in range(num_cols):
#         border_rows.append('w')
#
#     border_rows = np.array(border_rows)
#     border_rows = border_rows.reshape((1, num_cols))
#     arr = np.vstack((border_rows, arr))
#     arr = np.vstack((arr, border_rows))
#     # print(arr)
#
#     num_rows = arr.shape[0]
#     for i in range(num_rows):
#         border_cols.append('w')
#
#     border_cols = np.array(border_cols)
#     border_cols = border_cols.reshape((num_rows, 1))
#     # print(border_cols.shape)
#     arr = np.hstack((border_cols, arr))
#     arr = np.hstack((arr, border_cols))
#     return arr


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# import os
# for r in range(50):
#     os.makedirs('expert_trajectories_wide_orderless/init_maps_level{}'.format(r))


# file_path = 'good_levels/Zelda/zelda_lvl{}.txt'
rep = 'wide'
prob = ZeldaProblem()
actions_list = [act for act in list(TILES_MAP.values())]
for idx in range(50):
    map = to_2d_array_level('expert_trajectories_wide/zelda_lvl{}.txt'.format(idx))
    im = prob.render(map)
    im.save('expert_trajectories_wide_orderless/zelda_lvl{}_goalmap.png'.format(idx),
            "PNG")
    pt = []
    for init_map_idx in range(5):
        use_map = map.copy()
        action_space = []
        play_trace = generate_play_trace_wide(map, prob, rep, actions_list, render=False)
        init_action_space = 'expert_trajectories_wide_orderless/init_maps_level{}/action_space_{}.txt'.format(idx, init_map_idx)
        for pt in play_trace:
            action_space.append([pt[-2][0], pt[-2][1], INT_MAP[pt[-1]]])
        action_space = np.array(action_space)
        np.savetxt(init_action_space, action_space, delimiter = ' ', fmt='%s')
        init_map = play_trace[0][0]
        init_map_path = 'expert_trajectories_wide_orderless/init_maps_level{}/starting_map_{}.txt'.format(idx, init_map_idx)
        # print(init_map_path)
        # arr = to_char_level(copy.deepcopy(init_map), init_map_path)
        # Save starting map (destroyed map)
        to_char_level(init_map, dir=init_map_path)
        # arr = to_char_level(init_map.deepcopy(), init_map_path)
        # np.savetxt(init_map_path, arr, delimiter='', fmt='%s')
        im = prob.render(init_map)
        im.save('expert_trajectories_wide_orderless/init_maps_level{}/starting_map_{}.png'.format(idx, init_map_idx), "PNG")