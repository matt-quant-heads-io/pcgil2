import os

from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
from gym_pcgrl.envs.reps.wide_rep import WideRepresentation
from gym.envs.classic_control import rendering
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.pcgrl_env import PcgilOrderlessEnv as PcgrlEnv #PcgrlEnv

from utils import TILES_MAP, int_map_to_onehot
from PIL import Image

import numpy as np
import random

from utils import to_2d_array_level, convert_ob_to_int_arr, convert_action_to_npz_format, string_from_2d_arr, int_arr_from_str_arr

# This is for creating the directories
# path_dir = 'expert_trajectories_wide/init_maps_lvl{}'
#
# for idx in range(50):
#     os.makedirs(path_dir.format(idx))

################################################


# This code is for generating the maps

def render_map(map, prob, rep, filename='', ret_image=True):
    # format image of map for rendering
    if not filename:
        img = prob.render(map)
    else:
        img = to_2d_array_level(filename)
    img = rep.render(img, tile_size=16, border_size=(1, 1)).convert("RGB")
    img = np.array(img)
    if ret_image:
        return img
    else:
        ren = rendering.SimpleImageViewer()
        ren.imshow(img)
        input(f'')
        ren.close()

def generate_play_trace_wide(map, prob, rep, actions_list, render=False):
    play_trace = []
    # loop through from 0 to 13 (for 14 tile change actions)
    old_map = map.copy()
    for num_tc_action in range(14):
        new_map = old_map.copy()
        transition_info_at_step = [None, old_map, None, None]
        actions = actions_list.copy()
        row_idx, col_idx = random.randint(0, len(map) - 1), random.randint(0, len(map[0]) - 1)
        new_map[row_idx] = old_map[row_idx].copy()
        transition_info_at_step[2] = (row_idx, col_idx)
        # 1) get the tile type at row_i, col_i and remove it from the list
        old_tile_type = map[row_idx][col_idx]
        transition_info_at_step[3] = old_tile_type
        # 2) remove the tile from the actions_list
        actions.remove(old_tile_type)
        # 3) select an action from the list
        new_tile_type = random.choice(actions)
        # 4) update the map with the new tile type
        new_map[row_idx][col_idx] = new_tile_type
        transition_info_at_step[0] = new_map
        play_trace.insert(0, transition_info_at_step)

        old_map = new_map

        # Render
        if render:
            map_img = render_map(new_map, prob, rep)
            ren = rendering.SimpleImageViewer()
            ren.imshow(map_img)
            input(f'')
            ren.close()
    return play_trace


file_path = 'expert_trajectories_wide/zelda_lvl{}.txt'
output_path = 'expert_trajectories_wide/init_maps_lvl{}/starting_map_{}.txt'
rep = 'wide'
# pcgrl_env = PcgrlEnv("zeldahamm", "{}".format(rep))
# pcgrl_env = PcgrlEnv("zelda", "{}".format(rep))
# TODO: Need to change this for Turtle and Narrow Reps
actions_list = [act for act in list(TILES_MAP.values())]
prob = ZeldaProblem()
rep = WideRepresentation()

# Reverse the k,v in TILES MAP for persisting back as char map .txt format
REV_TILES_MAP = { "door": "g",
                  "key": "+",
                  "player": "A",
                  "bat": "1",
                  "spider": "2",
                  "scorpion": "3",
                  "solid": "w",
                  "empty": "."}


def to_char_level(map, dir=''):
    level = []

    for row in map:
        new_row = []
        for col in row:
            new_row.append(REV_TILES_MAP[col])
        # add side borders
        new_row.insert(0, 'w')
        new_row.append('w')
        level.append(new_row)
    top_bottom_border = ['w'] * len(level[0])
    level.insert(0, top_bottom_border)
    level.append(top_bottom_border)

    level_as_str = []
    for row in level:
        level_as_str.append(''.join(row) + '\n')

    with open(dir, 'w') as f:
        for row in level_as_str:
            f.write(row)


    # with open(file_name, 'r') as f:
    #     rows = f.readlines()
    #     for row in rows:
    #         new_row = []
    #         for char in row:
    #             if char != '\n':
    #                 new_row.append(TILES_MAP[char])
    #         level.append(new_row)


    # Remove the border
    # truncated_level = level[1: len(level) - 1]
    # level = []
    # for row in truncated_level:
    #     new_row = row[1: len(row) - 1]
    #     level.append(new_row)
    # return level


for idx in range(50):
    # get the good level map
    map = to_2d_array_level(file_path.format(idx))
    for init_map_idx in range(30):
        temp_map = map.copy()
        # generate a pod pt using the good level starting state
        play_trace = generate_play_trace_wide(temp_map, prob, rep, actions_list, render=False)
        # This is a test - 1 good level
        goal_map = play_trace[-1][1]
        init_map = play_trace[0][0]
        init_map_path = 'expert_trajectories_wide/init_maps_lvl{}/starting_map_{}.txt'.format(idx, init_map_idx)
        to_char_level(init_map, init_map_path)

        # This is for testing the map via rendering after start_map written back to file
        # render_map(to_2d_array_level(init_map_path),prob, rep, ret_image=False)


