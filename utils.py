# import os
# # from . import TILES_MAP # Add pcgil to path
# from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
# from gym.envs.classic_control import rendering
# from gym_pcgrl.envs.reps import REPRESENTATIONS
# from gym_pcgrl import wrappers
# from stable_baselines import PPO2
# from stable_baselines.bench import Monitor
# from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
#
# import numpy as np
import os
import re
import glob
import numpy as np
import wrappers
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv

class RenderMonitor(Monitor):
    """
    Wrapper for the environment to save data in .csv files.
    """
    def __init__(self, env, rank, log_dir, **kwargs):
        self.log_dir = log_dir
        self.rank = rank
        self.render_gui = kwargs.get('render', False)
        self.render_rank = kwargs.get('render_rank', 0)
        if log_dir is not None:
            log_dir = os.path.join(log_dir, str(rank))
        Monitor.__init__(self, env, log_dir)

    def step(self, action):
        if self.render_gui and self.rank == self.render_rank:
            self.render()
        return Monitor.step(self, action)




def get_action(obs, env, model, action_type=True):
    action = None
    if action_type == 0:
        action, _ = model.predict(obs)
    elif action_type == 1:
        action_prob = model.action_probability(obs)[0]
        action = np.random.choice(a=list(range(len(action_prob))), size=1, p=action_prob)
    else:
        action = np.array([env.action_space.sample()])
    return action

def make_env(env_name, representation, rank=0, log_dir=None, **kwargs):
    '''
    Return a function that will initialize the environment when called.
    '''
    max_step = kwargs.get('max_step', None)
    render = kwargs.get('render', False)
    def _thunk():
        if representation == 'wide':
            env = wrappers.ActionMapImagePCGRLWrapper(env_name, **kwargs)
        else:
            crop_size = kwargs.get('cropped_size', 28)
            env = wrappers.CroppedImagePCGRLWrapper(env_name, crop_size, **kwargs)
        # RenderMonitor must come last
        if render or log_dir is not None and len(log_dir) > 0:
            env = RenderMonitor(env, rank, log_dir, **kwargs)
        return env
    return _thunk

def make_vec_envs(env_name, representation, log_dir, n_cpu, **kwargs):
    '''
    Prepare a vectorized environment using a list of 'make_env' functions.
    '''
    if n_cpu > 1:
        env_lst = []
        for i in range(n_cpu):
            env_lst.append(make_env(env_name, representation, i, log_dir, **kwargs))
        env = SubprocVecEnv(env_lst)
    else:
        env = DummyVecEnv([make_env(env_name, representation, 0, log_dir, **kwargs)])
    return env

def get_exp_name(game, representation, experiment, **kwargs):
    exp_name = '{}_{}'.format(game, representation)
    if experiment is not None:
        exp_name = '{}_{}'.format(exp_name, experiment)
    return exp_name

def max_exp_idx(exp_name):
    log_dir = os.path.join("./runs", exp_name)
    log_files = glob.glob('{}*'.format(log_dir))
    if len(log_files) == 0:
        n = 0
    else:
        log_ns = [re.search('_(\d+)', f).group(1) for f in log_files]
        n = max(log_ns)
    return int(n)

def load_model(log_dir):
    model_path = os.path.join(log_dir, 'latest_model.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(log_dir, 'latest_model.zip')
    if not os.path.exists(model_path):
        model_path = os.path.join(log_dir, 'best_model.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(log_dir, 'best_model.zip')
    if not os.path.exists(model_path):
        files = [f for f in os.listdir(log_dir) if '.pkl' in f or '.zip' in f]
        if len(files) > 0:
            # selects the last file listed by os.listdir
            model_path = os.path.join(log_dir, np.random.choice(files))
        else:
            raise Exception('No models are saved')
    model = PPO2.load(model_path)
    return model


CHAR_MAP = {"door": 'a',
            "key": 'b',
            "player": 'c',
            "bat": 'd',
            "spider": 'e',
            "scorpion": 'f',
            "solid": 'g',
            "empty": 'h'}

# TODO: This is a placeholder mapping (confirm correct mapping with Ahmed)
ONEHOT_MAP = {"empty": [1, 0, 0, 0, 0, 0, 0, 0],
              "solid": [0, 1, 0, 0, 0, 0, 0, 0],
              "player": [0, 0, 1, 0, 0, 0, 0, 0],
              "key": [0, 0, 0, 1, 0, 0, 0, 0],
              "door": [0, 0, 0, 0, 1, 0, 0, 0],
              "bat": [0, 0, 0, 0, 0, 1, 0, 0],
              "scorpion": [0, 0, 0, 0, 0, 0, 1, 0],
              "spider": [0, 0, 0, 0, 0, 0, 0, 1]}

# TODO: This is a placeholder mapping (confirm correct mapping with Ahmed)
INT_MAP = {
    "empty": 0,
    "solid": 1,
    "player": 2,
    "key": 3,
    "door": 4,
    "bat": 5,
    "scorpion": 6,
    "spider": 7
}

TILES_MAP = {"g": "door",
             "+": "key",
             "A": "player",
             "1": "bat",
             "2": "spider",
             "3": "scorpion",
             "w": "solid",
             ".": "empty"}

def int_arr_from_str_arr(map):
    int_map = []
    for row_idx in range(len(map)):
        new_row = []
        for col_idx in range(len(map[0])):
            new_row.append(INT_MAP[map[row_idx][col_idx]])
        int_map.append(new_row)
    return int_map

def hamming_distance_pct(map1, map2):
    map1_str = ''
    map2_str = ''
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


def str_hamming_distance_pct(map1_str, map2_str):
    num_char_diffs = 0
    for idx, char in enumerate(map1_str):
        if char == map2_str[idx]:
            continue
        else:
            num_char_diffs += 1

    return round(num_char_diffs / len(map1_str), 6)


def string_from_2d_arr(map1):
    map1_str = ''
    for row_i in range(len(map1)):
        for col_i in range(len(map1[0])):
            map1_str += CHAR_MAP[map1[row_i][col_i]]
    return map1_str


# TODO: This is what we have to check!!!
def convert_action_to_npz_format(x, y, action, ob):
    temp_mat = np.zeros((ob.shape[0], ob.shape[1], ob.shape[2]))
    temp_mat[x][y][INT_MAP[action]] = 1
    new_temp_mat = np.ravel(temp_mat)
    return list(new_temp_mat).index(1)


def str_map_to_onehot(str_map):
    new_map = str_map.copy()
    for row_i in range(len(str_map)):
        for col_i in range(len(str_map[0])):
            new_tile = [0]*8
            new_tile[INT_MAP[str_map[row_i][col_i]]] = 1
            new_map[row_i][col_i] = new_tile
    return new_map


def int_map_to_onehot(int_map):
    new_map = []
    for row_i in range(len(int_map)):
        new_row = []
        for col_i in range(len(int_map[0])):
            new_tile = [0]*8
            new_tile[int_map[row_i][col_i]] = 1
            new_row.append(np.array(new_tile))
        new_map.append(np.array(new_row))
    return np.array(new_map)


class RenderMonitor(Monitor):
    """
    Wrapper for the environment to save data in .csv files.
    """
    def __init__(self, env, rank, log_dir, **kwargs):
        self.log_dir = log_dir
        self.rank = rank
        self.render_gui = kwargs.get('render', False)
        self.render_rank = kwargs.get('render_rank', 0)
        if log_dir is not None:
            log_dir = os.path.join(log_dir, str(rank))
        Monitor.__init__(self, env, log_dir)

    def step(self, action):
        if self.render_gui and self.rank == self.render_rank:
            self.render()
        return Monitor.step(self, action)


def make_env(env_name, representation, rank=0, log_dir=None, **kwargs):
    '''
    Return a function that will initialize the environment when called.
    '''
    max_step = kwargs.get('max_step', None)
    render = kwargs.get('render', False)
    def _thunk():
        if representation == 'wide':
            env = wrappers.ActionMapImagePCGRLWrapper(env_name, **kwargs)
        else:
            crop_size = kwargs.get('cropped_size', 28)
            env = wrappers.CroppedImagePCGRLWrapper(env_name, crop_size, **kwargs)
        # RenderMonitor must come last
        if render or log_dir is not None and len(log_dir) > 0:
            env = RenderMonitor(env, rank, log_dir, **kwargs)
        return env
    return _thunk


def make_vec_envs(env_name, representation, log_dir, n_cpu, **kwargs):
    '''
    Prepare a vectorized environment using a list of 'make_env' functions.
    '''
    if n_cpu > 1:
        env_lst = []
        for i in range(n_cpu):
            env_lst.append(make_env(env_name, representation, i, log_dir, **kwargs))
        env = SubprocVecEnv(env_lst)
    else:
        env = DummyVecEnv([make_env(env_name, representation, 0, log_dir, **kwargs)])
    return env


def convert_ob_to_int_arr(ob_dict):
    import numpy as np
    ob = ob_dict['map']
    new_map = []
    for row_i in range(len(ob)):
        new_row = []
        for col_i in range(len(ob[0])):
            new_row.append(INT_MAP[ob[row_i][col_i]])
        new_map.append(new_row)
    return new_map

# Python 3.7
# def to_2d_array_level(file_name):
#     level = []
#
#     with open(f'{os.path.dirname(__file__)}/good_levels_set/{file_name}', 'r') as f:
#         rows = f.readlines()
#         for row in rows:
#             new_row = []
#             for char in row:
#                 if char != '\n':
#                     new_row.append(TILES_MAP[char])
#             level.append(new_row)
#
#     # Remove the border
#     truncated_level = level[1: len(level) - 1]
#     level = []
#     for row in truncated_level:
#         new_row = row[1: len(row) - 1]
#         level.append(new_row)
#     return level


#Python 3.5
def to_2d_array_level(file_name):
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