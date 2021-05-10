from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.helper import get_int_prob, get_string_map
import numpy as np
import gym
from gym import spaces
import PIL

"""
The PCGRL GYM Environment
"""
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.helper import get_int_prob, get_string_map
import numpy as np
import gym
from gym import spaces
import PIL

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


def int_map_from_onehot(onehot):
    rowdim, coldim = len(onehot), len(onehot[0])
    int_map = []
    for row in range(rowdim):
        new_row = []
        for col in range(coldim):
            new_row.append(onehot[row][col].index(1))
        int_map.append(new_row)
    # print(f"int_map: {int_map}")
    # print(f"int_map rows: {len(int_map)}")
    # print(f"int_map cols: {len(int_map[0])}")
    return np.array(int_map)

def str_map_to_onehot(str_map):
    new_map = str_map.copy()
    for row_i in range(len(str_map)):
        for col_i in range(len(str_map[0])):
            new_tile = [0]*8
            new_tile[INT_MAP[str_map[row_i][col_i]]] = 1
            new_map[row_i][col_i] = new_tile
    return new_map


"""
The PCGRL GYM Environment
"""
class PcgilOrderlessEnv(gym.Env):
    """
    The type of supported rendering
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    """
    Constructor for the interface.

    Parameters:
        prob (string): the current problem. This name has to be defined in PROBLEMS
        constant in gym_pcgrl.envs.probs.__init__.py file
        rep (string): the current representation. This name has to be defined in REPRESENTATIONS
        constant in gym_pcgrl.envs.reps.__init__.py
    """
    def __init__(self, prob="binary", rep="narrow"):
        self.prob_name = prob

        self._prob = PROBLEMS[prob]()
        self._init_map = self._prob._init_map
        # self._action_sequence_str = self._prob._action_sequence_str
        self._action_sequence_int = self._prob._action_sequence_int
        # self._curr_map = self._prob._init_map.copy()

        self._init_map_oh = str_map_to_onehot(self._prob._init_map)
        # self._curr_map_oh = str_map_to_onehot(self._prob._init_map)


        # print(f"self._goal_map is {self._goal_map}")
        # print(f"self._init_map is {self._init_map}")
        self._rep = REPRESENTATIONS[rep]()
        self._rep.start_nonrandom_map(self._init_map_oh)

        # TODO: What should _rep_stats be?
        self._rep_stats = self._prob.get_stats(None)
        # print(f"init hamm dist is: {self._prob._hamming_dist}")
        self._iteration = 0
        self._changes = 0
        self._max_changes = max(int(0.2 * self._prob._width * self._prob._height), 10)
        self._max_iterations = self._max_changes * self._prob._width * self._prob._height
        self._heatmap = np.zeros((self._prob._height, self._prob._width))

        self.seed()
        self.viewer = None

        self.action_space = self._rep.get_action_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space = self._rep.get_observation_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space.spaces['heatmap'] = spaces.Box(low=0, high=self._max_changes, dtype=np.uint8, shape=(self._prob._height, self._prob._width))




    """
    Seeding the used random variable to get the same result. If the seed is None,
    it will seed it with random start.

    Parameters:
        seed (int): the starting seed, if it is None a random seed number is used.

    Returns:
        int[]: An array of 1 element (the used seed)
    """
    def seed(self, seed=None):
        seed = self._rep.seed(seed)
        self._prob.seed(seed)
        return [seed]

    """
    Resets the environment to the start state

    Returns:
        Observation: the current starting observation have structure defined by
        the Observation Space
    """
    def reset(self, map=None):
        print(f"inside of reset()!!!!!!!!!!!")
        self._changes = 0
        self._iteration = 0
        # self._rep.reset(self._prob._width, self._prob._height, get_int_prob(self._prob._prob, self._prob.get_tile_types()))
        self._prob = PROBLEMS[self.prob_name ]()
        self._init_map = self._prob._init_map
        # self._action_sequence_str = self._prob._action_sequence_str
        self._action_sequence_int = self._prob._action_sequence_int
        self._init_map = self._prob._init_map
        self._init_map_oh = str_map_to_onehot(self._prob._init_map)
        self._rep.start_nonrandom_map(self._init_map_oh)

        # TODO: figure out how to handle the get_stats
        self._rep_stats = len(self._action_sequence_int)
        self._heatmap = np.zeros((self._prob._height, self._prob._width))

        observation = self._rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()
        return observation

    """
    Get the border tile that can be used for padding

    Returns:
        int: the tile number that can be used for padding
    """
    def get_border_tile(self):
        return self._prob.get_tile_types().index(self._prob._border_tile)

    """
    Get the number of different type of tiles that are allowed in the observation

    Returns:
        int: the number of different tiles
    """
    def get_num_tiles(self):
        return len(self._prob.get_tile_types())

    """
    Adjust the used parameters by the problem or representation

    Parameters:
        change_percentage (float): a value between 0 and 1 that determine the
        percentage of tiles the algorithm is allowed to modify. Having small
        values encourage the agent to learn to react to the input screen.
        **kwargs (dict(string,any)): the defined parameters depend on the used
        representation and the used problem
    """
    def adjust_param(self, **kwargs):
        if 'change_percentage' in kwargs:
            percentage = min(1, max(0, kwargs.get('change_percentage')))
            self._max_changes = max(int(percentage * self._prob._width * self._prob._height), 1)
        self._max_iterations = self._max_changes * self._prob._width * self._prob._height
        self._prob.adjust_param(**kwargs)
        self._rep.adjust_param(**kwargs)
        self.action_space = self._rep.get_action_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space = self._rep.get_observation_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space.spaces['heatmap'] = spaces.Box(low=0, high=self._max_changes, dtype=np.uint8, shape=(self._prob._height, self._prob._width))

    """
    Advance the environment using a specific action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        observation: the current observation after applying the action
        float: the reward that happened because of applying that action
        boolean: if the problem eneded (episode is over)
        dictionary: debug information that might be useful to understand what's happening
    """
    def step(self, action):
        # print(f"action is: {action}")
        self._iteration += 1
        #save copy of the old stats to calculate the reward
        old_stats = self._rep_stats
        # update the current state to the new state based on the taken action
        change, x, y = self._rep.update(action)
        self._init_map = self._rep._map
        if change > 0:
            self._changes += change
            self._heatmap[y][x] += 1.0
            # TODO: _prob_get_stats returns the length of the self._action_sequence, and new self._action_sequence, and updates the action sequence
            self._rep_stats, self._action_sequence = self._prob.get_stats(action)
        # calculate the values
        observation = self._rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()
        # TODO: get_reward(self._rep_stats, old_stats) checks if new length of action sequence is less then old length, if so +1 else -1
        reward = self._prob.get_reward(self._rep_stats, old_stats)
        solved = self._prob.get_episode_over()
        done = solved or self._changes >= self._max_changes or self._iteration >= self._max_iterations
        info = self._prob.get_debug_info()
        # info
        info["iterations"] = self._iteration
        info["changes"] = self._changes
        info["max_iterations"] = self._max_iterations
        info["max_changes"] = self._max_changes
        # Use this to determine if the level is solvable !!
        info["solved"] = solved

        # print(f"\nobservation: {observation} \n"
        #       f"reward: {reward}\n"
        #       f"done: {done}\n"
        #       f"info: {info}")
        #return the values
        return observation, reward, done, info

    """
    Render the current state of the environment

    Parameters:
        mode (string): the value has to be defined in render.modes in metadata

    Returns:
        img or boolean: img for rgb_array rendering and boolean for human rendering
    """
    def render(self, mode='human'):
        tile_size=16
        img = self._prob.render(get_string_map(int_map_from_onehot(self._rep._map), self._prob.get_tile_types()))

        img = self._rep.render(img, self._prob._tile_size, self._prob._border_size).convert("RGB")
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            if not hasattr(img, 'shape'):
                img = np.array(img)
            self.viewer.imshow(img)
            return self.viewer.isopen

    # def render_map(self, map, prob, rep, filename='', ret_image=True):
    #     # format image of map for rendering
    #     if not filename:
    #         img = prob.render(map)
    #     else:
    #         img = to_2d_array_level(filename)
    #     img = rep.render(img, tile_size=16, border_size=(1, 1)).convert("RGB")
    #     img = np.array(img)
    #     if ret_image:
    #         return img
    #     else:
    #         from gym.envs.classic_control import rendering
    #         ren = rendering.SimpleImageViewer()
    #         ren.imshow(img)
    #         input(f'')
    #         ren.close()

    """
    Close the environment
    """
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None




class PcgilEnv(gym.Env):
    """
    The type of supported rendering
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    """
    Constructor for the interface.

    Parameters:
        prob (string): the current problem. This name has to be defined in PROBLEMS
        constant in gym_pcgrl.envs.probs.__init__.py file
        rep (string): the current representation. This name has to be defined in REPRESENTATIONS
        constant in gym_pcgrl.envs.reps.__init__.py
    """
    def __init__(self, prob="binary", rep="narrow"):

        self._prob = PROBLEMS[prob]()
        self._goal_map = self._prob._goal_map
        self._init_map = self._prob._init_map
        self._rep = REPRESENTATIONS[rep]()
        self._rep_stats = self._prob._hamming_dist
        print(f"init hamm dist is: {self._prob._hamming_dist}")
        self._iteration = 0
        self._changes = 0
        self._max_changes = max(int(0.2 * self._prob._width * self._prob._height), 10)
        self._max_iterations = self._max_changes * self._prob._width * self._prob._height
        self._heatmap = np.zeros((self._prob._height, self._prob._width))

        self.seed()
        self.viewer = None

        self.action_space = self._rep.get_action_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space = self._rep.get_observation_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space.spaces['heatmap'] = spaces.Box(low=0, high=self._max_changes, dtype=np.uint8, shape=(self._prob._height, self._prob._width))

    """
    Seeding the used random variable to get the same result. If the seed is None,
    it will seed it with random start.

    Parameters:
        seed (int): the starting seed, if it is None a random seed number is used.

    Returns:
        int[]: An array of 1 element (the used seed)
    """
    def seed(self, seed=None):
        seed = self._rep.seed(seed)
        self._prob.seed(seed)
        return [seed]

    """
    Resets the environment to the start state

    Returns:
        Observation: the current starting observation have structure defined by
        the Observation Space
    """
    def reset(self, map=None):
        self._changes = 0
        self._iteration = 0
        # self._rep.reset(self._prob._width, self._prob._height, get_int_prob(self._prob._prob, self._prob.get_tile_types()))
        self._rep.reset(self._init_map)
        self._rep_stats = self._prob.get_stats(self._init_map, self._goal_map)
        self._prob.reset(self._rep_stats)
        self._heatmap = np.zeros((self._prob._height, self._prob._width))

        observation = self._rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()
        return observation

    """
    Get the border tile that can be used for padding

    Returns:
        int: the tile number that can be used for padding
    """
    def get_border_tile(self):
        return self._prob.get_tile_types().index(self._prob._border_tile)

    """
    Get the number of different type of tiles that are allowed in the observation

    Returns:
        int: the number of different tiles
    """
    def get_num_tiles(self):
        return len(self._prob.get_tile_types())

    """
    Adjust the used parameters by the problem or representation

    Parameters:
        change_percentage (float): a value between 0 and 1 that determine the
        percentage of tiles the algorithm is allowed to modify. Having small
        values encourage the agent to learn to react to the input screen.
        **kwargs (dict(string,any)): the defined parameters depend on the used
        representation and the used problem
    """
    def adjust_param(self, **kwargs):
        if 'change_percentage' in kwargs:
            percentage = min(1, max(0, kwargs.get('change_percentage')))
            self._max_changes = max(int(percentage * self._prob._width * self._prob._height), 1)
        self._max_iterations = self._max_changes * self._prob._width * self._prob._height
        self._prob.adjust_param(**kwargs)
        self._rep.adjust_param(**kwargs)
        self.action_space = self._rep.get_action_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space = self._rep.get_observation_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space.spaces['heatmap'] = spaces.Box(low=0, high=self._max_changes, dtype=np.uint8, shape=(self._prob._height, self._prob._width))

    """
    Advance the environment using a specific action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        observation: the current observation after applying the action
        float: the reward that happened because of applying that action
        boolean: if the problem eneded (episode is over)
        dictionary: debug information that might be useful to understand what's happening
    """
    def step(self, action):
        # print(f"action is: {action}")
        self._iteration += 1
        #save copy of the old stats to calculate the reward
        old_stats = self._rep_stats
        # update the current state to the new state based on the taken action
        change, x, y = self._rep.update(action)
        if change > 0:
            self._changes += change
            self._heatmap[y][x] += 1.0
            self._rep_stats = self._prob.get_stats(get_string_map(self._rep._map, self._prob.get_tile_types()))
        # calculate the values
        observation = self._rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()
        reward = self._prob.get_reward(self._rep_stats, old_stats)
        solved  = self._prob.get_episode_over(self._rep_stats,old_stats)
        done = solved or self._changes >= self._max_changes or self._iteration >= self._max_iterations
        info = self._prob.get_debug_info(self._rep_stats,old_stats)
        # info
        info["iterations"] = self._iteration
        info["changes"] = self._changes
        info["max_iterations"] = self._max_iterations
        info["max_changes"] = self._max_changes
        # Use this to determine if the level is solvable !!
        info["solved"] = solved

        # print(f"\nobservation: {observation} \n"
        #       f"reward: {reward}\n"
        #       f"done: {done}\n"
        #       f"info: {info}")
        #return the values
        return observation, reward, done, info

    """
    Render the current state of the environment

    Parameters:
        mode (string): the value has to be defined in render.modes in metadata

    Returns:
        img or boolean: img for rgb_array rendering and boolean for human rendering
    """
    def render(self, mode='human'):
        tile_size=16
        img = self._prob.render(get_string_map(self._rep._map, self._prob.get_tile_types()))
        img = self._rep.render(img, self._prob._tile_size, self._prob._border_size).convert("RGB")
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            if not hasattr(img, 'shape'):
                img = np.array(img)
            self.viewer.imshow(img)
            return self.viewer.isopen

    """
    Close the environment
    """
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None



class PcgrlEnv(gym.Env):
    """
    The type of supported rendering
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    """
    Constructor for the interface.

    Parameters:
        prob (string): the current problem. This name has to be defined in PROBLEMS
        constant in gym_pcgrl.envs.probs.__init__.py file
        rep (string): the current representation. This name has to be defined in REPRESENTATIONS
        constant in gym_pcgrl.envs.reps.__init__.py
    """
    def __init__(self, prob="binary", rep="narrow"):
        self._prob_name = prob
        self._rep_name = rep
        self._prob = PROBLEMS[prob]()
        self._good_level_seed = 1 #random.randint(0, 5)
        self._start_map_seed = 1 #random.randint(0, 5)


        self._init_map_path = '/Users/matt/pcgil2/pcgil2/expert_trajectories_wide_orderless/init_maps_level{}/starting_map_{}.txt'.format(self._good_level_seed, self._start_map_seed)
        # TODO: store self._action_sequence_int
        self._map = self.to_2d_array_level(self._init_map_path)
        self._rep = REPRESENTATIONS[rep]()
        self._rep.start_nonrandom_map(self._map)
        self._rep_stats = None
        self._iteration = 0
        self._changes = 0
        self._max_changes = max(int(0.2 * self._prob._width * self._prob._height), 10)
        self._max_iterations = self._max_changes * self._prob._width * self._prob._height
        self._heatmap = np.zeros((self._prob._height, self._prob._width))

        self.seed()
        self.viewer = None

        self.action_space = self._rep.get_action_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space = self._rep.get_observation_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space.spaces['heatmap'] = spaces.Box(low=0, high=self._max_changes, dtype=np.uint8, shape=(self._prob._height, self._prob._width))




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
    """
    Seeding the used random variable to get the same result. If the seed is None,
    it will seed it with random start.

    Parameters:
        seed (int): the starting seed, if it is None a random seed number is used.

    Returns:
        int[]: An array of 1 element (the used seed)
    """
    def seed(self, seed=None):
        seed = self._rep.seed(seed)
        self._prob.seed(seed)
        return [seed]

    """
    Resets the environment to the start state

    Returns:
        Observation: the current starting observation have structure defined by
        the Observation Space
    """
    def reset(self, map=None):
        self._changes = 0
        self._iteration = 0
        self._prob = PROBLEMS[self._prob_name]()
        self._good_level_seed = 1 #random.randint(0, 5)
        self._start_map_seed = 1 #random.randint(0, 5)


        self._init_map_path = '/Users/matt/pcgil2/pcgil2/expert_trajectories_wide_orderless/init_maps_level{}/starting_map_{}.txt'.format(self._good_level_seed, self._start_map_seed)
        # TODO: store self._action_sequence_int
        self._map = str_map_to_onehot(self.to_2d_array_level(self._init_map_path))
        self._rep = REPRESENTATIONS[self._rep_name]()
        self._rep.start_nonrandom_map(self._map)

        self._rep_stats = self._prob.get_stats(get_string_map(int_map_from_onehot(self._rep._map), self._prob.get_tile_types()))
        self._prob.reset(self._rep_stats)
        self._heatmap = np.zeros((self._prob._height, self._prob._width))

        observation = self._rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()
        return observation

    """
    Get the border tile that can be used for padding

    Returns:
        int: the tile number that can be used for padding
    """
    def get_border_tile(self):
        return self._prob.get_tile_types().index(self._prob._border_tile)

    """
    Get the number of different type of tiles that are allowed in the observation

    Returns:
        int: the number of different tiles
    """
    def get_num_tiles(self):
        return len(self._prob.get_tile_types())

    """
    Adjust the used parameters by the problem or representation

    Parameters:
        change_percentage (float): a value between 0 and 1 that determine the
        percentage of tiles the algorithm is allowed to modify. Having small
        values encourage the agent to learn to react to the input screen.
        **kwargs (dict(string,any)): the defined parameters depend on the used
        representation and the used problem
    """
    def adjust_param(self, **kwargs):
        if 'change_percentage' in kwargs:
            percentage = min(1, max(0, kwargs.get('change_percentage')))
            self._max_changes = max(int(percentage * self._prob._width * self._prob._height), 1)
        self._max_iterations = self._max_changes * self._prob._width * self._prob._height
        self._prob.adjust_param(**kwargs)
        self._rep.adjust_param(**kwargs)
        self.action_space = self._rep.get_action_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space = self._rep.get_observation_space(self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space.spaces['heatmap'] = spaces.Box(low=0, high=self._max_changes, dtype=np.uint8, shape=(self._prob._height, self._prob._width))

    """
    Advance the environment using a specific action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        observation: the current observation after applying the action
        float: the reward that happened because of applying that action
        boolean: if the problem eneded (episode is over)
        dictionary: debug information that might be useful to understand what's happening
    """
    def step(self, action):
        # print(f"action is: {action}")
        self._iteration += 1
        #save copy of the old stats to calculate the reward
        old_stats = self._rep_stats
        # update the current state to the new state based on the taken action
        change, x, y = self._rep.update(action)
        if change > 0:
            self._changes += change
            self._heatmap[y][x] += 1.0
            self._rep_stats = self._prob.get_stats(get_string_map(int_map_from_onehot(self._rep._map), self._prob.get_tile_types()))
        # calculate the values
        observation = self._rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()
        reward = self._prob.get_reward(self._rep_stats, old_stats)
        solved  = self._prob.get_episode_over(self._rep_stats,old_stats)
        done = solved or self._changes >= self._max_changes or self._iteration >= self._max_iterations
        info = self._prob.get_debug_info(self._rep_stats,old_stats)
        # info
        info["iterations"] = self._iteration
        info["changes"] = self._changes
        info["max_iterations"] = self._max_iterations
        info["max_changes"] = self._max_changes
        # Use this to determine if the level is solvable !!
        info["solved"] = solved

        # print(f"\nobservation: {observation} \n"
        #       f"reward: {reward}\n"
        #       f"done: {done}\n"
        #       f"info: {info}")
        #return the values
        return observation, reward, done, info

    """
    Render the current state of the environment

    Parameters:
        mode (string): the value has to be defined in render.modes in metadata

    Returns:
        img or boolean: img for rgb_array rendering and boolean for human rendering
    """
    def render(self, mode='human'):
        tile_size=16
        img = self._prob.render(get_string_map(int_map_from_onehot(self._rep._map), self._prob.get_tile_types()))
        img = self._rep.render(img, self._prob._tile_size, self._prob._border_size).convert("RGB")
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            if not hasattr(img, 'shape'):
                img = np.array(img)
            self.viewer.imshow(img)
            return self.viewer.isopen

    """
    Close the environment
    """
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
