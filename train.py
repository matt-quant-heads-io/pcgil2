#pip install tensorflow==1.15
#Install stable-baselines as described in the documentation
import sys

import model
from model import FullyConvPolicyBigMap, FullyConvPolicySmallMap, CustomPolicyBigMap, CustomPolicySmallMap
from utils import get_exp_name, max_exp_idx, load_model, make_vec_envs
from stable_baselines import PPO2
from stable_baselines.results_plotter import load_results, ts2xy

import tensorflow as tf
import numpy as np
import os

n_steps = 0
log_dir = 'runs'
best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 10 == 0:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        # print(f"len(x) is {len(x)}")
        if len(x) > 100:
           #pdb.set_trace()
            mean_reward = np.mean(y[-100:])
            # print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, we save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print(f"Saving new best model: idx {n_steps}")
                _locals['self'].model.save(os.path.join(log_dir, f'best_model.pkl'))
            else:
                _locals['self'].model.save(os.path.join(log_dir, 'latest_model.pkl'))
        else:
            # print('{} monitor entries'.format(len(x)))
            pass
    n_steps += 1
    # Returning False will stop training early
    return True


def main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs):
    env_name = '{}-{}-v0'.format(game, representation)
    exp_name = get_exp_name(game, representation, experiment, **kwargs)
    resume = kwargs.get('resume', False)
    if representation == 'wide':
        policy = FullyConvPolicyBigMap
        if game == "sokoban":
            policy = FullyConvPolicySmallMap
    else:
        policy = CustomPolicyBigMap
        if game == "sokoban":
            policy = CustomPolicySmallMap
    if game == "binary":
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        kwargs['cropped_size'] = 10
    n = max_exp_idx(exp_name)
    global log_dir
    if not resume:
        n = n + 1
    log_dir = 'runs/{}_{}_{}'.format(exp_name, n, 'log')
    # os.mkdir(log_dir)
    if not resume:
        os.mkdir(log_dir)
    else:
        model = load_model(log_dir)
    kwargs = {
        **kwargs,
        'render_rank': 0,
        'render': render,
    }
    used_dir = log_dir
    if not logging:
        used_dir = None

    env = make_vec_envs(env_name, representation, log_dir, n_cpu, **kwargs)
    # print(f"\nenv from make_vec_envs: {env}\n")
    #if not resume or model is None:
    if not resume:
        model = PPO2(policy, env, verbose=1, n_steps=16, tensorboard_log="./runs")
        # print(f"policy: {policy}")
        # print(f"\nmake_vec_envs params: \n"
        #       f"env_name: {env_name}\n"
        #       f"representation: {representation},\n"
        #       f"log_dir: {log_dir}, \n"
        #       f"n_cpu: {n_cpu}\n"
        #       f"**kwargs: {kwargs}")
    else:
        if 'orderless' in env_name:
            model = PPO2.load("/Users/matt/pcgil2/pcgil2/runs/zeldaorderless_wide_zeldaorderless_7_log/best_model.pkl")
        else:
            model = PPO2.load("/Users/matt/pcgil2/pcgil2/runs/zeldaham_wide_zeldahamm_4_log/best_model.pkl")
        model.set_env(env)

    if not logging:
        model.learn(total_timesteps=int(steps), tb_log_name=exp_name)
    else:
        model.learn(total_timesteps=int(steps), tb_log_name=exp_name, callback=callback)

################################## MAIN ########################################
#game = 'zeldaham'
game = 'zeldaorderless'
representation = 'wide'
experiment = 'zeldaorderless'
# experiment = 'zeldahamm'
#steps = 1e8
steps = 1e6
render = False
logging = True
# n_cpu = 50
n_cpu = 8
kwargs = {
    'resume': True
}

if __name__ == '__main__':
    main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs)
