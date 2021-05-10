"""
Run a trained agent and get generated maps
"""
import model
from stable_baselines import PPO2

import time
from utils import make_vec_envs

def infer(game, representation, model_path, **kwargs):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    env_name = '{}-{}-v0'.format(game, representation)
    if game == "binary":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        model.FullyConvPolicy = model.FullyConvPolicySmallMap
        kwargs['cropped_size'] = 10

    env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    agent = PPO2.load(model_path, env=env)

    # print(f"shape: {obs.shape}")
    # print(f"obs: {obs}")
    print(f"env is {env}")
    obs = env.reset()
    dones = False
    wins = 0
    for i in range(kwargs.get('trials', 1)):
        dones = False
        obs = env.reset()
        while not dones:
            print(f"in trial {i}, total wins is {wins}")
            action, _ = agent.predict(obs)
            obs, r, dones, info = env.step(action)
            print(f"r is {r}")
            if info[0]['solved']:
                wins += 1
                dones = True
                print(f"YAYYAYYAYA")
            if kwargs.get('verbose', False):
                # print(info[0])
                pass
            if dones:
                break
        time.sleep(0.01)
    return wins

################################## MAIN ########################################
# game = 'zelda'
game = 'zeldaorderless'
representation = 'wide'
model_path = '/Users/matt/pcgil2/pcgil2/runs/zeldaorderless_wide_zeldaorderless_7_log/best_model.pkl'#.format(game, representation)
kwargs = {
    'change_percentage': 1,
    'trials': 1000,
    'verbose': True,
    'render' : True
}

if __name__ == '__main__':
    infer(game, representation, model_path, **kwargs)
