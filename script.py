from env.multi_channel import FeeEnv
from utils import initialize, load_data
from stable_baselines3 import PPO
from stable_baselines3 import SAC

from stable_baselines3.common.env_util import make_vec_env

import numpy as np


def test():
    args = initialize()
    data = load_data()
    env = FeeEnv(data, args)

    """
    reward = 0
    env.reset()
    done = False
    while not done:
        s, r, done, _ = env.step(action=np.array([1000]*4 + [1000]*4))
        reward+=r
        print(r)
    print('rew=',reward)
    print('rew=',1.4e+5)
    """


    model = PPO("MlpPolicy", env, tensorboard_log="./results/", verbose=1)
    model.learn(total_timesteps=150000, tb_log_name="PPO second test(1e6)", eval_freq=600, n_eval_episodes=3)
    model.save("ppo_fee_env")








if __name__ == '__main__':
    test()
