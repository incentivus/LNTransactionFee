from env.multi_channel import FeeEnv
from utils import initialize, load_data
from stable_baselines3 import PPO
from stable_baselines3 import SAC, TD3

from stable_baselines3.common.env_util import make_vec_env

import numpy as np


def normal(action):
    action[0:4] = .5 * 1000 * action[0:4] + \
                  .5 * 1000
    action[4:2 * 4] = .5 * 10000 * action[
                                   4:2 * 4] + \
                      .5 * 10000
    return action


def test():
    args = initialize()
    data = load_data()
    env = FeeEnv(data, args)

    reward = 0
    env.reset()
    done = False
    gamma = 1

    while not done:
        s, r, done, _ = env.step(action=np.array([1] * 20 + [0.1] * 20))
        reward += r

    print('rew=', reward)

    reward = 0
    obs = env.reset()
    done = False
    model = PPO.load("ppo_fee_env")
    model.set_env(env)
    while not done:
        action, s = model.predict(obs)
        s, r, done, _ = env.step(np.array(action[0]))
        reward += r

    print('rew=', reward)

    """
    obs =env.reset()
    action,s=model.predict(obs)
    print(action)
    print(obs)
    print(s)
    model.learn(total_timesteps=100000, tb_log_name="PPO second test(1e6)",
                eval_freq=600, n_eval_episodes=3, reset_num_timesteps=False)
    model.save("ppo_fee_env")

   # model = PPO("MlpPolicy", env, tensorboard_log="./results/", verbose=1)
   # model.learn(total_timesteps=150000, tb_log_name="PPO alpha test", eval_freq=600, n_eval_episodes=3)
   # model.save("ppo_alhpha_env")
   """


if __name__ == '__main__':
    test()
