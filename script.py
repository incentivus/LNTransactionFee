from env.multi_channel import FeeEnv
from utils import initialize, load_data
from stable_baselines3 import PPO
from stable_baselines3 import SAC, TD3
from sb3_contrib import TRPO
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
    s = env.reset()
    done = False
    cap = np.array([153243.0, 8500000.0, 4101029.0, 5900000.0, 2500000.0, 7000000.0])

    while not done:
        balance = s[0:6]
        action = 1 - np.divide(balance, cap)
        action = np.append(action, [0] * 6)

        action = 2 * action - 1
        action = np.array([1., -1., 0.04999924, -1., -1., -0.21926075,
                           0.8445677, 1., -1., 1., 1., 1.])

        print('nnormal action=', action)
        print('state=', s)

        s, r, done, _ = env.step(action)
        reward += r
        # input()
        print(reward)

    """
    print('rew=', reward)

    reward = 0
    obs = env.reset()
    done = False
    model = PPO.load("PPO_97851_alhpha_env.zip")
    model.set_env(env)

    while not done:
        action, s = model.predict(obs)
        print(action)
        s, r, done, _ = env.step(np.array(action[0]))
        reward += r

    print('rew=', reward)
    """


# obs =env.reset()
# action,s=model.predict(obs)
# print(action)
# print(obs)
# print(s)
# model.learn(total_timesteps=100000, tb_log_name="PPO second test(1e6)",
#             eval_freq=600, n_eval_episodes=3, reset_num_timesteps=False)
# model.save("ppo_fee_env")

# model = TRPO("MlpPolicy", env, tensorboard_log="./results/", verbose=1)
# model.learn(total_timesteps=200000, tb_log_name="TRPO new node 97851", eval_freq=600, n_eval_episodes=3)
# model.save("PPO_97851_alhpha_env")


if __name__ == '__main__':
    test()
