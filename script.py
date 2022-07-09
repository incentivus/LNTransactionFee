from env.multi_channel import FeeEnv
from utils import initialize, load_data
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def test():
    args = initialize()
    data = load_data()
    env = FeeEnv(data.src,
                 data.trgs,
                 data.channel_ids,
                 data.node_variables,
                 data.providers,
                 data.active_providers,
                 data.initial_active_channels,
                 data.initial_network_dictionary,
                 args)

    model = PPO("MlpPolicy", env, tensorboard_log="./results/", verbose=1)
    model.learn(total_timesteps=50000)
    #model.save("ppo_cartpole")

    model = PPO.load("ppo_cartpole")

    obs = env.reset()







if __name__ == '__main__':
    test()
