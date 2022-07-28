from env.multi_channel import FeeEnv
from utils import initialize, load_data, make_agent
from stable_baselines3 import SAC, TD3, PPO
from sb3_contrib import TRPO
from numpy import load
import gym


def train(agent, tb_log_dir, log_dir, seed, fee_base_upper_bound, max_episode_length, number_of_transaction_types,
          counts,
          amounts, epsilons):
    data = load_data()
    env = FeeEnv(data, fee_base_upper_bound, max_episode_length, number_of_transaction_types, counts, amounts, epsilons,
                 seed)
    model = make_agent(env, agent, tb_log_dir)
    model.learn(total_timesteps=30000, tb_log_name="test")
    model.save("test")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Lightning network environment for multichannel')
    parser.add_argument('--agent', choices=['PPO', 'TRPO', 'SAC', 'TD3', 'A2C', 'DDPG'], default='TRPO')
    parser.add_argument('--tb_log_dir', default='./results')
    parser.add_argument('--log_dir', default='./results/model')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fee_base_upper_bound', type=int, default=10000)
    parser.add_argument('--max_episode_length', type=int, default=200)
    parser.add_argument('--number_of_transaction_types', type=int, default=3)
    parser.add_argument('--counts', default=[10, 10, 10])
    parser.add_argument('--amounts', default=[10000, 50000, 100000])
    parser.add_argument('--epsilons', default=[.6, .6, .6])
    args = parser.parse_args()
    train(agent=args.agent, tb_log_dir=args.tb_log_dir, log_dir=args.log_dir, seed=args.seed,
          fee_base_upper_bound=args.fee_base_upper_bound, max_episode_length=args.max_episode_length,
          number_of_transaction_types=args.number_of_transaction_types, counts=args.counts, amounts=args.amounts,
          epsilons=args.epsilons)


if __name__ == '__main__':
    from numpy import load

    # data = load('results/evaluations.npz')
    # lst = data.files

    # for item in lst:
    #    print(item)
    #    print(data[item])
    main()
