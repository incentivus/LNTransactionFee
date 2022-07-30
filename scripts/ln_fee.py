from utils import load_data, make_agent, make_env
from stable_baselines3 import SAC, TD3, PPO
from numpy import load
import gym
import numpy as np


def train(env_params, train_params, tb_log_dir, log_dir, seed):
    data = load_data(env_params['node_index'])
    env = make_env(data, env_params, seed)
    model = make_agent(env, train_params['algo'], tb_log_dir)
    model.learn(total_timesteps=train_params['total_timesteps'], tb_log_name="test")
    # TODO : How to save for logging
    model.save(log_dir)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Lightning network environment for multichannel')
    parser.add_argument('--algo', choices=['PPO', 'TRPO', 'SAC', 'TD3', 'A2C', 'DDPG'], default='TRPO')
    parser.add_argument('--tb_log_dir', default='../LNTransactionFee/results/')
    parser.add_argument('--node_index', type=int, default=97851)
    parser.add_argument('--log_dir', default='./LNTransactionFee/results/trained_model/')
    parser.add_argument('--n_seed', type=int, default=1)
    parser.add_argument('--fee_base_upper_bound', type=int, default=10000)
    parser.add_argument('--total_timesteps', type=int, default=3000)
    parser.add_argument('--max_episode_length', type=int, default=200)
    parser.add_argument('--number_of_transaction_types', type=int, default=3)
    parser.add_argument('--counts', default=[10, 10, 10])
    parser.add_argument('--amounts', default=[10000, 50000, 100000])
    parser.add_argument('--epsilons', default=[.6, .6, .6])
    args = parser.parse_args()

    train_params = {'algo': args.algo,
                    'total_timesteps': args.total_timesteps}

    env_params = {'node_index': args.node_index,
                  'fee_base_upper_bound': args.fee_base_upper_bound,
                  'number_of_transaction_types': args.number_of_transaction_types,
                  'max_episode_length': args.max_episode_length,
                  'counts': args.counts,
                  'amounts': args.amounts,
                  'epsilons': args.epsilons}

    for seed in range(args.n_seed):
        train(env_params, train_params,
              tb_log_dir=args.tb_log_dir, log_dir=args.log_dir,
              seed=np.random.randint(low=0, high=1000000))


if __name__ == '__main__':
    main()
