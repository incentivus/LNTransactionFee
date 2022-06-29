import parser

def initialize():
    parser = argparse.ArgumentParser(description='Lightning network environment for multichannel')

    parser.add_argument('--env_name', default='PointMass',
                        help='continuous Gym environment (default: Pendulum-v1)')
    parser.add_argument('--reward_freq', type=int, default=1,
                        help='frequancy of sparse reward')
    parser.add_argument('--has_continuous_action_space', type=bool, default=True)

    parser.add_argument('--env_interactions', type=int, default=250, metavar='N')

    parser.add_argument('--trj_buffer_size', type=int, default=1000, metavar='N')
    parser.add_argument('--initial_trj', type=int, default=100, metavar='N')
    parser.add_argument('--trj_batch_size', type=int, default=10, metavar='N')

    parser.add_argument('--p_update_period', type=int, default=50, metavar='N')
    parser.add_argument('--r_update_period', type=int, default=50, metavar='N')
    parser.add_argument('--p_update_num', type=int, default=10, metavar='N')
    parser.add_argument('--r_update_num', type=int, default=40, metavar='N')

    parser.add_argument('--max_ep_len', type=int, default=80, metavar='N')

    parser.add_argument('--print_freq', type=int, default=1000, metavar='N')
    parser.add_argument('--test_freq', type=int, default=4000, metavar='N')
    parser.add_argument('--test_num', type=int, default=20, metavar='N')

    parser.add_argument('--action_std', type=float, default=0.6, metavar='G')
    parser.add_argument('--action_std_init', type=float, default=0.6, metavar='G')
    parser.add_argument('--action_std_decay_rate', type=float, default=0.05, metavar='G')
    parser.add_argument('--min_action_std', type=float, default=0.1, metavar='G')
    parser.add_argument('--action_std_decay_freq', type=int, default=int(2.5e5), metavar='N')

    parser.add_argument('--update_timestep', type=int, default=int(500*4), metavar='N')
    parser.add_argument('--max_training_timesteps', type=int, default=int(4000*30), metavar='N')

    parser.add_argument('--k_epochs', type=int, default=10, metavar='N')
    parser.add_argument('--eps_clip', type=float, default=0.2, metavar='G')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='G')

    parser.add_argument('--lr_actor', type=float, default=0.0003, metavar='G')
    parser.add_argument('--lr_critic', type=float, default=0.001, metavar='G')
    parser.add_argument('--reward_lr', type=float, default=0.0003, metavar='G')

    parser.add_argument('--random_seed', type=int, default=0, metavar='N')

    args = parser.parse_args()

    return args
