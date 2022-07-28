from simulator import preprocessing


def make_agent(env, agent, tb_log_dir):
    policy = "MlpPolicy"
    # create model
    if agent == "PPO":
        from stable_baselines3 import PPO
        model = PPO(policy, env, verbose=1, tensorboard_log=tb_log_dir)
    elif agent == "TRPO":
        from sb3_contrib import TRPO
        model = TRPO(policy, env, verbose=1, tensorboard_log=tb_log_dir)
    elif agent == "SAC":
        from stable_baselines3 import SAC
        model = SAC(policy, env, verbose=1, tensorboard_log=tb_log_dir)
    elif agent == "DDPG":
        from stable_baselines3 import DDPG
        model = DDPG(policy, env, verbose=1, tensorboard_log=tb_log_dir)
    elif agent == "TD3":
        from stable_baselines3 import TD3
        model = TD3(policy, env, verbose=1, tensorboard_log=tb_log_dir)
    else:
        raise NotImplementedError()

    return model


def load_data(node):
    """
    :return:
    data = dict{src: node chosen for simulation  (default: int)
                trgs: nodes src is connected to  (default: list)
                channel_ids: channel ids of src node [NOT USED YET]  (default: list)
                initial_balances: initial distribution of capacity of each channel  (default: list)
                capacities: capacity of each channel  (default: list)
                node_variables: ???  (default: )
                providers: Merchants of the whole network  (default: ?)
                active_providers: Merchants of the local network around src  (default: ?)
                active_channels: channel which their balances are being updated each timestep  (default: ?)
                network_dictionary: whole network data  (default: dict)
            }
    """
    print('==================Loading Network Data==================')
    data = {}
    providers_path = './data/merchants.json'
    directed_edges_path = './data/data.json'
    src_index = node
    subgraph_radius = 2
    data['providers'] = preprocessing.get_providers(providers_path)
    directed_edges = preprocessing.get_directed_edges(directed_edges_path)
    data['src'], data['trgs'], data['channel_ids'], n_channels = preprocessing.select_node(directed_edges, src_index)
    data['capacities'] = [153243, 8500000, 4101029, 5900000, 2500000, 7000000]
    data['initial_balances'] = [153243 / 2, 8500000 / 2, 4101029 / 2, 5900000 / 2, 2500000 / 2, 7000000 / 2]
    # data['initial_balances'] = [153243, 0, 0, 0, 0, 0]
    channels = []
    for trg in data['trgs']:
        channels.append((data['src'], trg))
    data['active_channels'], \
    data['network_dictionary'], \
    data['node_variables'], \
    data['active_providers'] = preprocessing.get_init_parameters(data['providers'],
                                                                 directed_edges,
                                                                 data['src'], data['trgs'],
                                                                 data['channel_ids'],
                                                                 data['capacities'],
                                                                 data['initial_balances'],
                                                                 subgraph_radius,
                                                                 channels)
    return data
