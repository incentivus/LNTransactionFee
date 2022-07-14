import argparse
import argparse
import Simulator.LightningNetworkSimulator.simulator.generating_transactions
from Simulator.LightningNetworkSimulator.simulator import preprocessing



def load_data():
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
    directed_edges_path = './data/new_data.pkl'
    src_index = 97851
    subgraph_radius = 2
    data['providers'] = preprocessing.get_providers(providers_path)
    directed_edges = preprocessing.get_directed_edges(directed_edges_path)
    data['src'], data['trgs'], data['channel_ids'], n_channels = preprocessing.select_node(directed_edges, src_index)
    data['capacities'] = [153243, 8500000, 4101029, 5900000, 2500000, 7000000]
    data['initial_balances'] = [153243/2, 8500000/2, 4101029/2, 5900000/2, 2500000/2, 7000000/2]
    channels = []
    for trg in data['trgs']:
        channels.append((data['src'], trg))
    data['active_channels'],\
    data['network_dictionary'],\
    data['node_variables'],\
    data['active_providers'] = preprocessing.get_init_parameters(data['providers'],
                                                                 directed_edges,
                                                                 data['src'], data['trgs'],
                                                                 data['channel_ids'],
                                                                 data['capacities'],
                                                                 data['initial_balances'],
                                                                 subgraph_radius,
                                                                 channels)
    return data


def initialize():
    print('=================initializing parameters================')
    parser = argparse.ArgumentParser(description='Lightning network environment for multichannel')

    parser.add_argument('--env_name', default='FeeEnv',
                        help='Lightning Network Fee simulation (default: FeeEnv)')

    parser.add_argument('--fee_rate_upper_bound', type=int, default=1000,
                        help='upper bound on the fee rate of each channel (default: 1000 msat)')
    parser.add_argument('--fee_base_upper_bound', type=int, default=10000,
                        help='upper bound on the fee base of each channel (default: 1)')
    parser.add_argument('--obs_upper_bound', type=int, default=2000000,
                        help='upper bound on the fee base of each channel (default: 1)')

    parser.add_argument('--max_episode_length', type=int, default=200,
                        help='max time steps in each episode (default: 200)')

    parser.add_argument('--number_of_transaction_types', type=int, default=3,
                        help='number of different transaction types to be simulated')
    parser.add_argument('--counts', default=[10, 10, 10],
                        help='number of different amount of transactions to be simulated')
    parser.add_argument('--amounts', default=[10000, 50000, 100000],
                        help='msat amounts of different amount of transactions to be simulated')
    parser.add_argument('--epsilons', default=[.6, .6, .6],
                        help='merchant ratios')

    parser.add_argument('--seed', type=int, default=12345,
                        help='randomness of simulation')

    args = parser.parse_args()

    return args
