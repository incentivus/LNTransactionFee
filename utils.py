import argparse
import argparse
import Simulator.LightningNetworkSimulator.simulator.generating_transactions
import Simulator.LightningNetworkSimulator.simulator.preprocessing



def load_data():
    """
    :return:
    data = dic{src,
                trgs,
                channel_ids,
                node_variables,
                providers,
                active_providers,
                initial_active_channels,
                initial_network_dictionary
            }
    """
    providers_path = './data/merchants.json'
    directed_edges_path = './data/new_data.pkl'
    src_index = 1360
    subgraph_radius = 2
    providers = preprocessing.get_providers(providers_path)
    directed_edges = preprocessing.get_directed_edges(directed_edges_path)
    src, trgs, channel_ids, number_of_channels = preprocessing.select_node(directed_edges, src_index)
    capacities = [2000000] * number_of_channels
    initial_bala


def initialize():
    parser = argparse.ArgumentParser(description='Lightning network environment for multichannel')

    parser.add_argument('--env_name', default='PointMass',
                        help='continuous Gym environment (default: Pendulum-v1)')
    parser.add_argument('--reward_freq', type=int, default=1,
                        help='frequancy of sparse reward')

    args = parser.parse_args()

    return args
