import gym
from gym import spaces
import numpy as np


class FeeEnv(gym.Env):
    def __init__(self,
                 node_variables,
                 providers,
                 active_providers,
                 initial_active_channels,
                 initial_network_dictionary,
                 args):
        """
        ### Description

        This environment corresponds to the LIGHTNING NETWORK simulation. A source node is chosen and a local network
        around that node with radius 2 is created and at each time step, a certain number of transitions are being simulated.

        ### Scales

        We are using the following scales for simulating the real world Lightning Network:

        - Fee Rate:                                        - Base Fee:
        - Transaction amounts:                             - Reward(income):
        - Capacity:                                        - Balance:

        ### Action Space

        The action is a `ndarray` with shape `(2*n_channel,)` which can take values `[0,upper bound]`
        indicating the fee rate and base fee of each channel starting from source node.

        | dim       | action                 | dim        | action                |
        |-----------|------------------------|------------|-----------------------|
        | 0         | fee rate channel 0     | 0+n_channel| fee base channel 0    |
        | ...       |        ...             | ...        |         ...           |
        | n_channel | fee rate last channel  | 2*n_channel| fee base last channel |

        ### Observation Space

        The observation is a `ndarray` with shape `(2*n_channel,)` with the values corresponding to the balance of each
        channel and also accumulative transaction amounts in each time steps.

        | dim       | observation            | dim        | observation                 |
        |-----------|------------------------|------------|-----------------------------|
        | 0         | balance channel 0      | 0+n_channel| sum trn amount channel 0    |
        | ...       |          ...           | ...        |            ...              |
        | n_channel | balance last channel   | 2*n_channel| sum trn amount last channel |

        ### Rewards

        Since the goal is to maximize the return in long term, reward is sum of incomes from fee payments of each channel.
        Reward scale is Sat in order to control the upperbound.

        """


        # Source node
        self.src = args.src
        self.n_channel = args.n_channel

        # Base fee and fee rate for each channel of src
        self.action_space = spaces.Box(low=-1, high=+1, shape=(2*self.n_channel,), dtype=np.float32)
        self.fee_rate_upper_bound = args.fee_rate_upper_bound
        self.fee_base_upper_bound = args.fee_base_upper_bound

        # Balance and transaction amount of each channel
        self.observation_space = spaces.Box(low=0, high=1, shape=(2*self.n_channel,), dtype=np.float32)

        # Initial values of each channel
        self.initial_balances = args.initial_balances
        self.capacities = args.capacities
        self.state = np.append(self.initial_balances, np.zeros(shape=(self.n_channel,)))

        self.time_step = 0
        self.max_episode_length = args.max_episode_length
        self.balance_ratio = 0.1

        # Simulator variables
        self.node_variables = node_variables
        self.providers = providers  # Network merchants
        self.active_providers = active_providers  # Subnetwork merchants
        self.active_channels = initial_active_channels  # Channels with changing balance
        self.network_dictionary = initial_network_dictionary  # For saving subgraph status

        self.simulator = simulator.simulator(self.src, self.trg, self.channel_id,  # channel datat
                                             self.active_channels,  # network structure
                                             self.network_dictionary,
                                             self.providers,
                                             count=self.count,
                                             amount=self.amount,
                                             epsilon=self.epsilon,
                                             node_variables=self.node_variables,
                                             active_providers=self.active_providers,
                                             fixed_transactions=False)

        # Logging results
        self.buffer = []
        self.avg_buffer = []
        self.k_buffer = []
        self.k_buffer_avg = []
        self.alpha_buffer = []
        self.alpha_buffer_avg = []
        self.beta_buffer = []
        self.beta_buffer_avg = []

        self.seed()

    def step(self, action):

        # Rescaling the action vector
        action[0:self.n_channel] = .5 * self.fee_rate_upper_bound * action +\
                                   .5 * self.fee_rate_upper_bound
        action[self.n_channel:2*self.n_channel] = .5 * self.fee_base_upper_bound * action +\
                                                  .5 * self.fee_base_upper_bound

        # Running simulator for a certain time interval
        balances, transaction_amounts, transaction_numbers = self.ln_sim(self.src, action)
        self.time_step += 1

        reward = np.sum(np.multiply(action[0:self.n_channel], transaction_amounts) +\
                        np.multiply(action[self.n_channel:2*self.n_channel], transaction_numbers))

        info = {}
        info["TimeLimit.truncated"] = True if self.step_count >= self.max_episode_length

        done = self.time_step >= self.max_episode_length or \
               np.sum(balances)/np.sum(self.capacities) <= self.balance_ratio

        self.state = np.append(balances, transaction_amounts)

        self.avg_buffer.append(reward)

        return self.state, reward, done, info

    def ln_sim(self, action):  # k(int) tx(float) r(npArray)

        self.simulator.set_node_fee(self.src, action)

        transactions = self.simulator.run_simulation(self.count, self.amount, action)
        k, tx, rebalancing_fee, rebalancing_type = self.simulator.get_coeffiecients(action, transactions, self.src,
                                                                                    self.trg, self.channel_id,
                                                                                    self.amount,
                                                                                    self.onchain_transaction_fee)
        other_k = self.simulator.get_k(trg, src, channel_id, transactions)

        capacity = self.simulator.get_capacity(self.src, self.trg, self.channel_id)
        balance = self.simulator.get_balance(self.src, self.trg, self.channel_id)

        return k, tx, other_k, rebalancing_fee, rebalancing_type, capacity, balance

    def get_buffers(self):
        return self.buffer, self.alpha_buffer_avg, self.beta_buffer_avg, self.k_buffer_avg

    def reset(self):
        self.time_step = 0
        self.state = np.append(self.initial_balances, np.zeros(shape=(self.n_channel,)))

        return np.array([self.state], dtype=np.float32)
