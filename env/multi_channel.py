import gym
from gym import spaces
import numpy as np


class env(gym.Env):
    def __init__(self, args):

        # Base fee and fee rate for each channel of src
        self.action_space = spaces.Box(low=-1,
                                       high=+1,
                                       shape=(2*args.n_channel,),
                                       dtype=np.float32)

        # Balance and transaction amount of each channel
        self.observation_space = spaces.Box(low=-1,
                                            high=1,
                                            shape=(2*args.n_channel+args.n_channel,),
                                            dtype=np.float32)

        # Initial values of each channel
        self.initial_balances = args.initial_balance
        self.initial_capacitys = args.capacity

        self.src = src
        self.trg = trg
        self.channel_id = channel_id

        self.node_variables = node_variables
        self.providers = providers
        self.active_providers = active_providers
        self.active_channels = initial_active_channels  # network structure
        self.network_dictionary = initial_network_dictionary

        self.state = 0  # constant balance
        self.step_count = 0
        self.episod_count = 0
        self.epsiod_runout = epsiod_runout
        # self.time_step = time_step
        self.np_random = None
        self.onchain_transaction_fee = 5e4

        self.count = count
        self.amount = amount  # satoshi
        self.epsilon = 0.6
        self.lagrange_multiplier = lagrange_multiplier
        self.reward_normalizer = reward_normalizer

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
        """
        re-scaling the actions
        """
        action[0] = action[0] * 1000 + 1000  # fee_rate_milli_satoshi: [0,2000]
        action[1] = action[1] * 5000 + 5000  # fee_base_milli_satoshi:  [0,10 000]
        action = np.array([action[0], action[1], 0])

        self.step_count = self.step_count + 1
        k, tx, other_k, rebalancing_fee, rebalancing_type, capacity, balance = self.ln_sim(action)

        reward = (action[0] * tx + action[1] * k) * self.reward_normalizer
        done = bool(self.step_count >= self.epsiod_runout)

        self.state = 0

        self.avg_buffer.append(reward)
        self.alpha_buffer.append(action[0])
        self.beta_buffer.append(action[1])
        self.k_buffer.append(k)

        if done:
            discounted_factor = 0.99
            disounted_return = sum(
                (discounted_factor ** i) * self.avg_buffer[i] for i in range(len(self.avg_buffer)))
            self.buffer.append(disounted_return)
            self.alpha_buffer_avg.append(mean(self.alpha_buffer))
            self.beta_buffer_avg.append(mean(self.beta_buffer))
            self.k_buffer_avg.append(mean(self.k_buffer))

            self.avg_buffer = []
            self.alpha_buffer = []
            self.beta_buffer = []
            self.k_buffer = []

        return np.array([self.state], dtype=np.float32), reward, done, {}

    def ln_sim(self, action):  # k(int) tx(float) r(npArray)

        self.simulator.set_node_fee(src, trg, channel_id, action)

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

    def reset(self, seed: Optional[int] = None):
        if seed is not None or self.np_random is None:
            self.np_random, seed = seeding.np_random(seed)
        self.episod_count = self.episod_count + 1
        self.state = 0
        self.step_count = 0
        return np.array([self.state], dtype=np.float32)

    def render(self, mode="human"):
        pass

    def close(self):
        pass