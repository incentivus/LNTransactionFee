import matplotlib.pyplot as plt
import pandas as pd
from utils import create_reward_dataframe_for_plotting
import os

if __name__ == "__main__":

    paths = ['../scripts/results/test_1/events.out.tfevents.1659091274.Aidas-MacBook.73543.0',
             '../scripts/results/test_2/events.out.tfevents.1659107137.Aidas-MacBook.74013.0']

    save_dir = '/Users/aida/PycharmProjects/LNTransactionFee/plotting/plots_and_stats/'
    df = create_reward_dataframe_for_plotting(paths)
    df.to_csv(os.path.join(save_dir, 'stats.csv'))

    print(df)
    steps = df['step']
    mean = df['mean']
    std = df['std']

    plt.plot(steps, mean, 'b-', label='reward')
    plt.fill_between(steps, mean - std, mean + std, color='b', alpha=0.2)
    plt.xlabel('step')
    plt.ylabel('ep_rew_mean')
    plt.title('reward plot')
    plt.savefig(os.path.join(save_dir, 'reward_plot.png'))

