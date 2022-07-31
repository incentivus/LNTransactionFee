# LNTransactionFee
Lightning Network Transaction Fee Solver


## How to Use

### Prerequisite


Please make sure you have installed `python3.6` or higher versions of python.


### Install


```
git clone https://github.com/incentivus/LNTransactionFee.git
cd LNTransactionFee
```
All dependencies will be handled using the command :

```pip install -r requirements.txt```


### Run

```
python3 -m scripts.ln_fee --algo PPO
```

### Parameters

| Parameter              | Default | choices                                      |
|------------------------|--------|----------------------------------------------|
| _--algo_               | PPO    | PPO, TRPO, SAC, TD3, A2C, DDPG, TQC, ARS     |
| _--total_timesteps_    | 100000 | Arbitrary Integer                            |
| _--max_episode_length_ | 200    | Arbitrary Integer less than total_timesteps  |
| _--counts_             | [10, 10, 10] | List of Integers                             |
| _--amounts_            | [10000, 50000, 100000] | List of Integers |
| _--epsilons_           | [.6, .6, .6] | List of floats between 0 and 1               |

- You can modify the transaction sampling parameters by changing counts, amounts and epsilons
  - `counts` contains count of each transaction type. 
  - `amounts` contains amount of each transaction type in satoshi.
  - `epsilons` is the ratio of merchants in final sampling.
- Please note that length of counts, amounts and epsilons lists should be the same.




You can check the results on tensorboard.

```
tensorboard --logdir plotting/tb_results/
```

## Trouble-shootings

If you are facing problems with tensorboard, run the command below in terminal :

```
python3 -m tensorboard.main --logdir plotting/tb_results/
```


## Citation



