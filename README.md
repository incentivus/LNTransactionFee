# Lightning-Network Transaction Fee Solver

This repository contains a code used to conduct experiments reported in the paper.

## How to Use

### Prerequisite

```
Python3
Stable-Baseline 3 contrib
NetworkX 
OpenAI Gym
sb3_contrib
```

- All the other dependencies will be handled in the following installation script via conda.

### Install

```
git clone https://github.com/incentivus/LNTransactionFee.git
cd LNTransactionFee
conda env create --file env.yaml --name lnfee
conda activate lnfee
# in the case of error during creation, use conda update commands:
# conda env update --file env.yaml
```

### Run

```
python -m SORS.scripts.sors --seed {seed} --log_dir ./log/directory/you/want --config_file ./SORS/experiments/sors.gin ./SORS/experiments/envs/delayed_{env_name}.gin # SORS
python -m SORS.scripts.offpolicy_rl --seed {seed} --log_dir ./log/directory/you/want --config_file ./SORS/experiments/sac.gin ./SORS/experiments/envs/delayed_{env_name}.gin # sac baseline
python -m SORS.scripts.offpolicy_rl --seed {seed} --log_dir ./log/directory/you/want --config_file ./SORS/experiments/sac.gin ./SORS/experiments/envs/{env_name}.gin # sac baseline with gt reward
```

You can check the results on `tensorboard`.
```
tensorboard --logdir ./log
```

## Citation

If you find this repository is useful in your research, please cite the paper:

```
@inproceedings{Memarian2021SORS,
  author = {Farzan Memarian and Wonjoon Goo and Rudolf Lioutikov and Scott Niekum and and Ufuk Tocpu},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  title = {Self-Supervised Online Reward Shaping in Sparse-Reward Environments},
  year = {2021}
}
```

## Trouble-shootings

- Mujoco-py related problems: reinstall `mujoco-py`
```
pip uninstall mujoco-py
pip install mujoco-py==2.0.2.13 --no-cache-dir --no-binary :all: --no-build-isolation
```
