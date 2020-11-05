import numpy as np
from algorithms.common.runner import run
from run_ddpg import agent_params, train_params

"""
Run experiment using PDS-DDPG
"""
use_gpu = False
use_ray = True

is_train = True
num_runs = 20

test_model_id = [250]
test_run_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

env_name = 'scenario_1'
exp_name = 's1_pds_ddpg_bs1024_end0'

agent_params['critic_size'] = [100, 100]
agent_params['actor_size'] = [100, 100]

if env_name == 'scenario_2':
    agent_params['reward_scale'] = [1/2, 1]
agent_params['batch_size'] = 1024
train_params['num_cpus'] = 20
train_params['exp_decay'] = False
train_params['explore_scale'] = [10, 1e-8, 1e-8]
train_params['stage'] = [0, 250, 375, 0]  # [0, 1000, 0, 100] [0, 1000, 500, 1000]
train_params['virtual_freq'] = 0
train_params['start_virtual'] = 10
train_params['start_train'] = 4000
train_params['seed'] = 0
if __name__ == '__main__':
    run('ddpg_pds', exp_name, env_name, agent_params, train_params, use_ray, use_gpu,
        is_train, num_runs, test_run_id, test_model_id)


