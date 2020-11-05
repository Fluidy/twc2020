from algorithms.common.runner import run
from run_ddpg_pds import agent_params, train_params

"""
Run experiment using PDS-TD3
"""
use_gpu = False
use_ray = True

is_train = True
num_runs = 20

test_model_id = [800]
test_run_id = [0]

env_name = 'scenario_1'
exp_name = 's1_pds_td3_bs1024_end0_k2'

agent_params.update({
    'policy_delay': 2,
    'target_noise': 1,
    'noise_clip': 10,
    }
)

train_params.update({
    'virtual_freq': 4,
    'stage': [0, 250, 550, 0],
})

if __name__ == '__main__':
    run('td3_pds', exp_name, env_name, agent_params, train_params, use_ray, use_gpu,
        is_train, num_runs, test_run_id, test_model_id)


