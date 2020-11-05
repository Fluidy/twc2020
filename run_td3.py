from algorithms.common.runner import run
from run_ddpg import agent_params, train_params

"""
Run experiment using TD3
"""
use_gpu = True
use_ray = True  # 1: Parallel training or test, 0: Serial

is_train = True  # 0: testing, 1: training
num_runs = 20

test_model_id = [1000]  # Models to test
test_run_id = [0]  # Runs to test 0, 1, 2, 3, 4, 5, 6, 7

env_name = 'scenario_1'
exp_name = 's1_td3_bs1024_end0'

agent_params.update({
    'policy_delay': 2,
    'target_noise': 1,
    'noise_clip': 10,
})

train_params.update({
    'num_cpus': 10,
})

if __name__ == '__main__':
    run('td3', exp_name, env_name, agent_params, train_params, use_ray, use_gpu,
        is_train, num_runs, test_run_id, test_model_id)

