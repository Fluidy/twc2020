from algorithms.common.runner import run
from run_ddpg import agent_params

env_name = 'scenario_3'
exp_name = 's3_non_predictive'

# Run experiment
run('non_predictive', exp_name, env_name, agent_params=agent_params, train_params=None, use_ray=False, use_gpu=False,
    is_train=False, num_runs=0, test_run_id=[0], test_model_id=[0])

