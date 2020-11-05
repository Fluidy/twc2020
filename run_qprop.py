from algorithms.common.runner import run
from run_ddpg import agent_params, train_params

"""
Run experiment using Q-Prop or TRPO
"""
use_gpu = False
use_ray = True  # 0: serial training/testing, 1: parallel

is_train = True  # 0: testing, 1: training
num_runs = 20

test_model_id = [100]  # Models to test
test_run_id = [0]  # Runs to test 0, 1, 2, 3, 4, 5, 6, 7

env_name = 'scenario_1'
exp_name = 's1_qprop_bs1024_end0'

agent_params.update({
    'qprop_flag': 1,  # 0: TRPO, 1: Conservative Q-Prop, 2: Aggressive Q-Prop
    'adv_norm': True,  # Whether to normalize the advantage
    'e_qf_full': False,
    'e_qf_sample_size': 100,
    'delta': 1e-3,  # KL divergence constraint for the policy, similar to learning rate
    'value_lr': 1e-3,
    'value_size': [200, 200],
    'v_batch_size': 32,  # increase this to avoid penalty?
    'train_vf_iters': 10,
    'train_qf_iters': 150*4,
    'return_clip': -1000,
    'td_lambda_flag': True,
    'lam': 0.97,
    'entropy_coeff': 0.,
    'backtrack_iters': 10,
    'backtrack_coeff': 0.5,
    'cg_damping': 0.01,
    'sub_sample': 1,
    'verbose': False,
    'auto_std': False,
    'log_std_initial': None,  # Initial log std of policy network's Gaussian output (depreciated since we update this param manually)
})

# Parameters for training (only effective when setting is_train is True)
train_params.update({
    'epoch_len': 150*4,
    'num_env': 1,
})

if __name__ == '__main__':
    run('qprop', exp_name, env_name, agent_params, train_params, use_ray, use_gpu,
        is_train, num_runs, test_run_id, test_model_id)


