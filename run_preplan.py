from algorithms.common.runner import run
from run_ddpg import agent_params

env_name = 'scenario_3'
exp_name = 's3_preplan'
perfect_channel = False

action_dir = 'experiments/' + exp_name + '/model/planned_actions.mat'
agent_params['action_dir'] = action_dir


# Optimize
try:
    import matlab.engine
except ImportError:
    print('MATLAB engine not installed. Make sure to run optimization in MATLAB first')
else:
    eng = matlab.engine.start_matlab()
    eng.cd('matlab/optimize')
    if perfect_channel:
        eng.opt_perfect(env_name, exp_name, nargout=0)
    else:
        eng.opt_predicted(env_name, exp_name, nargout=0)

# Run experiment
if perfect_channel:
    algorithm_name = 'perfect'
else:
    algorithm_name = 'preplan'
run(algorithm_name, exp_name, env_name, agent_params, train_params=None, use_ray=False, use_gpu=False,
    is_train=False, num_runs=0, test_run_id=[0], test_model_id=[0])

