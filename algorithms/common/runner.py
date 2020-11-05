from utils import save_params, load_params
from importlib import import_module
from environments.env import Env


def run(algorithm_name, exp_name, env_name, agent_params, train_params, use_ray, use_gpu, is_train,
        num_runs=None, test_run_id=None, test_model_id=None):
    """
    Runner for training or testing DRL algorithms
    """
    exp_dir = 'experiments/' + exp_name
    if use_ray:
        try:
            import ray
            ray.init(num_cpus=train_params['num_cpus'], num_gpus=1)
        except ImportError:
            ray = None
            use_ray = 0
            print('Ray is not installed. I will run in serial training/testing mode.')

    """
    Import DRL agent and training function according to algorithm_name
    """
    if algorithm_name in ['ddpg', 'ddpg_pds', 'td3', 'td3_pds']:
        train = import_module('algorithms.ddpg.train').train
        if algorithm_name == 'ddpg':
            Agent = import_module('algorithms.ddpg.agent').DDPGAgent
        elif algorithm_name == 'ddpg_pds':
            Agent = import_module('algorithms.ddpg_pds.agent').PDSDDPGAgent
        elif algorithm_name == 'td3':
            Agent = import_module('algorithms.td3.agent').TD3Agent
        else:
            Agent = import_module('algorithms.td3_pds.agent').PDSTD3Agent

    elif algorithm_name in ['qprop', 'qprop_pds']:
        train = import_module('algorithms.qprop.train').train
        if algorithm_name == 'qprop':
            Agent = import_module('algorithms.qprop.agent').QPropAgent
        else:
            Agent = import_module('algorithms.qprop_pds.agent').PDSQPropAgent
    elif algorithm_name in ['preplan', 'perfect']:
        train = None
        Agent = import_module('algorithms.preplan.agent').PrePlanAgent
    elif algorithm_name == 'non_predictive':
        train = None
        Agent = import_module('algorithms.non_predictive.agent').NonPredictiveAgent
    else:
        print('Unsupported algorithm')
        return

    if is_train:
        """
        Training
        """
        env_params = import_module('environments.' + env_name).env_params
        # Save all the experiment settings to a json file
        save_params([agent_params, train_params, env_params], exp_dir, 'exp_config')

        # Create environment
        env = Env(env_params)

        if use_ray:
            # Parallel training
            train = ray.remote(train)
            train_op = [train.remote(env, Agent, agent_params, train_params, exp_dir, run_id, use_gpu=use_gpu)
                        for run_id in range(num_runs)]
            ray.get(train_op)
        else:
            # Serial training
            [train(env, Agent, agent_params, train_params, exp_dir, run_id, use_gpu=use_gpu)
             for run_id in range(num_runs)]

    else:
        """
        Testing
        """
        # Get test set path
        test_set_dir = 'data/' + env_name

        # Load agent and env parameters from exp_dir
        env_params = load_params('data/' + env_name, 'env_config')
        if algorithm_name != 'perfect':
            if algorithm_name == 'preplan':
                env_params_train = load_params(exp_dir, 'env_config')
            elif algorithm_name == 'non_predictive':
                env_params_train = env_params
            else:
                agent_params, _, env_params_train = load_params(exp_dir, 'exp_config')
            if env_params_train != env_params:
                print('Warning: Testing and training env settings do not match!')

        # Create environment
        env = Env(env_params)

        # Import testing function
        test = import_module('algorithms.common.test').test

        if use_ray:
            # Parallel testing
            test = ray.remote(test)
            test_op = [test.remote(env, Agent, agent_params, exp_dir, run_id, model_id,
                                   test_set_dir=test_set_dir, use_gpu=use_gpu)
                       for run_id in test_run_id for model_id in test_model_id]
            ray.get(test_op)
        else:
            # Serial testing
            [test(env, Agent, agent_params, exp_dir, run_id, model_id,
                  test_set_dir=test_set_dir, use_gpu=use_gpu)
             for run_id in test_run_id for model_id in test_model_id]

