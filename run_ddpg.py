from algorithms.common.runner import run

"""
Run experiment using DDPG
"""
use_gpu = True
use_ray = True  # 1: Parallel training or test, 0: Serial

is_train = True  # 0: Testing, 1: Training
num_runs = 20

env_name = 'scenario_1'  # The environment to train/test
exp_name = 's1_ddpg_bs1024_end0'  # Where to save the experiment logs

# Runs and models id for testing (only effective when is_train is False)
test_model_id = [100]
test_run_id = [0]  # 0, 1, 2, 3, 4, 5, 6, 7

# Parameters for the DRL agent (only effective when is_train is True)
agent_params = {'critic_lr': 1e-3,  # Critic learning rate
                'actor_lr': 1e-4,  # Actor learning rate
                'critic_size': [200, 200],  # Critic hidden layers size
                'actor_size': [200, 200],  # Actor hidden layers size
                'action_scale': 40,
                'tau': 1e-3,
                'bn': False,  # Whether to use batch normalization
                'mem_size': 1e6,  # Replay buffer size
                'batch_size': 1024,
                'gamma': 1,
                'l2_coeff': 0.,  # L2 coefficient for critic network regularization
                'critic_initial_bias': -15,  # Critic output layer initial bias
                'actor_initial_bias': -1,  # Actor output layer initial bias
                'reward_scale': [1, 1],
                'penalty_scale': 30,
                'penalty_bound': [0, 50]}

# Parameters for training (only effective when setting is_train is True)
train_params = {'update_every': 1,  # Update the critic and actor networks every {} steps
                'start_train': 4000,  # Start training after {} steps
                'stage': [0, 500, 125, 0],
                'explore_scale': [10, 1e-8, 1e-8],
                'save_every': 50,
                'epoch_len': 150*4,  # Number of steps in each epoch
                'max_ep_len': 150,  # Maximum length of a episode
                'max_to_keep': 20,
                'exp_decay': False,
                'start_virtual': 10,
                'virtual_freq': 0,
                'num_cpus': 10,
                'seed': 0}

if __name__ == '__main__':
    # Run experiment
    run('ddpg', exp_name, env_name, agent_params, train_params, use_ray, use_gpu,
        is_train, num_runs, test_run_id, test_model_id)


