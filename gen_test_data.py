from environments.env import Env
from utils import save_params
from importlib import import_module

# Create environment
env_name = 'scenario_1'
env_params = import_module('environments.' + env_name).env_params
env = Env(env_params)

test_set_dir = 'data/' + env_name  # Directory for testing data
env.gen_traces(num=1000, length=300, data_dir=test_set_dir, name='test_data', seed=0)  # Generate test data
save_params(env_params, test_set_dir, 'env_config')  # Save environment settings


