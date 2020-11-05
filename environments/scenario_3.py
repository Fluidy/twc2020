"""
Scenario 2: Random acceleration on two roads.
"""
# Load default parameters from Scenario 1
from environments.scenario_1 import env_params

# Update parameters for Scenario 3
env_params['road_distances'] = [200]
env_params['acc_std'] = 0.1
env_params['speed_bound'] = [10, 20]
env_params['stop_pos'] = 750
env_params['stop_time_range'] = [0, 60]

