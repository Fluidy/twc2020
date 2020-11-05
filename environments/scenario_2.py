"""
Scenario 2: Random acceleration on two roads.
"""
# Load default parameters from Scenario 1
from environments.scenario_1 import env_params

# Update parameters for Scenario 2
env_params['road_distances'] = [100, 200]
env_params['acc_std'] = 0.3
env_params['speed_bound'] = [10, 20]
