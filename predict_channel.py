"""
Training/testing the channel (SNR) predictor based on previous channels or previous user positions using LSTM
"""
from predictor.runner import run


is_train = False  # True: training, False: testing
env_name = 'scenario_3'
exp_name = 's3_preplan'
pre_predict = False  # The 1st observation window starts before (True) or after (False) playback starts
new_data = True  # True: generate new training set, False: use existing training set

# Channel predictor parameters (only effective when is_train is True, otherwise will load from predictor_config.json)
predictor_params = {
    'input_feature': ['snr', 'pos'],
    'output_feature': 'snr',
    'obs_window': 20,
    'pre_window': 60,
    'hidden_nodes': 200,
    'activation': 'linear',  # linear, relu
    'epochs': 10,
    'batch_size': 32,
    'validation_split': 0.2,
    'pos_scale': 1000,  # User position normalization scale (only effective when use_pos is True)
}

# Run training/testing
run(env_name, exp_name, is_train, new_data, predictor_params, pre_predict)




