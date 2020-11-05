"""
Training or testing details for the LSTM channel predictor
"""
import os
import numpy as np
from importlib import import_module
from tensorflow.keras.models import load_model
from scipy.io import loadmat, savemat
from environments.env import Env
from utils import save_params, load_params
from predictor.model import create_model, get_training_set, pos_to_snr


def run(env_name, exp_name, is_train, new_data, predictor_params, pre_predict):
    data_dir = 'data/' + env_name
    model_dir = 'experiments/' + exp_name + '/model'
    env_params = import_module('environments.' + env_name).env_params  # import environment parameters
    env = Env(env_params)
    save_params(env_params, 'experiments/' + exp_name, 'env_config')

    if is_train:
        train(env, data_dir, model_dir, new_data, predictor_params)
    else:
        test(env, data_dir, model_dir, pre_predict)


def train(env, data_dir, model_dir, new_data, predictor_params):
    # Copying parameters
    obs_window = predictor_params['obs_window']
    pre_window = predictor_params['pre_window']
    hidden_nodes = predictor_params['hidden_nodes']
    epochs = predictor_params['epochs']
    batch_size = predictor_params['batch_size']
    validation_split = predictor_params['validation_split']
    pos_scale = predictor_params['pos_scale']
    input_feature = predictor_params['input_feature']
    output_feature = predictor_params['output_feature']
    activation = predictor_params['activation']
    assert obs_window % env.seg_TF == 0 and pre_window % env.seg_TF == 0, 'obs_window & pre_window should be divisible by seg_TF'

    # Training
    if not new_data:  # Load existing training data
        try:
            traces = loadmat(data_dir + '/train_data.mat')
            snr_traces = traces['snr_data']
            pos_traces = traces['pos_data']
            v_traces = traces['v']
        except FileNotFoundError:
            print('Training data not found. Generating new training data...')
            new_data = 1

    if new_data:  # Generate new training data
        snr_traces, pos_traces, v_traces = env.gen_traces(num=200,
                                                          length=env.playback_start + env.video_length,
                                                          data_dir=data_dir,
                                                          name='train_data')
    traces = {'snr': np.array(snr_traces),
              'pos': np.array(pos_traces)/pos_scale,
              'v': np.array(v_traces)}

    traces_in = [traces[name] for name in input_feature]
    feature_dim = len(traces_in)
    traces_out = traces[output_feature]
    x_set, y_set = get_training_set(np.array(traces_in).transpose([1, 2, 0]), traces_out, obs_window, pre_window)

    # Create the model and start training
    model = create_model(pre_window, feature_dim, hidden_nodes, activation)
    model.fit(x_set.reshape([-1, obs_window, feature_dim]), y_set.reshape([-1, pre_window, 1]),
              shuffle=True, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    # Save the model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(model_dir + '/predictor_model')

    # Save the predictor parameters to json file
    save_params(predictor_params, model_dir, 'predictor_config')


def test(env, data_dir, model_dir, pre_predict):

    # Load testing data
    try:
        data = loadmat(data_dir + '/test_data.mat')  # user
        pos_data = data['pos_data']
        v_data = data['v_data']
        snr_data = data['snr_data']

    except FileNotFoundError:
        print('Testing data not found. You need to run gen_test_data.py first')
        return

    # Load and copy predictor parameters
    predictor_params = load_params(model_dir, 'predictor_config')
    obs_window = predictor_params['obs_window']
    pre_window = predictor_params['pre_window']
    pos_scale = predictor_params['pos_scale']
    input_feature = predictor_params['input_feature']
    output_feature = predictor_params['output_feature']

    assert obs_window <= env.playback_start or pre_predict is False, 'obs_window should be shorter than env.playback_start when pre_predict is True'

    # Load trained model
    model = load_model(model_dir + '/predictor_model')

    # Start predicting channel
    y_predict_set = []
    y_true_set = []
    pos_predict_set = []
    pos_true_set = []
    for i, pos_trace in enumerate(pos_data):
        print('Predicting channels for trace: {}'.format(i))

        trace = {'snr': snr_data[i],
                 'pos': pos_data[i]/pos_scale,
                 'v': v_data[i]}
        input_trace = np.array([trace[name] for name in input_feature]).transpose()

        if pre_predict:
            t = env.playback_start - obs_window  # Observation begins before video playback
        else:
            t = env.playback_start  # Observation begins after video playback

        y_predict_data, y_true_data = [], []
        pos_predict_data, pos_true_data = [], []
        while t < env.playback_start + env.video_length:
            x = input_trace[t:t + obs_window]
            y_predict = model.predict(x.reshape([-1, obs_window, x.shape[1]])).squeeze()

            if output_feature == 'pos':
                y_predict *= predictor_params['pos_scale']
                pos_predict_data.append(y_predict)
                # Compute SNR based on predicted user position
                y_predict = [pos_to_snr(env, y) for y in y_predict]
            elif output_feature == 'v':
                # Compute user position based on predicted velocity
                y_predict = np.cumsum(y_predict) + pos_trace[t + obs_window - 1]
                # Compute SNR based on predicted user position
                pos_predict_data.append(y_predict)
                y_predict = [pos_to_snr(env, y) for y in y_predict]

            y_true = trace['snr'][t + obs_window:t + obs_window + pre_window]
            pos_true_data.append(pos_trace[t + obs_window:t + obs_window + pre_window])
            y_true_data.append(y_true)
            y_predict_data.append(y_predict)

            t += pre_window

        y_predict_set.append(np.concatenate(y_predict_data))
        y_true_set.append(np.concatenate(y_true_data))
        if output_feature != 'snr':
            pos_predict_set.append(np.concatenate(pos_predict_data))
            pos_true_set.append(np.concatenate(pos_true_data))

    savemat(model_dir + '/predicted_snr.mat', {'predicted_snr': y_predict_set,
                                               'true_snr': y_true_set,
                                               'pre_predict': pre_predict})

    if output_feature is not 'snr':
        savemat(model_dir + '/predicted_pos.mat', {'predicted_pos': pos_predict_set,
                                                   'true_pos': pos_true_set})
