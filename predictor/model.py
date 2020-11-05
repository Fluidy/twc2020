from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Activation
from tensorflow.keras.models import Sequential
import numpy as np


def create_model(pre_window, feature_dim, hidden_nodes, activation):
    # Create the Keras model
    model = Sequential()
    model.add(LSTM(units=hidden_nodes, input_dim=feature_dim, return_sequences=False))

    model.add(RepeatVector(pre_window))
    model.add(LSTM(units=hidden_nodes, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Activation(activation))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def get_training_set(traces_in, traces_out, obs_window, pre_window):
    # Prepare the training set
    x_set = []
    y_set = []
    for i in range(len(traces_in)):
        print('Preparing training data from SNR trace: {}'.format(i))
        for j in range(len(traces_in[i]) - obs_window - pre_window):
            x_set.append(traces_in[i][j:j + obs_window])
            y_set.append(traces_out[i][j + obs_window:j + obs_window + pre_window])
    return np.array(x_set), np.array(y_set)


def pos_to_snr(env, pos):
    # Compute SNR based on user position
    x = np.mod(pos, 2*env.cell_radius)
    if x < env.cell_radius:
        dist = np.sqrt(x**2 + env.road_distances[env.road]**2)
    else:
        dist = np.sqrt((2*env.cell_radius - x)**2 + env.road_distances[env.road]**2)
    snr = -35.3 - 37.6 * np.log10(dist) - env.noise
    return snr

