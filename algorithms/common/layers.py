"""
This file contains tensorflow layers including:
1. Multilayer perceptron (MLP)
2. PDS function
3. Reward function
4. Safety layer
"""

import tensorflow as tf
import numpy as np


def mlp(x, hidden_sizes, hidden_activation=tf.nn.relu, output_activation=tf.nn.tanh,
        output_bias_initializer=tf.zeros_initializer(), batch_norm=False, is_train=False):
    """
    A simple MLP with batch normalization option
    """
    # Hidden layers
    for i, h in enumerate(hidden_sizes[:-1]):
        x = tf.layers.dense(x, units=h, activation=None, name='hidden_' + str(i),
                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1/3, distribution='uniform'))   # VarianceScaling(scale=1/3, distribution='uniform'), glorot_uniform()
        if batch_norm:
            x = tf.layers.batch_normalization(x, training=is_train)
        x = hidden_activation(x)

    # Output layer
    out = tf.layers.dense(x, units=hidden_sizes[-1], activation=None, name='out',
                          kernel_initializer=tf.random_uniform_initializer(-1e-4, 1e-4),
                          bias_initializer=output_bias_initializer)
    if output_activation is not None:  # Only use BN for non-linear activation
        if batch_norm:
            out = tf.layers.batch_normalization(out, training=is_train,
                                                beta_initializer=output_bias_initializer)
        out = output_activation(out)

    return out


def pds_func(env, state, next_seg_size, action):
    """
    Compute the PDS state given the state and action
    """
    progress, time_frame, seg_size, buffer, snr = tf.split(state, [1, 1, 1, 1, env.state_dim - 4], axis=1)

    finish_cur_seg = tf.nn.relu(tf.sign(time_frame + 1 - env.seg_TF))  # Whether the current segment is about to finish
    time_frame = (1 - finish_cur_seg) * time_frame  # Equivalent to mod(time_frame, self.num_TF)
    buffer = buffer + action * env.delta_T - finish_cur_seg * seg_size

    stall = tf.nn.relu(tf.sign(next_seg_size - buffer))  # Whether the video stalls in the next TF
    time_frame = (1 - stall) * (time_frame + 1)

    progress = (progress * env.total_size + action * env.delta_T) / env.total_size
    not_done_pds = tf.nn.relu(tf.sign(1 - progress))  # Whether the transmission will not be finished in the next TF

    s_pds = tf.concat((progress, time_frame, next_seg_size, buffer, snr), axis=1)  # PDS
    return s_pds, snr, tf.squeeze(not_done_pds)


def reward_func(env, snr, rate, reward_scale):
    """
    Compute the reward
    """
    snr_0 = tf.gather(snr, [0], axis=1)
    snr_0 = tf.pow(10., (snr_0 / 10))
    x = rate / env.bandwidth * np.log(2)
    y = tf.matmul(tf.pow(x, np.arange(0, len(env.p))), env.p)
    energy = 1 / snr_0 * (tf.nn.relu(y) - x) * env.delta_T
    return tf.squeeze(-energy) / reward_scale


def safety_layer(env, state, next_seg_size, action):
    """
    Return the safe action that avoids video stalling
    """
    _, time_frame, seg_size, buffer, _ = tf.split(state, [1, 1, 1, 1, env.state_dim - 4], axis=1)

    finish_cur_seg = tf.nn.relu(tf.sign(time_frame + 1 - env.seg_TF))
    lower_bound = 1 / env.delta_T * tf.nn.relu(next_seg_size - buffer + finish_cur_seg * seg_size)

    # Add 1e-4 to the lower bound for avoiding stalling caused by insufficient computation accuracy in tensorflow.
    # Upper bound the action for avoiding the overflow of transmit power
    action = tf.maximum(action, tf.minimum(lower_bound + 1e-4, env.max_rate))
    return action
