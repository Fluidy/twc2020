import os
import numpy as np
import tensorflow as tf
from utils import Logger
from scipy.io import loadmat


def test(env, Agent, agent_params, exp_dir, run_id, model_id, test_set_dir, use_gpu=True):
    # CPU/GPU configuration
    config = tf.ConfigProto()
    if use_gpu:
        config.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # use CPU for training and testing

    # Set model and test log directories
    run_id = str(run_id)
    model_id = str(model_id)
    model_dir = exp_dir + '/train/run_' + run_id + '/checkpoints/model-' + model_id
    test_log_dir = exp_dir + '/test/run_' + run_id + '/model_' + model_id

    # Create environment and load testing set data
    test_set = loadmat(test_set_dir + '/test_data.mat')
    seg_size = test_set['seg_size_data']
    road = test_set['road_data']
    trajectory = test_set['pos_data']

    # Create agent
    agent = Agent(env, agent_params)
    reward_scale = agent_params['reward_scale']
    penalty_scale = agent_params['penalty_scale']  # Penalty coefficient, defined as 'lambda' in our paper
    penalty_min, penalty_max = agent_params['penalty_bound']

    # Create loggers
    num_roads = len(env.road_distances)
    road_loggers = [Logger(test_log_dir + '/all_road_' + str(road)) for road in range(num_roads)]
    if 'PrePlan' not in type(agent).__name__ and 'NonPredictive' not in type(agent).__name__:
        saver = tf.train.Saver(tf.global_variables())  # model saver
    else:
        saver = None

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        if 'PrePlan' not in type(agent).__name__ and 'NonPredictive' not in type(agent).__name__:
            saver.restore(sess, model_dir)

        # Start testing
        for episode in range(len(trajectory)):
            agent.episode = episode
            state = env.reset(ext_seg_size=seg_size[episode], ext_trajectory=trajectory[episode],
                              ext_road_choice=road[0][episode])
            ep_energy, ep_return, ep_penalty, ep_stall = 0, 0, 0, 0

            step_logger = Logger(test_log_dir + '/episode_' + str(episode + 1) + '_road_' + str(env.road))
            print('Testing episode {} using model {} in run {}'.format(episode + 1, model_id, run_id))
            for t in range(len(trajectory[episode])):
                next_seg_size = env.get_next_seg_size()
                action = agent.get_action(sess, [state], 0, [next_seg_size])[0]
                next_state, energy, shortage, done, _ = env.step(action)

                penalty = np.clip(penalty_scale * shortage, penalty_min, penalty_max)
                reward = -(energy + penalty) / reward_scale[env.road]

                if penalty > 0:
                    ep_stall += 1
                ep_return += reward
                ep_penalty += penalty
                ep_energy += energy

                record_dict = {'channel': state[4],
                               'action': action,
                               'energy': energy,
                               'penalty': penalty}
                step_logger.add_log(t + 1, record_dict)

                # State transition
                state = next_state

                # End of episode
                if done or t == len(trajectory[episode]) - 1:
                    record_dict = {'ep_len': t + 1,
                                   'ep_return': ep_return,
                                   'ep_penalty': ep_penalty,
                                   'ep_stall': ep_stall,
                                   'ep_energy': ep_energy}
                    road_loggers[env.road].add_log(episode + 1, record_dict)
                    step_logger.dump_log()
                    print('return: {}'.format(ep_return))
                    print('--------------------------------')
                    if done:
                        agent.done = True
                        break

    [road_loggers[road].dump_log() for road in range(num_roads)]
    tf.reset_default_graph()
