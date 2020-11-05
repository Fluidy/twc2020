import tensorflow as tf
import numpy as np
import time
import os
from utils import Logger
from algorithms.ddpg.explore import ExploreScheduler
from algorithms.ddpg_pds.trajectory_buffer import TrajectoryBuffer


def train(env, Agent, agent_params, train_params, exp_dir, run_id, use_gpu=False):
    """
    Train DDPG-type agent, including (PDS-)DDPG and (PDS-)TD3
    """
    # CPU/GPU configuration
    config = tf.ConfigProto()
    if use_gpu:
        config.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # use CPU for training and testing

    # Set random seed
    tf.set_random_seed(run_id + train_params['seed'])
    np.random.seed(run_id + train_params['seed'])

    # Copy parameters
    start_train = train_params['start_train']
    save_every = train_params['save_every']
    epoch_len = train_params['epoch_len']
    max_ep_len = train_params['max_ep_len']
    num_roads = len(env.road_distances)
    update_every = train_params['update_every']
    stage = np.array(train_params['stage'])*epoch_len  # Convert the unit of stage from epoch to time step (TF)
    total_steps = int(np.sum(stage))
    train_steps = int(np.sum(stage[:-1]))
    start_virtual = train_params['start_virtual']
    virtual_freq = train_params['virtual_freq']

    reward_scale = agent_params['reward_scale']
    penalty_scale = agent_params['penalty_scale']  # Penalty coefficient, defined as 'lambda' in our paper
    penalty_min, penalty_max = agent_params['penalty_bound']

    # Create agent, explore scheduler, and trajectory buffer
    agent = Agent(env, agent_params)
    explore_scheduler = ExploreScheduler(stage, train_params['explore_scale'])
    trajectory_buffer = TrajectoryBuffer()

    # Create loggers
    run_id = str(run_id)
    log_dir = exp_dir + '/train/run_' + run_id
    epoch_logger = Logger(log_dir + '/road_all')  # Log useful information for monitoring
    road_loggers = [Logger(log_dir + '/road_' + str(road)) for road in range(num_roads)]  # Log for each road
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=train_params['max_to_keep'])  # Model saver

    # Reset
    state = env.reset()
    trajectory_buffer.add_pos(env.pos[0])
    ep_len, ep_energy, ep_return, ep_penalty, ep_stall = 0, 0, 0, 0, 0
    epoch = 1  # Epoch counter (including both real and virtual epoch)
    real_epoch = 1  # Real epoch counter
    real_t = 1  # Real experience counter
    real = True  # Whether the current episode is real episode (instead of virtual episode)
    ep_cnt = 0

    with tf.Session(config=config) as sess:
        agent.initialize(sess)

        tic = time.time()
        # Start training
        for t in range(total_steps):
            noise_scale = explore_scheduler.update_scale()
            if train_params['exp_decay']:
                noise_scale = np.exp(noise_scale)
            next_seg_size = env.get_next_seg_size()  # Get $S_{n_t}$

            action = agent.get_action(sess, [state], noise_scale, [next_seg_size])
            next_state, energy, shortage, done, cut_off = env.step(action)

            penalty = np.clip(penalty_scale*shortage, penalty_min, penalty_max)
            reward = -(energy + penalty)/reward_scale[env.road]

            next2_seg_size = env.get_next_seg_size()  # Get $S_{n_{t+1}}$
            # Add new experience to replay buffer
            agent.buffer.add(state, action, reward, next_state, done,
                             next_seg_size, next2_seg_size, reward_scale[env.road])

            if real:
                # Add user position into trajectory memory
                trajectory_buffer.add_pos(env.pos[0])
                real_t += 1

            # Update actor and critic
            if start_train < t < train_steps and (t + 1) % update_every == 0:
                if 'TD3' in type(agent).__name__ and (t + 1) % agent.policy_delay != 0 and t > 2*start_train:
                    q_loss = agent.learn_batch(sess, train_actor=False)
                    # Log optimization information
                    epoch_logger.add_buffer({'q_loss': q_loss})
                else:
                    q_loss, policy_loss = agent.learn_batch(sess, train_actor=True)
                    # Log optimization information
                    record_dict = {'q_loss': q_loss,
                                   'policy_loss': policy_loss}
                    epoch_logger.add_buffer(record_dict)

            # Update episode information
            if penalty > 0:
                ep_stall += 1
            ep_return += reward
            ep_penalty += penalty
            ep_energy += energy
            ep_len += 1

            # State transit
            state = next_state

            # End of episode
            if cut_off or done or (ep_len == max_ep_len):
                if (done or (ep_len == max_ep_len)) and real:
                    # Log real episode information
                    record_dict = {'ep_len': ep_len,
                                   'ep_return': ep_return,
                                   'ep_penalty': ep_penalty,
                                   'ep_stall': ep_stall,
                                   'ep_energy': ep_energy,
                                   'noise': noise_scale}

                    epoch_logger.add_buffer(record_dict)
                    road_loggers[env.road].add_buffer(record_dict)
                    trajectory_buffer.finish(env.road, env.seg_size)

                ep_cnt += 1

                # Reset
                if ep_cnt % (virtual_freq + 1) == 0 or ep_cnt < start_virtual or t >= np.sum(stage[:2]):
                    # Start a new real episode
                    state = env.reset()
                    real = True

                else:
                    # Start an virtual episode
                    trajectory, road_choice, seg_size = trajectory_buffer.sample()  # Sample historical traces
                    state = env.reset(ext_trajectory=trajectory,
                                      ext_seg_size=seg_size,
                                      ext_road_choice=road_choice)
                    real = False
                ep_len, ep_energy, ep_return, ep_penalty, ep_stall = 0, 0, 0, 0, 0

            # End of epoch
            if (t + 1) % epoch_len == 0:
                # Save model and log to files
                if epoch % save_every == 0 and t < train_steps:
                    saver.save(sess, log_dir + '/checkpoints/model', global_step=epoch,
                               write_meta_graph=False)  # Only save tf variables
                    epoch_logger.dump_log()  # Save epoch log to files
                    [road_loggers[road].dump_log() for road in range(num_roads)]

                print('run: {}, epoch: {}, return: {}, time: {}'.
                      format(run_id, epoch, epoch_logger.get('ep_return'), time.time() - tic))
                epoch += 1
                tic = time.time()

            # Update the moving average in loggers
            if real_t % epoch_len == 0 and real:
                epoch_logger.summarize_buffer(real_epoch)
                [road_loggers[road].summarize_buffer(real_epoch) for road in range(num_roads)]
                real_epoch += 1

    # Save epoch log to files
    epoch_logger.dump_log()
    [road_loggers[road].dump_log() for road in range(num_roads)]

    # Reset the graph after training
    tf.reset_default_graph()

