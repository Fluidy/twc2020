import tensorflow as tf
import numpy as np
import os
import time
from importlib import import_module
from utils import Logger
from algorithms.ddpg.explore import ExploreScheduler


def train(env, Agent, agent_params, train_params, exp_dir, run_id, use_gpu, use_actor_ray=False):
    """
    Train DDPG-type agent
    """
    if use_actor_ray:
        try:
            import ray
            ray.init(num_cpus=train_params['num_env'], num_gpus=1)
        except ImportError:
            ray = None
            use_actor_ray = 0
            print('Ray is not installed. I will run in serial training/testing mode.')

    config = tf.ConfigProto()
    if use_gpu:
        config.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # use CPU for training and testing

    # Set random seed
    tf.set_random_seed(run_id)
    np.random.seed(run_id)

    # Copy parameters
    save_every = train_params['save_every']
    num_env = train_params['num_env']
    max_ep_len = train_params['max_ep_len']
    num_epochs = int(np.sum(train_params['stage']))
    train_epoch = np.sum(train_params['stage'][:-1])
    num_roads = len(env.road_distances)
    start_virtual = train_params['start_virtual']
    virtual_freq = train_params['virtual_freq']

    # Create explore scheduler
    explore_scheduler = ExploreScheduler(train_params['stage'], train_params['explore_scale'])

    # Create the global agent and model saver
    agent = Agent(env, agent_params)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=train_params['max_to_keep'])  # Model saver

    # Create the distributed actors
    epoch_len_per_env = int(train_params['epoch_len']/num_env)
    agent_params['mem_size'] = int(agent_params['mem_size']/num_env)
    agent_params['batch_size'] = int(agent_params['batch_size']/num_env)

    if 'PDS' in type(agent).__name__:
        Actor = import_module('algorithms.qprop_pds.actor').PDSQPropActor
    else:
        Actor = import_module('algorithms.qprop.actor').QPropActor
    if use_actor_ray:
        ParaActor = ray.remote(Actor)
        actors_list = [ParaActor.remote(env, agent_params, None, i, epoch_len_per_env, max_ep_len,
                                        use_actor_ray, use_gpu) for i in range(num_env)]
    else:
        actors_list = [Actor(env, agent_params, None, i, epoch_len_per_env, max_ep_len,
                             use_actor_ray, use_gpu) for i in range(num_env)]

    # Create loggers
    run_id = str(run_id)
    log_dir = exp_dir + '/train/run_' + run_id
    epoch_logger = Logger(log_dir + '/road_all')  # Log useful information for monitoring
    road_loggers = [Logger(log_dir + '/road_' + str(road)) for road in range(num_roads)]  # Log for each road
    real_epoch = 0
    with tf.Session(config=config) as sess:
        agent.initialize(sess)

        for epoch in range(num_epochs):
            tic = time.time()
            if (epoch + 1) % (virtual_freq + 1) == 0 or epoch < start_virtual or epoch >= np.sum(train_params['stage'][:2]):
                real = True
            else:
                real = False

            v_vars, pi_vars = agent.get_vars(sess)
            cur_log_std = [explore_scheduler.update_scale()]
            if not train_params['exp_decay']:
                cur_log_std = np.log(cur_log_std)

            # Roll out
            if use_actor_ray:
                output = ray.get([actor.roll_out.remote(v_vars, pi_vars, cur_log_std, real) for actor in actors_list])
            else:
                output = [actor.roll_out(v_vars, pi_vars, cur_log_std, real, sess) for actor in actors_list]

            # Log useful information
            if real:
                for info in output:
                    for j in range(len(info[0])):
                        record_dict = {'ep_len': info[0][j],
                                       'ep_energy': info[1][j],
                                       'ep_penalty': info[2][j],
                                       'ep_return': info[3][j],
                                       'ep_stall': info[4][j]}
                        epoch_logger.add_buffer(record_dict)
                        road_loggers[info[5][j]].add_buffer(record_dict)

            # Sample from GAE buffer
            if use_actor_ray:
                buffer_data = ray.get([actor.get_gae_buffer.remote() for actor in actors_list])
            else:
                buffer_data = [actor.get_gae_buffer() for actor in actors_list]

            # Update the networks
            if epoch < train_epoch:
                pi_loss, value_loss, log_std = agent.learn(sess, cur_log_std, buffer_data)
                q_loss = 0
                if (epoch + 1) * epoch_len_per_env * num_env > train_params['start_train'] and agent_params['qprop_flag'] > 0:
                    for _ in range(agent_params['train_qf_iters']):
                        if use_actor_ray:
                            replay_data = ray.get(
                                [actor.sample.remote() for actor in actors_list])  # Sample from replay buffer
                        else:
                            replay_data = [actor.sample() for actor in actors_list]
                        q_loss = agent.learn_q(sess, replay_data)

                record_dict = {'pi_loss': pi_loss,
                               'value_loss': value_loss,
                               'q_loss': q_loss,
                               'log_std': log_std}
                epoch_logger.add_buffer(record_dict)

            if real:
                real_epoch += num_env
                epoch_logger.summarize_buffer(real_epoch)
                [road_loggers[road].summarize_buffer(real_epoch) for road in range(num_roads)]

            print('run: {}, epoch: {}, return: {}, log_std: {}, time: {}'.
                  format(run_id, (epoch + 1), epoch_logger.get('ep_return'), log_std, time.time() - tic))

            # Save model and log to files
            if (epoch + 1) % save_every == 0 and epoch < train_epoch:
                saver.save(sess, log_dir + '/checkpoints/model', global_step=(epoch + 1),
                           write_meta_graph=False)  # Only save tf variables
                epoch_logger.dump_log()  # Save epoch log to files
                [road_loggers[road].dump_log() for road in range(num_roads)]

    epoch_logger.dump_log()  # Save epoch log to files
    [road_loggers[road].dump_log() for road in range(num_roads)]
    tf.reset_default_graph()
