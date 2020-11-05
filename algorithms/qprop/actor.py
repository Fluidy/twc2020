import os
from algorithms.qprop.agent import QPropAgent
from algorithms.qprop.buffer import GAEBuffer
from algorithms.qprop.tf_utils import *
from algorithms.ddpg_pds.trajectory_buffer import TrajectoryBuffer


class QPropActor(QPropAgent):
    """
    Q-Prop actor that interacts with the environment
    """
    def __init__(self, env, params, _, env_id, epoch_len, max_ep_len, use_ray, use_cpu):
        self.env = env
        self.use_ray = use_ray
        self.reward_scale = params['reward_scale']
        self.penalty_scale = params['penalty_scale']  # Penalty coefficient, defined as 'lambda' in our paper
        self.penalty_min, self.penalty_max = params['penalty_bound']

        super().__init__(self.env, params, build_network=False)

        self.trajectory_buffer = TrajectoryBuffer()
        self.epoch_len = epoch_len
        self.max_ep_len = max_ep_len
        self.gae_buffer = GAEBuffer(self.state_dim, self.action_dim, self.epoch_len,
                                    self.gamma, self.lam, self.td_lambda_flag)

        # Build value and policy networks
        with tf.variable_scope(str(env_id)):
            action_out, _, mu, _, logp, log_std, _, _ = self._build_policy(self.state_ph)
            self.value = self._build_value()

        self.tensor_to_eval = [action_out, mu, logp, log_std, self.value]

        # Set up operations for synchronizing network params from the global agent
        self.v_vars = tf.global_variables(str(env_id) + '/value')
        self.pi_vars = tf.global_variables(str(env_id) + '/policy')
        self.v_vars_flatten_ph = tf.placeholder(tf.float32, shape=flat_concat(self.v_vars).shape.as_list())
        self.pi_vars_flatten_ph = tf.placeholder(tf.float32, shape=flat_concat(self.pi_vars).shape.as_list())
        self.sync_v_op = assign_vars_from_flat(self.v_vars_flatten_ph, self.v_vars)
        self.sync_pi_op = assign_vars_from_flat(self.pi_vars_flatten_ph, self.pi_vars)

        if self.use_ray:
            config = tf.ConfigProto()
            if use_cpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # -1: use CPU
            else:
                config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = None

    def roll_out(self, global_v_vars, global_pi_vars, log_std_in, real, sess=None):
        if sess is not None:
            self.sess = sess

        # Synchronize network params from the global agent
        self.sess.run([self.sync_pi_op, self.sync_v_op],
                      feed_dict={self.pi_vars_flatten_ph: global_pi_vars,
                                 self.v_vars_flatten_ph: global_v_vars})

        len_list, return_list, energy_list, penalty_list, stall_list, road_list = [], [], [], [], [], []
        if real:
            state = self.env.reset()
            self.trajectory_buffer.add_pos(self.env.pos[0])
        else:
            trajectory, road_choice, seg_size = self.trajectory_buffer.sample()
            state = self.env.reset(ext_trajectory=trajectory,
                                   ext_seg_size=seg_size,
                                   ext_road_choice=road_choice)

        ep_len, ep_energy, ep_penalty, ep_return, ep_stall = 0, 0, 0, 0, 0
        for t in range(self.epoch_len):
            next_seg_size = self.env.get_next_seg_size()
            actor_outs = self.get_actor_outs(state, log_std_in, next_seg_size)
            action, mu, log_p, log_std, value = [actor_outs[0][0], actor_outs[1][0], actor_outs[2][0],
                                                 actor_outs[3][0], actor_outs[4][0]]

            next_state, energy, shortage, done, cutoff = self.env.step(action)
            next2_seg_size = self.env.get_next_seg_size()

            penalty = np.clip(self.penalty_scale * shortage, self.penalty_min, self.penalty_max)
            reward = -(energy + penalty) / self.reward_scale[self.env.road]

            self.gae_buffer.add(state, action, reward, value, log_p, mu, log_std,
                                next_seg_size, self.reward_scale[self.env.road])
            self.buffer.add(state, action, reward, next_state, done,
                            next_seg_size, next2_seg_size, self.reward_scale[self.env.road])

            if real:
                self.trajectory_buffer.add_pos(self.env.pos[0])

            if penalty > 0:
                ep_stall += 1
            ep_return += reward
            ep_penalty += penalty
            ep_energy += energy
            ep_len += 1

            state = next_state

            if done or (ep_len == self.max_ep_len) or (t == self.epoch_len - 1) or cutoff:
                last_value = 0 if done else self.sess.run(self.value, feed_dict={self.state_ph: state.reshape(1, -1),
                                                                                 self.train_value_ph: False})
                self.gae_buffer.finish_path(last_value)

                if done or (ep_len == self.max_ep_len):
                    len_list.append(ep_len)
                    energy_list.append(ep_energy)
                    penalty_list.append(ep_penalty)
                    return_list.append(ep_return)
                    stall_list.append(ep_stall)
                    road_list.append(self.env.road)

                if real:
                    self.trajectory_buffer.finish(self.env.road, self.env.seg_size)
                    state = self.env.reset()
                    self.trajectory_buffer.add_pos(self.env.pos[0])
                else:
                    trajectory, road_choice, seg_size = self.trajectory_buffer.sample()
                    state = self.env.reset(ext_trajectory=trajectory,
                                           ext_seg_size=seg_size,
                                           ext_road_choice=road_choice)
                ep_len, ep_energy, ep_penalty, ep_return, ep_stall = 0, 0, 0, 0, 0
        return len_list, energy_list, penalty_list, return_list, stall_list, road_list

    def get_gae_buffer(self):
        return self.gae_buffer.get()

    def sample(self):
        return self.buffer.sample()

    def get_actor_outs(self, state, log_std_in, next_seg_size):
        return self.sess.run(self.tensor_to_eval, feed_dict={self.state_ph: state.reshape(1, -1),
                                                             self.train_value_ph: False,
                                                             self.log_std_ph: log_std_in})


