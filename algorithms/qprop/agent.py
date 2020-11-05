from algorithms.common.layers import mlp
from algorithms.ddpg.agent import DDPGAgent
from algorithms.qprop.tf_utils import *
from algorithms.qprop.np_utils import *


class QPropAgent(DDPGAgent):
    """
    The global Q-Prop agent that collects experience from distributed actor
    and trains the global policy and value networks, as well as the Q-network
    """

    def __init__(self, env, params, build_network=True, *args):
        super().__init__(env, params, build_network=False)

        # TRPO parameters
        self.td_lambda_flag = params['td_lambda_flag']
        self.lam = params['lam']
        self.entropy_coeff = params['entropy_coeff']
        self.value_size = params['value_size']
        self.value_lr = params['value_lr']
        self.train_vf_iters = params['train_vf_iters']
        self.v_batch_size = params['v_batch_size']
        self.delta = params['delta']
        self.cg_damping = params['cg_damping']
        self.sub_sample = params['sub_sample']
        self.backtrack_iters = params['backtrack_iters']
        self.backtrack_coeff = params['backtrack_coeff']
        self.verbose = params['verbose']
        self.adv_norm = params['adv_norm']
        self.return_clip = params['return_clip']

        # Q-Prop parameters
        self.qprop_flag = params['qprop_flag']
        self.e_qf_full = params['e_qf_full']
        self.e_qf_sample_size = params['e_qf_sample_size']
        self.train_qf_iters = params['train_qf_iters']
        self.log_std_initial = params['log_std_initial']
        self.auto_std = params['auto_std']

        # Placeholders
        self.advantage_ph = tf.placeholder(tf.float32, shape=[None])
        self.return_ph = tf.placeholder(tf.float32, shape=[None])
        self.logp_old_ph = tf.placeholder(tf.float32, shape=[None])
        self.old_mu_ph = tf.placeholder(tf.float32, shape=[None, self.action_dim])
        self.old_log_std_ph = tf.placeholder(tf.float32, shape=[None, self.action_dim])
        self.eta_ph = tf.placeholder(tf.float32, shape=[None])
        self.train_value_ph = tf.placeholder(tf.bool, shape=())

        self.all_phs = [self.state_ph, self.action_ph, self.advantage_ph, self.return_ph,
                        self.logp_old_ph, self.old_mu_ph, self.old_log_std_ph]

        self.log_std_ph = tf.placeholder(tf.float32, shape=self.action_dim)

        if build_network:
            with tf.variable_scope('global'):
                [self.action_out, _, self.mu, self.logp_exp, self.logp, self.log_std,
                 self.d_kl, self.entropy] = self._build_policy(self.state_ph)
                self.value = self._build_value()
            with tf.variable_scope('global', reuse=True):
                _, action_tar_samples, mu_tar, _, _, _, _, _ = self._build_policy(self.next_state_ph)

            with tf.variable_scope('main'):
                self.q_value, self.q_e_value = self._build_critic(state=self.state_ph,
                                                                  policy_action=self.mu,
                                                                  action_in=self.action_ph,
                                                                  train_critic=self.train_critic_ph)
            with tf.variable_scope('target'):
                _, self.q_e_tar = self._build_critic(state=self.next_state_ph,
                                                     policy_action=mu_tar,
                                                     action_in=mu_tar)
            with tf.variable_scope('target', reuse=True):
                next_state_samples = tf.repeat(self.next_state_ph, self.e_qf_sample_size, axis=0)
                self.q_tar, _ = self._build_critic(state=next_state_samples,
                                                   policy_action=action_tar_samples,
                                                   action_in=action_tar_samples)
            # Get losses
            self.pi_loss, self.value_loss, self.q_loss = self._get_losses()

            # Get training operations
            [self.v_all_vars_flatten, self.pi_all_vars_flatten,
             self.optimize_v, self.optimize_q, self.init_tar, self.update_tar,
             self.pi_vars_flatten, self.pi_gradient, self.hvp, self.x_ph,
             self.pi_vars_ph, self.set_pi_op, self.cv] = self._get_training_ops()

    def _get_losses(self):
        # DDPG loss
        if self.e_qf_full:
            self.e_q_tar = tf.reduce_mean(tf.reshape(self.q_tar, shape=[-1, self.e_qf_sample_size]), axis=1)
        else:
            self.e_q_tar = self.q_e_tar
        ddpg_loss = - tf.reduce_mean(self.q_e_value * self.eta_ph)

        # Policy loss
        ratio = tf.exp(self.logp_exp - self.logp_old_ph)
        pi_loss = - tf.reduce_mean(ratio * self.advantage_ph) + ddpg_loss - self.entropy_coeff * self.entropy

        # Value loss
        value_loss = tf.reduce_mean(tf.square(self.return_ph - self.value))

        # Q-value loss
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables('main/critic')
                            if 'kernel' in v.name])
        y = tf.stop_gradient(self.reward_ph + self.gamma * (1 - self.done_ph) * self.e_q_tar)
        q_loss = tf.reduce_mean(tf.square(y - self.q_value)) + self.l2_coeff * l2_loss
        return pi_loss, value_loss, q_loss

    def _get_training_ops(self):
        # Get global vars
        v_all_vars_flatten = flat_concat(tf.global_variables('global/value'))
        pi_all_vars_flatten = flat_concat(tf.global_variables('global/policy'))

        # Initialize and update target Q-network
        init_tar = [v_tar.assign(v_main) for v_main, v_tar
                    in zip(tf.global_variables('main'), tf.global_variables('target'))]
        update_tar = [v_tar.assign(self.tau * v_main + (1 - self.tau) * v_tar) for v_main, v_tar
                      in zip(tf.global_variables('main'), tf.global_variables('target'))]

        # Operations for moving average and std updated in batch normalization
        v_bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='global/value')
        pi_bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='global/policy')
        q_bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='main/critic')

        # Train value network
        v_optimizer = tf.train.AdamOptimizer(learning_rate=self.value_lr)
        with tf.control_dependencies(v_bn_ops):
            optimize_v = v_optimizer.minimize(self.value_loss,
                                              var_list=tf.trainable_variables('global/value'))

        # Train Q-network
        q_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
        with tf.control_dependencies(q_bn_ops):
            optimize_q = q_optimizer.minimize(self.q_loss,
                                              var_list=tf.trainable_variables('main/critic'))

        # Train policy network
        pi_vars = tf.trainable_variables('global/policy')
        pi_vars_flatten = flat_concat(pi_vars)

        with tf.control_dependencies(pi_bn_ops):
            pi_gradient = flat_grad(self.pi_loss, pi_vars)
        x_ph, hvp = hessian_vector_product(self.d_kl, pi_vars)
        hvp += self.cg_damping * x_ph  # hessian vector product

        pi_vars_ph = tf.placeholder(tf.float32, shape=x_ph.shape.as_list())
        set_pi_op = assign_vars_from_flat(pi_vars_ph, pi_vars)

        # Control variance
        cv = tf.reduce_sum(tf.gradients(self.q_e_value, self.mu)[0] * (self.action_ph - self.mu), axis=1)

        return [v_all_vars_flatten, pi_all_vars_flatten, optimize_v, optimize_q,
                init_tar, update_tar, pi_vars_flatten, pi_gradient, hvp, x_ph,
                pi_vars_ph, set_pi_op, cv]

    def _build_policy(self, state):
        # BN is disabled in policy network due to the following reasons:
        #
        # 1. In our implementation of TRPO, the old (sampling) policy's log_p, mu, log_std are recorded when we sample
        # the actions (in testing mode). However, for the optimization policy, training mode is used. This will cause
        # different shifts in the old and new action distributions given the same state, which violates the trust region
        # constraint.
        #
        # 2. One may consider recompute log_p, mu, log_std of the old policy in training mode for policy optimization.
        # Nevertheless, the sampling policy (in testing mode) and optimization policy (in training mode) are still
        # different, which is against the on-policy assumption of TRPO and hence leads to bad performance.
        # The same holds even if we always use training mode, because the batch sizes are different.  (batch size = 1
        # during sampling and batch size = epoch length during training). See more: https://arxiv.org/pdf/1910.09191.pdf

        with tf.variable_scope('policy'):
            mu = mlp(x=state,
                     hidden_sizes=self.actor_size + [self.action_dim],
                     batch_norm=False,  # Disable BN for policy network
                     output_bias_initializer=tf.constant_initializer(self.actor_initial_bias))
            mu = self.action_scale * (mu + 1)

            if self.auto_std:
                log_std = tf.get_variable(name='log_std',
                                          initializer=self.log_std_initial*np.ones(self.action_dim, dtype=np.float32))
            else:
                log_std = self.log_std_ph

            std = tf.exp(log_std)
            action_out = tf.clip_by_value(mu + tf.random_normal(tf.shape(mu)) * std, 1e-8, 4 * self.action_scale)

            mu_rep = tf.repeat(mu, self.e_qf_sample_size, axis=0)
            action_out_samples = tf.clip_by_value(mu_rep + tf.random_normal(tf.shape(mu_rep)) * std, 1e-8,
                                                  4 * self.action_scale)

            # Past action substituted into current policy distribution
            logp_exp = gaussian_log_likelihood(self.action_ph, mu, log_std)
            # Current action substituted into current policy distribution, will be logp_old in the next epoch
            logp = gaussian_log_likelihood(action_out, mu, log_std)

            d_kl = sample_average_kl(self.old_mu_ph, self.old_log_std_ph, mu, log_std)

            entropy = tf.reduce_sum(log_std + .5 * np.log(2.0 * np.pi * np.e))
        return action_out, action_out_samples, mu, logp_exp, logp, log_std, d_kl, entropy

    def _build_value(self):
        with tf.variable_scope('value'):
            value = tf.squeeze(mlp(x=self.state_ph,
                                   hidden_sizes=self.value_size + [1],
                                   output_activation=None,
                                   output_bias_initializer=tf.constant_initializer(self.critic_initial_bias),
                                   batch_norm=self.bn,
                                   is_train=self.train_value_ph),
                               axis=1)
        return value

    def learn(self, sess, log_std_in, buffer_data):
        pi_loss_new, value_loss = 0, 0

        # Unzip the buffered data
        buffer_data = [np.r_[v] for v in zip(*buffer_data)]
        # Clip the return
        buffer_data[3] = np.clip(buffer_data[3], -self.return_clip, 0)

        """
        Compute the advantage with control variance, adv = adv - eta*adv_bar
        """
        adv = buffer_data[2]
        adv_bar = self.get_adv_bar(sess, state=buffer_data[0], action=buffer_data[1],
                                   next_seg_size=buffer_data[-2],
                                   reward_scale=buffer_data[-1],  # Not effective, only used in PDS Q-Prop
                                   )
        if self.qprop_flag == 1:
            eta = (adv * adv_bar) > 0
        elif self.qprop_flag == 2:
            eta = np.sign(adv * adv_bar)
        else:
            eta = np.zeros_like(adv)  # Degenerate into TRPO

        adv -= eta * adv_bar  # Q-Prop learning signal

        # Normalize adv (This actually makes Q-Prop biased
        if self.adv_norm:
            adv = (adv - adv.mean()) / (adv.std())
        buffer_data[2] = adv

        """
        Train policy network
        """
        inputs = {k: np.r_[v] for k, v in zip(self.all_phs, buffer_data)}
        inputs[self.eta_ph] = eta
        inputs_sparse = {k: v[::self.sub_sample] for k, v in inputs.items()}

        inputs_sparse[self.train_value_ph] = False
        inputs_sparse[self.train_critic_ph] = False
        inputs_sparse[self.log_std_ph] = log_std_in
        inputs[self.train_value_ph] = False
        inputs[self.train_critic_ph] = False
        inputs[self.log_std_ph] = log_std_in

        Hx = lambda x: sess.run(self.hvp, feed_dict={**inputs_sparse, self.x_ph: x})
        g, pi_loss_old = sess.run([self.pi_gradient, self.pi_loss], feed_dict=inputs)

        if np.allclose(g, 0):
            pi_loss_new = pi_loss_old
            if self.verbose:
                print('Got zero gradient. Not updating')
        else:
            x = conjugate_gradient_alg(Hx, g)
            alpha = np.sqrt(2 * self.delta / np.dot(x, Hx(x)))
            old_policy_vars = sess.run(self.pi_vars_flatten)

            for j in range(self.backtrack_iters):
                step = self.backtrack_coeff ** j
                sess.run(self.set_pi_op, feed_dict={self.pi_vars_ph: old_policy_vars - alpha * x * step})
                kl, pi_loss_new = sess.run([self.d_kl, self.pi_loss], feed_dict=inputs)

                if kl <= self.delta * 2 and pi_loss_new <= pi_loss_old:
                    if self.verbose:
                        print('Accepting new params at step %d of line search.' % j)
                    break
                if j == self.backtrack_iters - 1:
                    if self.verbose:
                        print('Line search failed! Keeping old params.')
                    sess.run(self.set_pi_op, feed_dict={self.pi_vars_ph: old_policy_vars})
                    kl, pi_loss_new = sess.run([self.d_kl, self.pi_loss], feed_dict=inputs)

        """
        Train value function
        """
        for _ in range(self.train_vf_iters):
            for sampled_state, sampled_return in iter_batches(arrays=(buffer_data[0], buffer_data[3]),
                                                              include_final_partial_batch=False,
                                                              batch_size=self.v_batch_size):
                value_loss, _ = sess.run([self.value_loss, self.optimize_v], feed_dict={self.state_ph: sampled_state,
                                                                                        self.return_ph: sampled_return,
                                                                                        self.train_value_ph: True})

        if self.auto_std:
            log_std = np.mean(sess.run(self.log_std))
        else:
            log_std = np.mean(log_std_in)
        return pi_loss_new, value_loss, log_std

    def learn_q(self, sess, replay_data):
        [state, action, reward, next_state, done] = zip(*replay_data)
        state = np.r_[state]
        action = np.r_[action]
        reward = np.r_[reward]
        next_state = np.r_[next_state]
        done = np.r_[done]

        q_loss, _ = sess.run([self.q_loss, self.optimize_q],
                             feed_dict={self.state_ph: state,
                                        self.action_ph: action,
                                        self.reward_ph: reward,
                                        self.next_state_ph: next_state,
                                        self.done_ph: done,
                                        self.train_critic_ph: True})
        sess.run(self.update_tar)
        return q_loss

    def get_vars(self, sess):
        # Get global variables
        v_all_vars_flatten = sess.run(self.v_all_vars_flatten)
        pi_all_vars_flatten = sess.run(self.pi_all_vars_flatten)
        return v_all_vars_flatten, pi_all_vars_flatten

    def get_adv_bar(self, sess, state, action, *args, **kwargs):
        adv_bar = sess.run(self.cv, feed_dict={self.state_ph: state,
                                               self.action_ph: action,
                                               self.train_critic_ph: False,
                                               self.train_value_ph: False})
        return adv_bar

    def get_action(self, sess, state, noise_scale=0, *args):
        action = sess.run(self.action_out, feed_dict={self.state_ph: state,
                                                      self.train_value_ph: False,
                                                      self.log_std_ph: [-10]})
        return action
