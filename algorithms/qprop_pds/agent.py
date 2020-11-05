from algorithms.common.layers import mlp, safety_layer
from algorithms.qprop.tf_utils import *
from algorithms.qprop.agent import QPropAgent
from algorithms.ddpg_pds.agent import PDSDDPGAgent


class PDSQPropAgent(QPropAgent, PDSDDPGAgent):
    def __init__(self, env, params, build_network=True, *args):
        super().__init__(env, params, False, *args)
        self.all_phs = [self.state_ph, self.action_ph, self.advantage_ph, self.return_ph,
                        self.logp_old_ph, self.old_mu_ph, self.old_log_std_ph,
                        self.next_seg_size_ph, self.reward_scale_ph]

        if build_network:
            with tf.variable_scope('global'):
                [self.action_out, _, self.mu, self.logp_exp, self.logp, self.log_std,
                 self.d_kl, self.entropy] = self._build_policy(self.state_ph, self.next_seg_size_ph)
                self.value = self._build_value()
            with tf.variable_scope('global', reuse=True):
                _, action_tar_samples, mu_tar, _, _, _, _, _ = self._build_policy(self.next_state_ph,
                                                                                  self.next2_seg_size_ph)

            with tf.variable_scope('main'):
                self.q_value, self.q_e_value = self._build_critic(state=self.state_ph,
                                                                  policy_action=self.mu,
                                                                  action_in=self.action_ph,
                                                                  train_critic=self.train_critic_ph,
                                                                  next_seg_size=self.next_seg_size_ph)
            with tf.variable_scope('target'):
                _, self.q_e_tar = self._build_critic(state=self.next_state_ph,
                                                     policy_action=mu_tar,
                                                     action_in=mu_tar,
                                                     next_seg_size=self.next2_seg_size_ph)
            with tf.variable_scope('target', reuse=True):
                self.q_tar, _ = self._build_critic(state=tf.repeat(self.next_state_ph, self.e_qf_sample_size, axis=0),
                                                   policy_action=action_tar_samples,
                                                   action_in=action_tar_samples,
                                                   next_seg_size=tf.repeat(self.next2_seg_size_ph, self.e_qf_sample_size,
                                                                           axis=0))

            # Get losses
            self.pi_loss, self.value_loss, self.q_loss = self._get_losses()

            # Get training operations
            [self.v_all_vars_flatten, self.pi_all_vars_flatten,
             self.optimize_v, self.optimize_q, self.init_tar, self.update_tar,
             self.pi_vars_flatten, self.pi_gradient, self.hvp, self.x_ph,
             self.pi_vars_ph, self.set_pi_op, self.cv] = self._get_training_ops()

    def _build_policy(self, state, next_seg_size=None):
        with tf.variable_scope('policy'):
            mu = mlp(x=state,
                     hidden_sizes=self.actor_size + [self.action_dim],
                     batch_norm=False,  # Disable BN for policy network
                     output_bias_initializer=tf.constant_initializer(self.actor_initial_bias))
            mu = self.action_scale * (mu + 1)

            mu = safety_layer(env=self.env, state=state, next_seg_size=next_seg_size, action=mu)

            if self.auto_std:
                log_std = tf.get_variable(name='log_std',
                                          initializer=self.log_std_initial * np.ones(self.action_dim, dtype=np.float32))
            else:
                log_std = self.log_std_ph
            std = tf.exp(log_std)
            action_out = tf.clip_by_value(mu + tf.random_normal(tf.shape(mu)) * std, 1e-8, 4 * self.action_scale)
            action_out = safety_layer(env=self.env, state=state, next_seg_size=next_seg_size, action=action_out)

            mu_rep = tf.repeat(mu, self.e_qf_sample_size, axis=0)
            action_out_samples = tf.clip_by_value(mu_rep + tf.random_normal(tf.shape(mu_rep)) * std, 1e-8,
                                                  4 * self.action_scale)
            action_out_samples = safety_layer(env=self.env,
                                              state=tf.repeat(state, self.e_qf_sample_size, axis=0),
                                              next_seg_size=tf.repeat(next_seg_size, self.e_qf_sample_size, axis=0),
                                              action=action_out_samples)

            # Past action substituted into current policy distribution
            logp_exp = gaussian_log_likelihood(self.action_ph, mu, log_std)
            # Current action substituted into current policy distribution, will be logp_old in the next epoch
            logp = gaussian_log_likelihood(action_out, mu, log_std)

            d_kl = sample_average_kl(self.old_mu_ph, self.old_log_std_ph, mu, log_std)

            entropy = tf.reduce_sum(log_std + .5 * np.log(2.0 * np.pi * np.e))

        return action_out, action_out_samples, mu, logp_exp, logp, log_std, d_kl, entropy

    def learn_q(self, sess, replay_data):
        [state, action, reward, next_state, done,
         next_seg_size, next2_seg_size, reward_scale] = zip(*replay_data)
        state = np.r_[state]
        action = np.r_[action]
        reward = np.r_[reward]
        next_state = np.r_[next_state]
        done = np.r_[done]
        next_seg_size = np.r_[next_seg_size]
        next2_seg_size = np.r_[next2_seg_size]
        reward_scale = np.r_[reward_scale]

        q_loss, _ = sess.run([self.q_loss, self.optimize_q],
                             feed_dict={self.state_ph: state,
                                        self.action_ph: action,
                                        self.reward_ph: reward,
                                        self.next_state_ph: next_state,
                                        self.done_ph: done,
                                        self.train_critic_ph: True,
                                        self.next_seg_size_ph: next_seg_size,
                                        self.next2_seg_size_ph: next2_seg_size,
                                        self.reward_scale_ph: reward_scale})
        sess.run(self.update_tar)
        return q_loss

    def get_adv_bar(self, sess, state, action, next_seg_size=None, reward_scale=None):
        adv_bar = sess.run(self.cv, feed_dict={self.state_ph: state,
                                               self.action_ph: action,
                                               self.train_critic_ph: False,
                                               self.train_value_ph: False,
                                               self.next_seg_size_ph: next_seg_size,
                                               self.reward_scale_ph: reward_scale})
        return adv_bar

    def get_action(self, sess, state, noise_scale=0, next_seg_size=None):
        action = sess.run(self.action_out, feed_dict={self.state_ph: state,
                                                      self.next_seg_size_ph: next_seg_size,
                                                      self.train_value_ph: False,
                                                      self.log_std_ph: [-10]})
        return action

