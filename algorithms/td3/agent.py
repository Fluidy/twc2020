import tensorflow as tf
from algorithms.common.layers import mlp
from algorithms.ddpg.agent import DDPGAgent


class TD3Agent(DDPGAgent):
    def __init__(self, env, params, build_network=True):
        super().__init__(env, params, build_network=False)

        # Copy the parameters for TD3
        self.policy_delay = params['policy_delay']
        self.target_noise = params['target_noise']
        self.noise_clip = params['noise_clip']

        # Build the computation graph
        if build_network:
            with tf.variable_scope('main'):
                self.action_out = self._build_actor(state=self.state_ph,
                                                    train_actor=self.train_actor_ph)

                self.q1, self.q2, self.policy_value = self._build_critic(state=self.state_ph,
                                                                         policy_action=self.action_out,
                                                                         action_in=self.action_ph,
                                                                         train_critic=self.train_critic_ph)

            with tf.variable_scope('target'):
                action_tar = self._build_actor(state=self.next_state_ph)

                # Target policy smoothing, by adding clipped noise to target actions
                epsilon = tf.random_normal(tf.shape(action_tar), stddev=self.target_noise)
                epsilon = tf.clip_by_value(epsilon, -self.noise_clip, self.noise_clip)
                # Ensure the target action is still bounded to [0, 2*self.action_scale]
                action_tar = tf.clip_by_value(action_tar + epsilon, 0, 2*self.action_scale)

                # Target Q-values, using noised action from the target policy
                self.q1_tar, self.q2_tar, _ = self._build_critic(state=self.next_state_ph,
                                                                 policy_action=action_tar,  # not used
                                                                 action_in=action_tar)

            # Get TD3 losses
            self.actor_loss, self.critic_loss = self._get_losses()

            # Get all the operations for training
            self.optimize_actor, self.optimize_critic, self.init_tar, self.update_tar = self._get_training_ops()

    def _get_losses(self):
        # Q-learning backup, using the clipped double-Q target
        min_q_tar = tf.minimum(self.q1_tar, self.q2_tar)
        y = tf.stop_gradient(self.reward_ph + self.gamma * (1 - self.done_ph) * min_q_tar)

        # TD3 losses
        actor_loss = -tf.reduce_mean(self.policy_value)
        q1_loss = tf.reduce_mean(tf.square(self.q1 - y))
        q2_loss = tf.reduce_mean(tf.square(self.q2 - y))
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables('main/critic')
                            if 'kernel' in v.name])
        critic_loss = q1_loss + q2_loss + self.l2_coeff * l2_loss
        return actor_loss, critic_loss

    def _build_critic(self, state, policy_action, action_in, train_critic=False):
        with tf.variable_scope('critic_1'):  # Q1(s, a)
            q_value_1 = tf.squeeze(mlp(x=tf.concat([state, action_in], axis=1),
                                       hidden_sizes=self.critic_size + [1],
                                       output_activation=None,
                                       output_bias_initializer=tf.constant_initializer(self.critic_initial_bias),
                                       batch_norm=self.bn,
                                       is_train=train_critic),
                                   axis=1)

        with tf.variable_scope('critic_2'):  # Q2(s, a), the same as critic_1, just with different initial parameters
            q_value_2 = tf.squeeze(mlp(x=tf.concat([state, action_in], axis=1),
                                       hidden_sizes=self.critic_size + [1],
                                       output_activation=None,
                                       output_bias_initializer=tf.constant_initializer(self.critic_initial_bias),
                                       batch_norm=self.bn,
                                       is_train=train_critic),
                                   axis=1)

        with tf.variable_scope('critic_1', reuse=True):  # Q1(s, mu(s))
            policy_value = tf.squeeze(mlp(x=tf.concat([state, policy_action], axis=1),
                                          hidden_sizes=self.critic_size + [1],
                                          output_activation=None,
                                          batch_norm=self.bn,
                                          is_train=train_critic),
                                      axis=1)

        return q_value_1, q_value_2, policy_value


