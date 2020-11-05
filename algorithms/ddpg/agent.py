import tensorflow as tf
import numpy as np
from algorithms.ddpg.replay_buffer import ReplayBuffer
from algorithms.common.layers import mlp


class DDPGAgent(object):
    """
    The basic DDPG agent
    """
    def __init__(self, env, params, build_network=True, *args):
        # Copy params
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.critic_size = params['critic_size']
        self.actor_size = params['actor_size']
        self.critic_lr = params['critic_lr']
        self.actor_lr = params['actor_lr']
        self.critic_initial_bias = params['critic_initial_bias']
        self.actor_initial_bias = params['actor_initial_bias']
        self.action_scale = params['action_scale']

        self.bn = params['bn']
        self.batch_size = params['batch_size']
        self.mem_size = int(params['mem_size'])
        self.gamma = params['gamma']
        self.l2_coeff = params['l2_coeff']
        self.tau = params['tau']

        # Initialize replay buffer
        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, self.mem_size, self.batch_size)

        # Placeholders
        self.state_ph = tf.placeholder(tf.float32, [None, self.state_dim])
        self.action_ph = tf.placeholder(tf.float32, [None, self.action_dim])
        self.reward_ph = tf.placeholder(tf.float32, [None])
        self.next_state_ph = tf.placeholder(tf.float32, [None, self.state_dim])
        self.done_ph = tf.placeholder(tf.float32, [None])
        # Placeholders for batch normalization
        self.train_actor_ph = tf.placeholder(tf.bool, shape=())
        self.train_critic_ph = tf.placeholder(tf.bool, shape=())

        if build_network:
            # Build actor and critic network
            with tf.variable_scope('main'):
                self.action_out = self._build_actor(state=self.state_ph,
                                                    train_actor=self.train_actor_ph)
                self.q_value, self.policy_value = self._build_critic(state=self.state_ph,
                                                                     policy_action=self.action_out,
                                                                     action_in=self.action_ph,
                                                                     train_critic=self.train_critic_ph)
            with tf.variable_scope('target'):
                action_tar = self._build_actor(state=self.next_state_ph)
                _, self.policy_value_tar = self._build_critic(state=self.next_state_ph,
                                                              policy_action=action_tar,
                                                              action_in=self.action_ph)  # Actually, action_in is not used here

            # Get DDPG losses
            self.actor_loss, self.critic_loss = self._get_losses()

            # Get all the operations for training
            self.optimize_actor, self.optimize_critic, self.init_tar, self.update_tar = self._get_training_ops()

    def _build_actor(self, state, train_actor=False):
        with tf.variable_scope('actor'):
            action = mlp(x=state,
                         hidden_sizes=self.actor_size + [self.action_dim],
                         output_bias_initializer=tf.constant_initializer(self.actor_initial_bias),
                         batch_norm=self.bn,
                         is_train=train_actor)
        # Action shaping
        action = (action + 1)*self.action_scale
        return action

    def _build_critic(self, state, policy_action, action_in, train_critic=False):
        with tf.variable_scope('critic'):  # Q(s, a)
            q_value = tf.squeeze(mlp(x=tf.concat([state, action_in], axis=-1),
                                     hidden_sizes=self.critic_size + [1],
                                     output_activation=None,
                                     output_bias_initializer=tf.constant_initializer(self.critic_initial_bias),
                                     batch_norm=self.bn,
                                     is_train=train_critic),
                                 axis=1)

        with tf.variable_scope('critic', reuse=True):  # Q(s, mu(s))
            policy_value = tf.squeeze(mlp(x=tf.concat([state, policy_action], axis=-1),
                                          hidden_sizes=self.critic_size + [1],
                                          output_activation=None,
                                          output_bias_initializer=tf.constant_initializer(self.critic_initial_bias),
                                          batch_norm=self.bn,
                                          is_train=train_critic),
                                      axis=1)
        return q_value, policy_value

    def _get_losses(self):
        # Q-learning backup
        y = tf.stop_gradient(self.reward_ph + self.gamma * (1 - self.done_ph) * self.policy_value_tar)

        # DDPG losses
        actor_loss = - tf.reduce_mean(self.policy_value)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables('main/critic')
                            if 'kernel' in v.name])
        critic_loss = tf.reduce_mean(tf.square(y - self.q_value)) + self.l2_coeff * l2_loss
        return actor_loss, critic_loss

    def _get_training_ops(self):
        # Target networks operations
        init_tar = [v_tar.assign(v_main) for v_main, v_tar
                    in zip(tf.global_variables('main'), tf.global_variables('target'))]  # Should use global vars to include the moving average and std
        update_tar = [v_tar.assign(self.tau * v_main + (1 - self.tau) * v_tar) for v_main, v_tar
                      in zip(tf.global_variables('main'), tf.global_variables('target'))]

        # Operations for moving average and std updated in batch normalization
        critic_bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='main/critic')
        actor_bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='main/actor')

        # Training operations
        actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
        critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)

        with tf.control_dependencies(actor_bn_ops):
            optimize_actor = actor_optimizer.minimize(self.actor_loss,
                                                      var_list=tf.trainable_variables('main/actor'))

        with tf.control_dependencies(critic_bn_ops):
            optimize_critic = critic_optimizer.minimize(self.critic_loss,
                                                        var_list=tf.trainable_variables('main/critic'))

        return optimize_actor, optimize_critic, init_tar, update_tar

    def get_action(self, sess, state, noise_scale, *args):
        action = sess.run(self.action_out, feed_dict={self.state_ph: state,
                                                      self.train_actor_ph: False})
        action = np.maximum(action + noise_scale * np.random.normal(size=self.action_dim), 0)
        return action

    def learn_batch(self, sess, train_actor=True):
        state, action, reward, next_state, done = self.buffer.sample()

        # Train critic
        critic_loss, _ = sess.run([self.critic_loss, self.optimize_critic],
                                  feed_dict={self.state_ph: state,
                                             self.action_ph: action,
                                             self.reward_ph: reward,
                                             self.next_state_ph: next_state,
                                             self.done_ph: done,
                                             self.train_actor_ph: False,
                                             self.train_critic_ph: True})

        output = [critic_loss]
        if train_actor:
            # Train actor
            actor_loss, _, _ = sess.run([self.actor_loss, self.optimize_actor, self.update_tar],
                                        feed_dict={self.state_ph: state,
                                                   self.train_actor_ph: True,
                                                   self.train_critic_ph: False})

            output.append(actor_loss)
        return output

    def initialize(self, sess):
        sess.run(tf.global_variables_initializer())
        sess.run(self.init_tar)







