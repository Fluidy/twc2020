import tensorflow as tf
import numpy as np
from algorithms.ddpg.agent import DDPGAgent
from algorithms.ddpg_pds.replay_buffer import PDSReplayBuffer
from algorithms.common.layers import mlp, pds_func, reward_func, safety_layer


class PDSDDPGAgent(DDPGAgent):
    def __init__(self, env, params, build_network=True):
        super().__init__(env, params, build_network=False)
        self.env = env  # This contains all the information for PDS learning and safety layer

        # Override the replay buffer
        self.buffer = PDSReplayBuffer(self.state_dim, self.action_dim, self.mem_size, self.batch_size)

        # Placeholders for PDS, safety layer and n-step update
        self.noise_ph = tf.placeholder(tf.float32, [None, self.action_dim])
        self.next_seg_size_ph = tf.placeholder(tf.float32, [None, 1])
        self.next2_seg_size_ph = tf.placeholder(tf.float32, [None, 1])
        self.reward_scale_ph = tf.placeholder(tf.float32, [None])

        if build_network:
            # Build actor and critic networks
            with tf.variable_scope('main'):
                self.action_out = self._build_actor(state=self.state_ph,
                                                    train_actor=self.train_actor_ph,
                                                    next_seg_size=self.next_seg_size_ph)
                self.q_value, self.policy_value = self._build_critic(state=self.state_ph,
                                                                     next_seg_size=self.next_seg_size_ph,
                                                                     policy_action=self.action_out,
                                                                     action_in=self.action_ph,
                                                                     train_critic=self.train_critic_ph)
            with tf.variable_scope('target'):
                action_tar = self._build_actor(state=self.next_state_ph,
                                               next_seg_size=self.next2_seg_size_ph)
                _, self.policy_value_tar = self._build_critic(state=self.next_state_ph,
                                                              next_seg_size=self.next2_seg_size_ph,
                                                              policy_action=action_tar,
                                                              action_in=self.action_ph)

            # Losses and training operations are the same as DDPG
            self.actor_loss, self.critic_loss = self._get_losses()
            self.optimize_actor, self.optimize_critic, self.init_tar, self.update_tar = super()._get_training_ops()

    def _build_actor(self, state, train_actor=False, next_seg_size=None):
        proto_action = super()._build_actor(state, train_actor=train_actor)
        noised_action = tf.nn.relu(proto_action + self.noise_ph)
        action = safety_layer(env=self.env,
                              state=state,
                              next_seg_size=next_seg_size,
                              action=noised_action)

        return action

    def _build_critic(self, state, policy_action, action_in, train_critic=False, next_seg_size=None):
        with tf.variable_scope('critic'):  # Q(s, a)
            pds_state, snr, not_done_pds = pds_func(env=self.env,
                                                    state=state,
                                                    next_seg_size=next_seg_size,
                                                    action=action_in)
            pds_value = tf.squeeze(mlp(x=pds_state,
                                       hidden_sizes=self.critic_size + [1],
                                       output_activation=None,
                                       output_bias_initializer=tf.constant_initializer(self.critic_initial_bias),
                                       batch_norm=self.bn,
                                       is_train=train_critic),
                                   axis=1)
            q_value = pds_value * not_done_pds + reward_func(env=self.env,
                                                             snr=snr,
                                                             rate=action_in,
                                                             reward_scale=self.reward_scale_ph)

        with tf.variable_scope('critic', reuse=True):  # Q(s, mu(s))
            pds_state_policy, snr_policy, not_done_pds_policy = pds_func(self.env, state, next_seg_size, policy_action)
            policy_pds_value = tf.squeeze(mlp(x=pds_state_policy,  # The only difference is here
                                              hidden_sizes=self.critic_size + [1],
                                              output_activation=None,
                                              output_bias_initializer=tf.constant_initializer(self.critic_initial_bias),
                                              batch_norm=self.bn,
                                              is_train=train_critic),
                                          axis=1)
            policy_value = policy_pds_value * not_done_pds_policy + reward_func(env=self.env,
                                                                                snr=snr_policy,
                                                                                rate=policy_action,
                                                                                reward_scale=self.reward_scale_ph)
        return q_value, policy_value

    def get_action(self, sess, state, noise_scale, next_seg_size=None):
        noise = noise_scale * np.random.normal(size=self.action_dim)
        action = sess.run(self.action_out, feed_dict={self.state_ph: state,
                                                      self.train_actor_ph: False,
                                                      self.noise_ph: [noise],
                                                      self.next_seg_size_ph: next_seg_size})[0]
        return action

    def learn_batch(self, sess, train_actor=True):
        state, action, reward, next_state, done, next_seg_size, next2_seg_size, reward_scale = self.buffer.sample()
        # train critic
        critic_loss, _ = sess.run([self.critic_loss, self.optimize_critic],
                                  feed_dict={self.state_ph: state,
                                             self.action_ph: action,
                                             self.reward_ph: reward,
                                             self.next_state_ph: next_state,
                                             self.done_ph: done,
                                             self.train_actor_ph: False,
                                             self.train_critic_ph: True,
                                             self.noise_ph: np.zeros([len(state), 1]),
                                             self.next_seg_size_ph: next_seg_size,
                                             self.next2_seg_size_ph: next2_seg_size,
                                             self.reward_scale_ph: reward_scale})
        output = [critic_loss]

        if train_actor:
            # train actor
            actor_loss, _, _ = sess.run([self.actor_loss, self.optimize_actor, self.update_tar],
                                        feed_dict={self.state_ph: state,
                                                   self.train_actor_ph: True,
                                                   self.train_critic_ph: False,
                                                   self.noise_ph: np.zeros([len(state), 1]),
                                                   self.next_seg_size_ph: next_seg_size,
                                                   self.reward_scale_ph: reward_scale})
            output.append(actor_loss)
        return output



