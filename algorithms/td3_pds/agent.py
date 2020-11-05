import tensorflow as tf
from algorithms.common.layers import mlp, pds_func, reward_func, safety_layer
from algorithms.td3.agent import TD3Agent
from algorithms.ddpg_pds.agent import PDSDDPGAgent


class PDSTD3Agent(TD3Agent, PDSDDPGAgent):
    def __init__(self, env, params):
        super().__init__(env, params, build_network=False)
        self.env = env  # This contains all the information need for PDS learning and safety layer

        # Build the computation graph
        with tf.variable_scope('main'):
            self.action_out = self._build_actor(state=self.state_ph,
                                                next_seg_size=self.next_seg_size_ph,
                                                train_actor=self.train_actor_ph)

            self.q1, self.q2, self.policy_value = self._build_critic(state=self.state_ph,
                                                                     next_seg_size=self.next_seg_size_ph,
                                                                     policy_action=self.action_out,
                                                                     action_in=self.action_ph,
                                                                     train_critic=self.train_critic_ph)

        with tf.variable_scope('target'):
            action_tar = self._build_actor(state=self.next_state_ph, next_seg_size=self.next2_seg_size_ph)

            # Target policy smoothing, by adding clipped noise to target actions
            epsilon = tf.random_normal(tf.shape(action_tar), stddev=self.target_noise)
            epsilon = tf.clip_by_value(epsilon, -self.noise_clip, self.noise_clip)
            # Ensure the target action is safe and valid
            action_tar = safety_layer(env=self.env,
                                      state=self.next_state_ph,
                                      next_seg_size=self.next2_seg_size_ph,
                                      action=tf.nn.relu(action_tar + epsilon))

            # Target Q-values, using noised action from the target policy
            self.q1_tar, self.q2_tar, _ = self._build_critic(state=self.next_state_ph,
                                                             next_seg_size=self.next2_seg_size_ph,
                                                             policy_action=action_tar,  # not used
                                                             action_in=action_tar)

        # Get TD3 losses
        self.actor_loss, self.critic_loss = self._get_losses()

        # Get all the operations for training
        self.optimize_actor, self.optimize_critic, self.init_tar, self.update_tar = self._get_training_ops()

    def _build_critic(self, state, policy_action, action_in, train_critic=False, next_seg_size=None):
        with tf.variable_scope('critic_1'):  # Q1(s, a)
            pds_state, snr, not_done_pds = pds_func(self.env, state, next_seg_size, action_in)
            pds_value = tf.squeeze(mlp(x=pds_state,
                                       hidden_sizes=self.critic_size + [1],
                                       output_activation=None,
                                       output_bias_initializer=tf.constant_initializer(self.critic_initial_bias),
                                       batch_norm=self.bn,
                                       is_train=train_critic),
                                   axis=1)
            q_value_1 = pds_value * not_done_pds + reward_func(env=self.env,
                                                               snr=snr,
                                                               rate=action_in,
                                                               reward_scale=self.reward_scale_ph)

        with tf.variable_scope('critic_2'):  # Q2(s, a)
            pds_value = tf.squeeze(mlp(x=pds_state,
                                       hidden_sizes=self.critic_size + [1],
                                       output_activation=None,
                                       output_bias_initializer=tf.constant_initializer(self.critic_initial_bias),
                                       batch_norm=self.bn,
                                       is_train=train_critic),
                                   axis=1)
            q_value_2 = pds_value * not_done_pds + reward_func(env=self.env,
                                                               snr=snr,
                                                               rate=action_in,
                                                               reward_scale=self.reward_scale_ph)

        with tf.variable_scope('critic_1', reuse=True):  # Q1(s, mu(s))
            pds_state_policy, snr_policy, not_done_pds_policy = pds_func(self.env, state, next_seg_size, policy_action)
            policy_pds_value = tf.squeeze(mlp(x=pds_state_policy,
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
        return q_value_1, q_value_2, policy_value

