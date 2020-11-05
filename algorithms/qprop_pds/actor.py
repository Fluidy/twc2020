from algorithms.qprop_pds.agent import PDSQPropAgent
from algorithms.qprop_pds.buffer import PDSGAEBuffer
from algorithms.qprop.actor import QPropActor


class PDSQPropActor(PDSQPropAgent, QPropActor):
    def __init__(self, env, agent_params, _, env_id, epoch_len, max_ep_len, use_ray, use_cpu):
        super().__init__(env, agent_params, False, env_id, epoch_len, max_ep_len, use_ray, use_cpu)

        self.gae_buffer = PDSGAEBuffer(self.state_dim, self.action_dim, self.epoch_len,
                                       self.gamma, self.lam, self.td_lambda_flag)

    def get_actor_outs(self, state, log_std_in, next_seg_size):
        return self.sess.run(self.tensor_to_eval,
                             feed_dict={self.state_ph: state.reshape(1, -1),
                                        self.next_seg_size_ph: [next_seg_size],
                                        self.train_value_ph: False,
                                        self.log_std_ph: log_std_in})

    def _build_policy(self, state, next_seg_size=None):
        return super()._build_policy(state, next_seg_size=self.next_seg_size_ph)
