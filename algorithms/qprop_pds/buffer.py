import numpy as np
from algorithms.qprop.buffer import GAEBuffer


class PDSGAEBuffer(GAEBuffer):
    def __init__(self, state_dim, action_dim, size, gamma, lam, td_lambda):
        super().__init__(state_dim, action_dim, size, gamma, lam, td_lambda)
        self.next_seg_size_buf = np.zeros([size, 1], dtype=np.float32)
        self.reward_scale_buf = np.zeros(size, dtype=np.float32)

    def add(self, state, action, reward, value, logp, mu, log_std, next_seg_size=None, reward_scale=None):
        self.next_seg_size_buf[self.pointer] = next_seg_size
        self.reward_scale_buf[self.pointer] = reward_scale
        super().add(state, action, reward, value, logp, mu, log_std)

    def get(self):
        buffer_data = super().get()
        return buffer_data + [self.next_seg_size_buf, self.reward_scale_buf]

