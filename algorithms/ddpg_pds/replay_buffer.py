import numpy as np
from algorithms.ddpg.replay_buffer import ReplayBuffer


class PDSReplayBuffer(ReplayBuffer):
    """
    Replay buffer for PDS-DDPG, just add some extra information to the basic replay buffer
    """
    def __init__(self, state_dim, action_dim, length, batch_size):
        super().__init__(state_dim, action_dim, length, batch_size)

        # Add the information needed for PDS-DDPG
        self.next_seg_size_buf = np.zeros([length, 1], dtype=np.float32)  # The size of segment to be played in the next TF
        self.next2_seg_size_buf = np.zeros([length, 1], dtype=np.float32)  # The size of segment to be played in the next*2 TF
        self.reward_scale_buf = np.zeros(length, dtype=np.float32)
        self.exp_len_buf = np.zeros(length, dtype=np.int16)

    def add(self, state, action, reward, next_state, done, next_seg_size=None, next2_seg_size=None, reward_scale=None):
        self.next_seg_size_buf[self.pointer] = next_seg_size
        self.next2_seg_size_buf[self.pointer] = next2_seg_size
        self.reward_scale_buf[self.pointer] = reward_scale
        super().add(state, action, reward, next_state, done)

    def sample(self):
        batch = super().sample()
        return batch + (self.next_seg_size_buf[self.sampled_indices],
                        self.next2_seg_size_buf[self.sampled_indices],
                        self.reward_scale_buf[self.sampled_indices])



