import numpy as np


class ReplayBuffer(object):
    """
    A basic replay buffer :)
    """
    def __init__(self, state_dim, action_dim, length, batch_size):
        self.state_buf = np.zeros([length, state_dim], dtype=np.float32)
        self.action_buf = np.zeros([length, action_dim], dtype=np.float32)
        self.reward_buf = np.zeros(length, dtype=np.float32)
        self.next_state_buf = np.zeros([length, state_dim], dtype=np.float32)
        self.done_buf = np.zeros(length, dtype=np.float32)

        self.length, self.batch_size = length, batch_size
        self.pointer, self.full = 0, False
        self.sampled_indices = None

    def add(self, state, action, reward, next_state, done, *args, **kwargs):
        # Add a new experience into the replay buffer
        self.state_buf[self.pointer] = state
        self.action_buf[self.pointer] = action
        self.reward_buf[self.pointer] = reward
        self.next_state_buf[self.pointer] = next_state
        self.done_buf[self.pointer] = done

        self.pointer += 1
        if self.pointer == self.length:
            self.pointer = 0
            self.full = True

    def sample(self):
        # Sample a mini-batch from the replay buffer
        assert self.pointer > 0 or self.full
        if self.full:
            idx = np.random.choice(self.length, self.batch_size, replace=False)
        else:
            idx = np.random.choice(self.pointer, self.batch_size, replace=False)
        self.sampled_indices = idx
        return self.state_buf[idx], self.action_buf[idx], self.reward_buf[idx], self.next_state_buf[idx], self.done_buf[idx]

