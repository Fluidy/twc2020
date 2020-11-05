import numpy as np
import scipy.signal


class GAEBuffer:
    def __init__(self, state_dim, action_dim, size, gamma, lam, td_lambda):
        self.state_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.action_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.reward_buf = np.zeros(size, dtype=np.float32)
        self.return_buf = np.zeros(size, dtype=np.float32)
        self.value_buf = np.zeros(size, dtype=np.float32)
        self.log_p_buf = np.zeros(size, dtype=np.float32)
        self.mu_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.log_std_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.path_start_idx, self.size = 0, 0, size
        self.td_lambda = td_lambda

    def add(self, state, action, reward, value, logp, mu, log_std, *args):
        assert self.pointer < self.size  # buffer has to have room
        self.state_buf[self.pointer] = state
        self.action_buf[self.pointer] = action
        self.reward_buf[self.pointer] = reward
        self.value_buf[self.pointer] = value
        self.log_p_buf[self.pointer] = logp
        self.mu_buf[self.pointer] = mu
        self.log_std_buf[self.pointer] = log_std
        self.pointer += 1

    def finish_path(self, last_value=0):
        """
        Compute GAE-lambda advantage
        Note: advantage is only computed when a trajectory ends (using monte carlo estimation) or gets cut of by
        by when the buffer is full (using TD backup).

        :param last_value:
        :return:
        """
        path_slice = slice(self.path_start_idx, self.pointer)
        rewards = np.append(self.reward_buf[path_slice], last_value)
        values = np.append(self.value_buf[path_slice], last_value)

        # Compute the advantage for each state in the just finished path using GAE-lambda
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.adv_buf[path_slice] = discount_cum_sum(deltas, self.gamma*self.lam)

        if self.td_lambda:
            self.return_buf[path_slice] = self.adv_buf[path_slice] + self.value_buf[path_slice]  # TD(lambda)
        else:
            self.return_buf[path_slice] = discount_cum_sum(rewards, self.gamma)[:-1]  # reward-to-go TD(1)

        self.path_start_idx = self.pointer

    def get(self):
        assert self.pointer == self.size  # buffer has to be full
        self.pointer, self.path_start_idx = 0, 0
        return [self.state_buf, self.action_buf, self.adv_buf, self.return_buf,
                self.log_p_buf, self.mu_buf, self.log_std_buf]


def discount_cum_sum(x, discount):
    """
    input: vector x = [x0, x1, x2]
    output: [x0 + discount * x1 + discount^2 * x2,
             x1 + discount * x2,
             x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

