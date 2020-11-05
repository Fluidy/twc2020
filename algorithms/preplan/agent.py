from scipy.io import loadmat


class PrePlanAgent(object):
    """
    Transmit based on pre-determined rate
    """
    def __init__(self, env, agent_params):
        self.planned_action = loadmat(agent_params['action_dir'])['R']
        self.episode = None
        self.t = 0
        self.max_t = env.seg_TF*(len(env.seg_size) - 1)
        self.done = False

    def get_action(self, *args, **kwargs):
        if self.t == self.max_t or self.done:
            self.t = 0
            self.done = False

        action = self.planned_action[self.episode][self.t] + 1e-4
        self.t += 1

        return [action]
