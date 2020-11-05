class NonPredictiveAgent(object):
    """
    Non-Predictive Transmission
    """
    def __init__(self, env, _):
        self.env = env

    def get_action(self, *args, **kwargs):
        action = self.env.seg_size[self.env.seg_cnt + 1]/self.env.seg_TF + 1e-8

        return [action]
