import numpy as np


class ExploreScheduler(object):
    """
    Exploration noise scale scheduler
    """
    def __init__(self, stage, scale_bound):
        self.scale_bound = scale_bound
        self.scale = scale_bound[0]
        self.stage = stage
        self.decay = (scale_bound[0] - scale_bound[1])/stage[1]
        self.test_scale = scale_bound[-1]
        self.cnt = 0

    def update_scale(self):
        # Update the noise scale
        if self.stage[0] < self.cnt <= self.stage[0] + self.stage[1]:
            self.scale -= self.decay
        if self.cnt > np.sum(self.stage[:-1]):
            self.scale = self.test_scale
        self.cnt += 1
        return self.scale

    def reset(self):
        self.scale = self.scale_bound[0]

