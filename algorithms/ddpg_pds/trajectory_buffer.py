import numpy as np


class TrajectoryBuffer(object):
    """
    Trajectory Buffer that stores historical user trajectories
    """
    def __init__(self):
        self.trajectories = []
        self.road_choices = []
        self.seg_sizes = []
        self.cur_trajectory = []

    def add_pos(self, pos):
        self.cur_trajectory.append(pos)

    def finish(self, road_choice, seg_sizes):
        self.trajectories.append(self.cur_trajectory)
        self.road_choices.append(road_choice)
        self.seg_sizes.append(seg_sizes)
        self.cur_trajectory = []

    def sample(self):
        idx_1, idx_2 = np.random.choice(len(self.trajectories), 2)
        return self.trajectories[idx_1], self.road_choices[idx_1], self.seg_sizes[idx_2]

