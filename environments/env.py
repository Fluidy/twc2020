import numpy as np
from scipy.io import loadmat
from collections import deque
from scipy.io import savemat
import os


class Env(object):
    """
    The wireless video streaming environment
    """

    def __init__(self, env_params):
        # Copying the env params
        self.road_distances = env_params['road_distances']
        self.acc_std = env_params['acc_std']
        self.speed_min, self.speed_max = env_params['speed_bound']
        self.cell_radius = env_params['cell_radius']
        self.Nt = env_params['Nt']
        self.delta_T = env_params['delta_T']  # Duration of each TF (s)

        self.noise = env_params['noise'] - 30  # dBm to dB
        self.bandwidth = env_params['bandwidth']
        self.power = env_params['power']
        self.max_rate = env_params['max_rate']

        self.num_seg = env_params['num_seg']
        self.seg_TF = env_params['seg_TF']  # Number of TF for each segment
        self.vbr_mean = env_params['vbr_mean']
        self.vbr_std = env_params['vbr_std']
        self.vbr_max = env_params['vbr_max']
        self.playback_start = env_params['playback_start']
        self.video_length = self.seg_TF * self.num_seg

        self.stop_pos = env_params['stop_pos']
        self.stop_time_range = env_params['stop_time_range']

        # Default status
        self.speed = None
        self.speed_before_stop = None
        self.seg_size = None  # Segment size vector
        self.pos = None  # Position of t
        self.TF_cnt = None  # Number of TF that the current segment to be played in the current time step, [0, seg_TF]
        self.seg_cnt = None  # Index of the segment that is going to be played in the current time step, [0, num_seg]
        self.buffer = None
        self.cum_buffer = None
        self.total_size = None
        self.progress = None
        self.ext_trajectory = None  # external defined user pos
        self.road = None
        self.stop_time = None
        self.stop_cnt = None

        # Reset state
        self.state = self.reset()
        self.state_dim = len(self.state)
        self.action_dim = 1

        # Params for computing the polynomial approximation of exp(-E1^(-1)(x))/E1^(-1)(x) from MATLAB
        self.p = loadmat('matlab/func_fitting/p.mat')['p'].astype(np.float32)

    def reset(self, ext_seg_size=None, ext_trajectory=None, ext_road_choice=None, reset_to_playback_start=True):

        self.TF_cnt = 1  # Start from 1 because the video is played (i.e. episode starts) after receiving the first seg
        self.seg_cnt = 0

        if ext_seg_size is None:
            self.seg_size = np.minimum(np.random.normal(self.vbr_mean, self.vbr_std, self.num_seg),
                                       self.vbr_max) * self.delta_T * self.seg_TF
        else:
            self.seg_size = ext_seg_size
        self.total_size = np.sum(self.seg_size)

        if ext_road_choice is None:
            self.road = np.random.choice(len(self.road_distances))
        else:
            self.road = ext_road_choice

        if ext_trajectory is None:
            self.ext_trajectory = None
            self.speed = np.random.uniform(self.speed_min, self.speed_max)
            pos_0 = np.random.uniform(0, 2 * self.cell_radius)
            self.pos = [pos_0 for _ in range(self.Nt)]  # duplicated padding
            self.stop_time = np.random.uniform(self.stop_time_range[0], self.stop_time_range[1])
            self.stop_cnt = 0
        else:
            self.ext_trajectory = deque(ext_trajectory)
            pos_0 = self.ext_trajectory.popleft()
            self.pos = [pos_0 for _ in range(self.Nt)]

        self.buffer = self.seg_size[0]
        self.cum_buffer = self.buffer
        self.progress = self.cum_buffer / self.total_size

        snr = self.compute_snr()
        self.state = np.r_[self.progress, self.TF_cnt, self.seg_size[self.seg_cnt], self.buffer, snr]

        if reset_to_playback_start:
            [self.step() for _ in range(self.playback_start)]

        return self.state

    def step(self, rate=None):
        energy, shortage, done, cut_off = 0, 0, 0, 0

        if rate is not None:
            rate = np.squeeze(rate)
            max_rate = (self.total_size - self.cum_buffer) / self.delta_T

            # buffer evolution
            if self.TF_cnt == self.seg_TF:  # the current segment is finishing
                self.TF_cnt = 0
                self.buffer = self.buffer + rate * self.delta_T - self.seg_size[self.seg_cnt]
                self.seg_cnt += 1
            else:
                self.buffer = self.buffer + rate * self.delta_T

            self.cum_buffer = self.cum_buffer + rate * self.delta_T
            self.progress = self.cum_buffer / self.total_size  # total video downloading progress

            # stalling or not
            if self.seg_size[self.seg_cnt] <= self.buffer:
                self.TF_cnt += 1
            else:
                shortage = self.seg_size[self.seg_cnt] - self.buffer

            rate = np.minimum(rate, max_rate)
            energy = self.power_rate(rate, self.state[4]) * self.delta_T  # compute the energy before pos changes
            if self.cum_buffer >= self.total_size:
                done = 1

        # Position evolution
        self.pos[1:] = self.pos[:-1]
        if self.ext_trajectory is None:
            if self.pos[0] > self.stop_pos and self.stop_cnt * self.delta_T < self.stop_time:
                self.stop_cnt += 1
                if self.speed > 0:
                    self.speed_before_stop = self.speed
                self.speed = 0
            else:
                if self.speed == 0:
                    self.speed = self.speed_before_stop
                self.pos[0] = self.pos[0] + self.speed * self.delta_T
                self.speed += np.random.normal(scale=self.acc_std) * self.delta_T
                self.speed = np.clip(self.speed, self.speed_min, self.speed_max)

        else:
            self.pos[0] = self.ext_trajectory.popleft()
            if len(self.ext_trajectory) == 0:
                cut_off = 1

        # SNR
        snr_db = self.compute_snr()
        self.state = np.r_[self.progress, self.TF_cnt, self.seg_size[self.seg_cnt], self.buffer, snr_db]
        return self.state, energy, shortage, done, cut_off

    def compute_snr(self):
        """
        Compute the receive SNRs (unit transmit power) between the user and its 1st and 2nd nearest BSs
        based on the user's current position and road
        """
        snr_db = np.zeros(self.Nt * 2)
        road_dist = self.road_distances[self.road]
        for i in range(self.Nt):
            x = np.mod(self.pos[i], 2 * self.cell_radius)
            if x < self.cell_radius:
                dist_1 = np.sqrt(x ** 2 + road_dist ** 2)
                dist_2 = np.sqrt((2 * self.cell_radius - x) ** 2 + road_dist ** 2)
            else:
                dist_2 = np.sqrt(x ** 2 + road_dist ** 2)
                dist_1 = np.sqrt((2 * self.cell_radius - x) ** 2 + road_dist ** 2)

            snr_db[2 * i] = -35.3 - 37.6 * np.log10(dist_1) - self.noise  # SNR nearest BS
            snr_db[2 * i + 1] = -35.3 - 37.6 * np.log10(dist_2) - self.noise  # second nearest BS
        return snr_db

    def power_rate(self, rate, snr_dB):
        # Compute the average transmit power based on the average data rate
        snr = np.power(10, (snr_dB / 10))
        x = rate / self.bandwidth * np.log(2)
        y = np.matmul(np.power(x, np.arange(0, len(self.p))), self.p)[0]
        power = 1 / snr * (np.maximum(y, 0) - x)
        return power

    def get_next_seg_size(self):
        # Get the size of the segment to be played in the next TF
        if self.TF_cnt == self.seg_TF:
            cnt = self.seg_cnt + 1
        else:
            cnt = self.seg_cnt
        return [self.seg_size[cnt]]

    def gen_traces(self, num, length, data_dir=None, name=None, verbose=True, seed=None):
        """
        Generate traces including user position, SNR, segment size, and road choice. Save the traces to data_dir.
        :param num: Number of traces to be generated
        :param length: Length of each trace
        :param data_dir: Directory to save the trace data
        :param name: Name of the data
        :param verbose: Print the information or not
        :param seed: Random seed
        :return: SNR and user position traces
        """
        if seed is not None:
            np.random.seed(seed)
        pos_data = []
        snr_data = []
        seg_size_data = []
        road_data = []
        v_data = []
        for i in range(num):
            if verbose:
                print('Generating trace: {}'.format(i))

            self.reset(reset_to_playback_start=False)
            seg_size_data.append(self.seg_size)  # Generate seg size
            road_data.append(self.road)  # Generate road choice

            # Generate pos
            pos_trace = [self.pos[0]]
            snr_trace = [self.state[4]]
            v_trace = [self.speed]
            for t in range(length - 1):
                self.step()
                pos_trace.append(self.pos[0])
                snr_trace.append(self.state[4])
                v_trace.append(self.speed)
            pos_data.append(pos_trace)
            snr_data.append(snr_trace)
            v_data.append(v_trace)

        if data_dir is not None:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            savemat(data_dir + '/' + name + '.mat', {'pos_data': pos_data,
                                                     'seg_size_data': seg_size_data,
                                                     'road_data': road_data,
                                                     'snr_data': snr_data,
                                                     'v_data': v_data})
        return snr_data, pos_data, v_data
