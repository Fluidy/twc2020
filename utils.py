import numpy as np
from scipy.io import savemat
import os
import json
import tensorflow as tf


class Logger(object):
    def __init__(self, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.log_dir = log_dir
        self.tf_file_writer = tf.summary.FileWriter(log_dir)
        self.log = dict()  # contains all the logged information
        self.buffer = dict()  # temporary buffer within one epoch

    def add_log(self, idx, record_dict):
        tf_summary = tf.Summary()  # create a tf summary row
        for key, value in record_dict.items():
            tf_summary.value.add(tag=key, simple_value=value)

            if not (key in self.log.keys()):  # If key does not exist, create one
                self.log[key] = []

            self.log[key].append(value)

        self.tf_file_writer.add_summary(tf_summary, idx)  # add row and idx into the file writer

    def add_buffer(self, record_dict):
        for key, value in record_dict.items():
            if not (key in self.buffer.keys()):  # If key does not exist, create one
                self.buffer[key] = []

            self.buffer[key].append(value)

    def summarize_buffer(self, idx):
        record_dict = dict()
        for key, value in self.buffer.items():
            if len(self.buffer[key]) > 0:
                record_dict[key] = np.mean(value)  # Log average value (can also log max, min, or std)
                self.buffer[key] = []
            else:  # When the epoch idx add nothing to the buffer and there is a previous value in log
                record_dict[key] = self.get(key)  # get the most recent value
        self.add_log(idx, record_dict)
        self.tf_file_writer.flush()

    def dump_log(self):
        savemat(self.log_dir + '/log.mat', self.log)

    def get(self, key):
        return self.log[key][-1]


def save_params(params, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '/' + name + '.json', 'w') as fp:
        json.dump(params, fp, separators=(',', ':\t'), indent=4, sort_keys=False)


def load_params(path, name):
    with open(path + '/' + name + '.json', 'r') as fp:
        return json.load(fp)


