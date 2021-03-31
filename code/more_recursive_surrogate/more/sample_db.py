from collections import deque
import numpy as np


class SimpleSampleDatabase:
    def __init__(self, size):
        self.size = size
        self.data_x = deque(maxlen=size)
        self.data_y = deque(maxlen=size)

    def add_data(self, data_x, data_y):
        self.data_x.extend(data_x)
        self.data_y.extend(data_y)

    def get_data(self):
        data_x = np.vstack(self.data_x)
        data_y = np.vstack(self.data_y)
        return data_x, data_y

    def add_data_with_counter(self, data_x, data_y):
        """Add samples with counter to indicate how often they were used.

        Before adding new samples to the deque all counters
        are incremented.

        Args:
            data_x (list) : first element sample, second element counter
            data_y (np.ndarray) : rewards
        """
        for d in self.data_x:
            d[1] += 1

        self.data_x.extend(data_x)
        self.data_y.extend(data_y)

    def get_data_with_counter(self):
        """Get samples, reward and counter for samples.

        Returns:
            data_x (np.ndarray) : samples
            data_y (np.ndarray) : reward
            counter (list) : indicate for each sample how often it was used
        """
        counter = [d[1] for d in self.data_x]
        data_x = np.vstack([d[0] for d in self.data_x])
        data_y = np.vstack(self.data_y)
        return data_x, data_y, counter
