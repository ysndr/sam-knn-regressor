# %%
import math
import numpy as np

from skmultiflow.data.base_stream import Stream
from sklearn.datasets import make_regression
from skmultiflow.utils import check_random_state


class JumpingSineGenerator(Stream):
    def __init__(self, random_state=None, granularity=30, cont=0.01, abrupt_drift_rate=100, offset=0):
        super().__init__()

        self.abrupt_drift_rate = abrupt_drift_rate
        self.granularity = granularity
        self.cont = cont
        self.offset = offset
        self._offset = None

        self._x_idx = 0

        self.random_state = random_state
        self._random_state = None
        # This is the actual random_state object used internally
        self.name = "Regression Generator"

    def prepare_for_use(self):
        """ Prepare the stream for usage

        Uses the make_regression function from scikit-learn to generate a 
        regression problem. This problem will be kept in memory and provided 
        as demanded.

        """
        self._random_state = check_random_state(self.random_state)
        self._offset = self.offset
        self._x_idx = 0

    def n_remaining_samples(self):
        """
        Returns
        -------
        int
            Number of samples remaining.
        """
        return -1

    def has_more_samples(self):
        """
        Returns
        -------
        Boolean
            True if stream has more samples.
        """
        return True

    def next_sample(self, batch_size=1):
        """ next_sample

        Returns batch_size samples from the generated regression problem.

        Parameters
        ----------
        batch_size: int
            The number of sample to return.

        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix and the labels matrix for 
            the batch_size samples that were requested.

        """

        X = []
        y = []
        for x in range(batch_size):
            if self._random_state.rand() < 1/self.abrupt_drift_rate:
                self.offset = self.offset + self._random_state.rand()
            X.append(x + self._x_idx)
            y.append(math.sin(self.offset + x/(2*math.pi*self.granularity)
                              ) + self._random_state.normal(scale=0.05) + self.cont)

        self._x_idx += batch_size

        return (X, y)
    ` `

    def restart(self):
        """
        Restart the stream to the initial state.
        """
        # Note: No need to regenerate the data, just reset the idx
        self.prepare_for_use()

    def get_data_info(self):
        return "Jumping Sine Generator"


def stairs(seed=None, size=1000, granularity=30, cont=0.01, abrupt=100):
    if seed:
        np.random.seed(seed=seed)
    start = np.random.rand()*100
    data = []
    for x in range(size):
        if np.random.rand() < 1/abrupt:
            start = start + (200 + np.random.rand() * 200)/granularity
        data.append([x, start + x/granularity +
                     np.random.normal(scale=0.05) + cont])
    return data
