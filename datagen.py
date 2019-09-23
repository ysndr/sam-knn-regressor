# %%
import math
import numpy as np
from random import shuffle

from skmultiflow.data.base_stream import Stream
from sklearn.datasets import make_regression
from skmultiflow.utils import check_random_state

class NormalBlopps(Stream):
    def __init__(self, random_state=None, var=1, dims=1, cont=0.01, abrupt_drift_rate=100):
        super().__init__()

        self.abrupt_drift_rate = abrupt_drift_rate
        self.dims = dims
        self.var = var
        self.cont = cont
        self.offset = 0
        self._offset = None

        self._x_idx = 0

        self.random_state = random_state
        self._random_state = None
        # This is the actual random_state object used internally
        self.name = "Regression Generator"

    def prepare_for_use(self):
        """ Prepare the stream for usage
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
        dimfaks = [np.random.rand() * 5 for _ in range(self.dims)]
        dimpots = [int(np.random.rand() * 4) for _ in range(self.dims)]
        dimvars = [np.random.rand() * self.var for _ in range(self.dims)]
        dimmeans = [np.random.rand() * 100 for _ in range(self.dims)]

        for _ in range(batch_size):
            #check for abrupt drift
            if self._random_state.rand() < 1/self.abrupt_drift_rate:
                dimfaks = [np.random.rand() * 5 for _ in range(self.dims)]
                dimpots = [int(np.random.rand() * 4) for _ in range(self.dims)]
                dimvars = [np.random.rand() * self.var for _ in range(self.dims)]
                dimmeans = [np.random.rand() * 100 for _ in range(self.dims)]

            value = 0
            sample = []
            for i in range(self.dims):
                sample.append(np.random.normal(loc=dimmeans[i], scale=dimvars[i]))
                value += dimfaks[i] * (sample[i] ** dimpots[i])
            
            X.append(sample)
            y.append(value)

        self._x_idx += batch_size

        return (X, y)

    def restart(self):
        """
        Restart the stream to the initial state.
        """
        # Note: No need to regenerate the data, just reset the idx
        self.prepare_for_use()

    def get_data_info(self):
        return "NormalBlopps Generator"

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
            y.append(math.sin(self.offset + x/(2*math.pi*self.granularity))
                     + self._random_state.normal(scale=0.05) + self.cont)

        self._x_idx += batch_size

        return (X, y)

    def restart(self):
        """
        Restart the stream to the initial state.
        """
        # Note: No need to regenerate the data, just reset the idx
        self.prepare_for_use()

    def get_data_info(self):
        return "Jumping Sine Generator"


class StairsGenerator(Stream):
    def __init__(self, random_state=None, granularity=10, abrupt_drift_rate=300, offset=0):
        super().__init__()

        self.abrupt_drift_rate = abrupt_drift_rate
        self.granularity = granularity
        self.offset = offset
        self._offset = None

        self._x_idx = 0

        self.random_state = random_state
        self._random_state = None
        # This is the actual random_state object used internally
        self.name = "Regression Generator"

    def prepare_for_use(self):
        """ Prepare the stream for usage
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
        self.cont = 2
        for x in range(batch_size):
            if x > batch_size/2 and self.cont == 2:
                self.cont = 5*np.random.rand()
            if self._random_state.rand() < 1/self.abrupt_drift_rate:
                #self.offset += 1
                pass
            y.append(self.offset + (x*self.cont)/self.granularity + self._random_state.normal(scale=1))
            X.append(x)
        self._x_idx += batch_size
        zipped = list(zip(X, y))
        shuffle(zipped)
        

        return ([zipped[i][0] for i in range(len(X))], [zipped[i][1] for i in range(len(X))])


    def restart(self):
        """
        Restart the stream to the initial state.
        """
        # Note: No need to regenerate the data, just reset the idx
        self.prepare_for_use()

    def get_data_info(self):
        return "Stairs Generator"