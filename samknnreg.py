import numpy as np
import math
import sklearn.neighbors as sk
from skmultiflow.core import RegressorMixin
#from skmultiflow.utils.utils import *

class SAMKNNREG(RegressorMixin):

    def __init__(self, n_neighbors=5, max_LTM_size=5000, leaf_size=30, nominal_attributes=None):
        """
        test text
        """
        self.n_neighbors = n_neighbors
        self.max_LTM_size = max_LTM_size
        self.STMX, self.STMy, self.LTMX, self.LTMy = ([], [], [], [])
        self.STMerror, self.LTMerror, self.COMBerror = (0, 0, 0)
        #self.window = InstanceWindow(max_size=max_window_size, dtype=float)
        self.c = 0
        self.first_fit = True
        self.leaf_size = leaf_size
        self.nominal_attributes = nominal_attributes
        if self.nominal_attributes is None:
            self._nominal_attributes = []

    def partial_fit(self, X, y, sample_weight=None):
        """ Partially (incrementally) fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the target values of all samples in X.

        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed. Usage varies depending on the learning method.

        Returns
        -------
        self

        """
        r = X.shape[0]
        for i in range(r):
            print("fitting:", X[i,:], y[i])
            self._partial_fit(X[i,:], y[i])

    def _partial_fit(self, x, y):
        self._evaluateMemories(x, y)
        self._cleanLTM(x, y)

        self.STMX.append(x)
        self.STMy.append(y)

        self._adaptSTM()

    def _cleanLTM(self, x, y):
    
        tree = sk.KDTree(np.array(self.STMX), self.leaf_size, metric='euclidean')
        dist, ind = tree.query(np.asarray([x]), k=self.n_neighbors)
        dist = dist[0]
        dist = np.where(dist < 0.1, 0.1, dist)
        ind = ind[0]
        dmax = np.amax(dist)
        qmax = np.amax(np.abs( (np.array(self.STMy)[ind] - y) / (dist**2) ))

        tree = sk.KDTree(np.array(self.LTMX), self.leaf_size, metric='euclidean')
        ind, dist = tree.query_radius(np.asarray([x]), dmax, return_distance=True)
        dist = dist[0]
        dist = np.where(dist < 0.01, 0.01, dist)
        ind = ind[0]
        qtest = np.abs( (np.array(self.LTMy)[ind] - y) / (dist**2) )
        dirty = ind[np.nonzero(qtest > qmax)]
        for entry in dirty:
            self.LTMX.__delitem__(entry)
        #self.LTMX = list(np.delete(np.array(self.LTMX), dirty, axis=0))

    def _predict(self, X, y, x):
        X = np.asarray(X)
        y = np.asarray(y)

        tree = sk.KDTree(X, self.leaf_size, metric='euclidean')
        dist, ind = tree.query(np.asarray(np.asarray([x])), k=self.n_neighbors)
        dist = dist[0]
        dist = np.where(dist < 0.01, 0.01, dist)
        ind = ind[0]
        norm = np.sum(1/dist)
        inverse_weighted = (y[ind]) / dist
        pred = ( np.sum(inverse_weighted) / norm )
        return pred

    def STMpredict(self, x):
        return self._predict(self.STMX, self.STMy, x)

    def LTMpredict(self, x):
        return self._predict(self.LTMX, self.LTMy, x)

    def COMBpredict(self, x):
        return self._predict(self.STMX + self.LTMX, self.STMy + self.LTMy, x)

    def _evaluateMemories(self, x, y):

        self.STMerror += math.log(abs(self.STMpredict(x)-y))
        self.LTMerror += math.log(abs(self.LTMpredict(x)-y))
        self.COMBerror += math.log(abs(self.COMBpredict(x)-y))
        print("Errors:  STM: ", self.STMerror, "  LTM: ", self.LTMerror, "  COMB: ", self.COMBerror)

    def _adaptSTM(self):
        pass

    def predict(self, X):
        """ predict
        
        Predicts the value of the X sample, by searching the KDTree for 
        the n_neighbors-Nearest Neighbors.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.
            
        Returns
        -------
        list
            A list containing the predicted values for all instances in X.
        
        """
        
        mem_list = [self.STMerror, self.LTMerror, self.COMBerror]
        best_mem_ind = np.argmin(mem_list)
        if (best_mem_ind == 0): 
            return np.array([self.STMpredict(x) for x in X])
        elif (best_mem_ind == 1): 
            return np.array([self.LTMpredict(x) for x in X])
        elif (best_mem_ind == 2): 
            return np.array([self.COMBpredict(x) for x in X])

    def fit(self, X, y):
        if (len(self.LTMX) < 10):
            self.LTMX = list(X[0:10,:]) 
            self.STMX = list(X[0:10,:]) 
            self.LTMy = list(y[0:10])
            self.STMy = list(y[0:10])
            self.partial_fit(X[10:,:], y[10:])
        else:
            self.partial_fit(X, y)

    def predict_proba(self):
        pass

    def print_model(self):
        print("Errors:  STM: ", self.STMerror, "  LTM: ", self.LTMerror, "  COMB: ", self.COMBerror)
        print("LTM:")
        print(self.LTMX)
        print("STM:")
        print(self.STMX)

if __name__ == "__main__":
    
    X = np.arange(0, 1000, 5)
    np.random.shuffle(X)
    y = X**2
    X = np.reshape(X, (X.shape[0], -1))
    print(X)
    print(y)
    model = SAMKNNREG()
    model.fit(X, y)
    print(model.predict(np.array([[3],[8],[15],[79]])))
    #model.print_model()