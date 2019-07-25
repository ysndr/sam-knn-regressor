import numpy as np
import sklearn.neighbors as sk
from skmultiflow.core import RegressorMixin
from skmultiflow.utils.utils import *

class SAMKNNREG(RegressorMixin):

    def __init__(self, n_neighbors=5, max_LTM_size=5000, leaf_size=30, nominal_attributes=None):
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
        r, f = get_dimensions(X)
        for i in range(r):
            self._partial_fit(X[i,:], y[i])

    def _partial_fit(self, x, y):
        self._evaluateMemories(x, y)

        self._cleanLTM(x, y)

        self.STMX.append(x)
        self.STMy.append(y)

        self._adaptSTM()

    def _cleanLTM(self, x, y):
        tree = sk.KDTree(self.STMX, self.leaf_size, metric='euclidean')
        dist, ind = tree.query(np.asarray(x), k=self.n_neighbors)
        dmax = np.amax(dist)
        dist = np.where(dist<1, dist, np.ones_like(dist))
        qmax = np.amax(np.abs( (self.STMy[ind] - y) / (dist**2) )) 

        tree = sk.KDTree(self.LTMX, self.leaf_size, metric='euclidean')
        ind, dist = tree.query_radius(np.asarray(x), dmax, return_distance=True)





    def _evaluateMemories(self, x, y):
        self.STMerror += (self.STMpredict(x)-y)**2
        self.LTMerror += (self.LTMpredict(x)-y)**2
        self.COMBerror += (self.COMBpredict(x)-y)**2

    def _adaptSTM(self):
        pass

    def predict(self, X):
        """ predict
        
        Predicts the label of the X sample, by searching the KDTree for 
        the n_neighbors-Nearest Neighbors.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.
            
        Returns
        -------
        list
            A list containing the predicted labels for all instances in X.
        
        """
        r, c = get_dimensions(X)
        proba = self.predict_proba(X)
        predictions = []
        for i in range(r):
            predictions.append(np.argmax(proba[i]))

        tree = sk.KDTree(self.window.get_attributes_matrix(), self.leaf_size, metric='euclidean')
        dist, ind = tree.query(np.asarray(X), k=self.n_neighbors)

        return np.array(predictions)

    def predict_proba(self, X):
        """ predict_proba
         
        Calculates the probability of each sample in X belonging to each 
        of the labels, based on the knn algorithm.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
        
        Raises
        ------
        ValueError: If there is an attempt to call this function before, 
        at least, n_neighbors samples have been analyzed by the learner, a ValueError
        is raised.
        
        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, n_features), in which each outer entry is 
            associated with the X entry of the same index. And where the list in 
            index [i] contains len(self.target_value) elements, each of which represents
            the probability that the i-th sample of X belongs to a certain label.
         
        """
        #if self.window is None or self.window.n_samples < self.n_neighbors:
        #    raise ValueError("KNN must be (partially) fitted on n_neighbors samples before doing any prediction.")
        proba = []
        r, c = get_dimensions(X)

        self.classes = list(set().union(self.classes, np.unique(self.window.get_targets_matrix())))

        new_dist, new_ind = self.__predict_proba(X)
        for i in range(r):
            votes = [0.0 for _ in range(int(max(self.classes) + 1))]
            for index in new_ind[i]:
                votes[int(self.window.get_targets_matrix()[index])] += 1. / len(new_ind[i])
            proba.append(votes)

        return np.array(proba)

    def __predict_proba(self, X):
        """ __predict_proba
        
        Private implementation of the predict_proba method.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
        
        Returns
        -------
        tuple list
            One list with the k-nearest neighbor's distances and another 
            one with their indexes.
        
        """
        # To use our own KDTree implementation please replace it as follows
        # tree = KDTree(self.window.get_attributes_matrix(), metric='euclidean',
        #              nominal_attributes=self._nominal_attributes, return_distance=True)

        tree = sk.KDTree(self.window.get_attributes_matrix(), self.leaf_size, metric='euclidean')
        dist, ind = tree.query(np.asarray(X), k=self.n_neighbors)
        return dist, ind