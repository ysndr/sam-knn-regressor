import numpy as np
import datagen
import math
import sklearn.neighbors as sk
import matplotlib.pyplot as plt
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
            if(i%100 == 0):
                print("fitting:", X[i,:], y[i])
            self._partial_fit(X[i,:], y[i])

    def _partial_fit(self, x, y):
        self._evaluateMemories(x, y)
        self._cleanLTM(x, y)

        self.STMX.append(x)
        self.STMy.append(y)

        self._adaptSTM()

    def _cleanDiscarded(self, discarded_X, discarded_y):
        stm_tree = sk.KDTree(np.array(self.STMX), self.leaf_size, metric='euclidean')
        for x,y in zip(self.STMX, self.STMy):
            dist, ind = stm_tree.query(np.asarray([x]), k=self.n_neighbors)
            dist = dist[0]
            dist = np.where(dist < 1, 1, dist)
            ind = ind[0]
            dmax = np.amax(dist)
            qmax = np.amax(self._clean_metric(np.array(self.STMy)[ind] - y, dist))
            print("qmax:", qmax)

            discarded_tree = sk.KDTree(discarded_X, self.leaf_size, metric='euclidean')
            ind, dist = discarded_tree.query_radius(np.asarray([x]), dmax, return_distance=True)
            dist = dist[0]
            dist = np.where(dist < 1, 1, dist)
            ind = ind[0]
            qtest = 0.9*(self._clean_metric(discarded_y[ind] - y, dist))
            print(qtest)
            dirty = ind[np.nonzero(qtest > qmax)]
            if(dirty.size):
                discarded_X = np.delete(np.array(discarded_X), dirty, axis=0)
                discarded_y = np.delete(np.array(discarded_y), dirty, axis=0)
                print("Discarded Set cleaned")
        return discarded_X, discarded_y

    def _clean_metric(self, diffs, dists):
        return np.abs(diffs / (dists))


    def _cleanLTM(self, x, y):
        LTMX = np.array(self.LTMX)
        tree = sk.KDTree(np.array(self.STMX), self.leaf_size, metric='euclidean')
        dist, ind = tree.query(np.asarray([x]), k=self.n_neighbors)
        dist = dist[0]
        dist = np.where(dist < 1, 1, dist)
        ind = ind[0]
        dmax = np.amax(dist)
        qmax = np.amax(self._clean_metric( np.array(self.STMy)[ind] -y, dist))

        tree = sk.KDTree(LTMX, self.leaf_size, metric='euclidean')
        ind, dist = tree.query_radius(np.asarray([x]), dmax, return_distance=True)
        dist = dist[0]
        dist = np.where(dist < 1, 1, dist)
        ind = ind[0]
        qtest = 0.9*(self._clean_metric( np.array(self.LTMy)[ind] - y, dist))
        dirty = ind[np.nonzero(qtest > qmax)]
        if(dirty.size):
            if (LTMX.shape[0] - len(dirty) < 5):
                print("LTM dirty but too small!")
                return
            self.LTMX = list(np.delete(np.array(self.LTMX), dirty, axis=0))
            self.LTMy = list(np.delete(np.array(self.LTMy), dirty, axis=0))
            print("LTM cleaned")


    def _predict(self, X, y, x):
        X = np.asarray(X)
        y = np.asarray(y)

        tree = sk.KDTree(X, self.leaf_size, metric='euclidean')
        dist, ind = tree.query(np.asarray(np.asarray([x])), k=self.n_neighbors)
        dist = dist[0]
        dist = np.where(dist < 0.1, 0.1, dist)
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

        self.STMerror += math.log(1+abs(self.STMpredict(x)-y))
        self.LTMerror += math.log(1+abs(self.LTMpredict(x)-y))
        self.COMBerror += math.log(1+abs(self.COMBpredict(x)-y))
        #print("Errors:  STM: ", self.STMerror, "  LTM: ", self.LTMerror, "  COMB: ", self.COMBerror)

    def _adaptSTM(self):
        STMX = np.array(self.STMX)
        STMy = np.array(self.STMy)
        best_MLE = self.STMerror/len(self.STMX)
        old_error = best_MLE
        best_size = STMX.shape[0]
        old_size = best_size
        slice_size = int(STMX.shape[0]/2)
        while (slice_size >= 50):
            MLE = 0
            for n in range(self.n_neighbors, slice_size):
                pred = self._predict(STMX[-slice_size:-slice_size+n, :], STMy[-slice_size:-slice_size+n], STMX[-slice_size+n, :])
                MLE += math.log(1+abs(pred - STMy[-slice_size+n]))
            MLE = MLE/slice_size
            if (MLE < best_MLE):
                best_MLE = MLE
                best_size = slice_size
            slice_size = int(slice_size / 2)
        
        if(old_size != best_size):
            _, ax = plt.subplots()
            print("ADAPTING: old size & error: ", old_size, old_error, "new size & error: ", best_size, best_MLE)
            discarded_X = STMX[0:-best_size, :]
            discarded_y = STMy[0:-best_size]
            self.STMX = list(STMX[-best_size:, :])
            self.STMy = list(STMy[-best_size:])
            self.STMerror = best_MLE
            ax.scatter(self.STMX, self.STMy, label="newSTM", s= 13)
            original_discard_size = discarded_X.size
            ax.scatter(list(discarded_X), list(discarded_y), label="oldDiscard", s= 10)
            discarded_X, discarded_y = self._cleanDiscarded(discarded_X, discarded_y)
            ax.scatter(list(discarded_X), list(discarded_y), label="cleanedDiscard", s= 10)
            
            if (discarded_X.size):
                self.LTMX += (list(discarded_X))
                self.LTMy += (list(discarded_y))
                print("Added", discarded_X.size, "from", original_discard_size, "to LTM. ")
                #ax.scatter(self.LTMX, self.LTMy, label="LTM", s= 6)
            else:
                print("All discarded Samples are dirty")
            ax.legend()
            plt.show()

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
        if (len(self.STMX) < 10):
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
    """
    X = np.arange(0, 1000, 5)
    np.random.shuffle(X)
    y = X**2
    X = np.reshape(X, (X.shape[0], -1))
    """
    generator = datagen.StairsGenerator()
    generator.prepare_for_use()
    data = [generator.next_sample(500), generator.next_sample(500), generator.next_sample(500)]
    X = []
    y = []
    for i in range(len(data)):
        X += data[i][0]
        y += data[i][1]
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], -1))
    print(X)
    print(y)
    model = SAMKNNREG()
    model.fit(X, y)
    #print(model.predict(np.array([[3],[8],[15],[79]])))
    print(len(model.LTMX), len(model.STMX))
    fig, ax = plt.subplots()
    ax.scatter(X, y, label="original")
    ax.scatter(model.STMX, model.STMy, label="STM", s= 6)
    ax.scatter(model.LTMX, model.LTMy, label="LTM", s = 6)
    ax.legend()
    plt.show()