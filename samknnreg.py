import numpy as np
import datagen
import math
import sklearn.neighbors as sk
import matplotlib.pyplot as plt
import time
from skmultiflow.core import RegressorMixin
from pykdtree.kdtree import KDTree


#from skmultiflow.utils.utils import *

class SAMKNNRegressor(RegressorMixin):

    def __init__(self, n_neighbors=5, max_LTM_size=5000, leaf_size=30, nominal_attributes=None):
        """
        test text
        """
        self.n_neighbors = n_neighbors # k
        self.max_LTM_size = max_LTM_size # LTM size 
        self.STMX, self.STMy, self.LTMX, self.LTMy = ([], [], [], [])
        self.STMerror, self.LTMerror, self.COMBerror = (0, 0, 0)
        self.leaf_size = leaf_size

        self.adaptions = 0
            
        #self.window = InstanceWindow(max_size=max_window_size, dtype=float)


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
            # if(i%100 == 0):
                # print("fitting:", X[i,:], y[i])
            self._partial_fit(X[i,:], y[i])

    def _partial_fit(self, x, y):
        self.STMX.append(x)
        self.STMy.append(y)
        
        # build up initial LTM
        if len(self.LTMX) < self.n_neighbors:
            self.LTMX.append(x)
            self.LTMy.append(y)

        if len(self.STMX) < self.n_neighbors:
            return

        self._evaluateMemories(x, y)
        start_time = time.time()
        self._adaptSTM()
        end_time = time.time()

        # print(end_time - start_time, "seconds taken for adaptSTM")
        self._cleanLTM(x, y)

    def _cleanDiscarded(self, discarded_X, discarded_y):        
        stm_tree = KDTree(np.array(self.STMX))
        STMy = np.array(self.STMy)

        clean_mask = np.zeros(discarded_X.shape[0], dtype=bool)

        for x,y in zip(self.STMX, self.STMy):
            # searching for points from the stm in the stm will also return that points, so we query one more to get k neighbours
            dist, ind = stm_tree.query(np.array([x]), k=self.n_neighbors +1)
            dist = dist[0]
            ind = ind[0]

            
            """
            find weighted maximum difference and max distance among next n neighbours in STM
            """
            dist_max = np.amax(dist)
            w_diff = self._clean_metric(STMy[ind] - y, dist, dist_max)
            w_diff_max = np.amax(w_diff)

            # print("w_diff_max:", w_diff_max)
            """
            Query all points among the discarded that lie inside the maximum distance. Delete every point that has a greater weighted difference than the previously gathered maximum distance.
            """
            discarded_tree = KDTree(discarded_X)

            dist, ind = discarded_tree.query(
                np.array([x]),
                k=len(discarded_X),
                distance_upper_bound=dist_max)
            
            keep = ind < len(discarded_X)
            ind = ind[keep]
            dist = dist[keep]


            disc_w_diff = self._clean_metric(discarded_y[ind] - y, dist, dist_max)
                        
            clean = ind[disc_w_diff < w_diff_max]

            """
            We create a mask which us used to drop all values in the discarded
            set whose weighted difference is to far from __all__ points.
            E.g. it does not appear in neighbourhood of any other point or
            is too different if it does.
            """

            clean_mask[clean] = True

        discarded_X = discarded_X[clean_mask]
        discarded_y = discarded_y[clean_mask]

        return discarded_X, discarded_y

    def _clean_metric(self, diffs, dists, norm=1.):
        # inverse distance weighting
        # TODO: find factor
        return np.abs(diffs) * 1/np.exp(dists/norm)
        


    def _cleanLTM(self, x, y):
        LTMX = np.array(self.LTMX)
        LTMy = np.array(self.LTMy)
        STMX = np.array(self.STMX)
        STMy = np.array(self.STMy)

        stmtree = KDTree(STMX)#, self.leaf_size, metric='euclidean')
        
        dist, ind = stmtree.query(np.array([x]), k=self.n_neighbors)
        dist = dist[0] # only queriing one point
        ind = ind[0] # ^
        #dist = np.where(dist < 1, 1, dist) # TODO: what about really close points?
        
        # print(dist)
        dist_max = np.amax(dist)
        
        qs = self._clean_metric(STMy[ind] - y, dist, dist_max)
        w_diff_max = np.amax(qs)

        ltmtree = KDTree(LTMX) #, self.leaf_size, metric='euclidean')
        dist, ind = ltmtree.query(
            np.array([x]),
            k=len(LTMX),
            distance_upper_bound=dist_max)

        keep = ind < len(LTMX)
        ind = ind[keep]
        dist = dist[keep]


        #dist = np.where(dist < 1, 1, dist) ^
        qstest = .5 * self._clean_metric(LTMy[ind] - y, dist, dist_max)
        dirty = ind[qstest > w_diff_max]


        if(dirty.size):
            if (LTMX.shape[0] - len(dirty) < 5):
                # print("LTM dirty but too small!")
                return
            self.LTMX = np.delete(LTMX, dirty, axis=0).tolist()
            self.LTMy = np.delete(LTMy, dirty, axis=0).tolist()
            # print("LTM cleaned")


    def _predict(self, X, y, x):
        X = np.array(X)
        y = np.array(y)

        tree = KDTree(X) #, self.leaf_size, metric='euclidean')
        dist, ind = tree.query(np.array([x]), k=self.n_neighbors)

        dist = dist[0]
        ind = ind[0]

        clean = np.nonzero(dist)
        dist = dist[clean]
        ind = ind[clean]
        if len(dist) == 0:
            return 0

        pred = np.sum(y[ind] / dist)
        norm = np.sum(1 / dist)

        if norm == 0:
            return 1

        return pred / norm

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

        best_MLE = self.STMerror/STMX.shape[0]
        best_size = STMX.shape[0]

        old_error = best_MLE
        old_size = best_size

        slice_size = int(STMX.shape[0]/2)
        
        while (slice_size >= 50):
            MLE = 0

            for n in range(self.n_neighbors, slice_size):
                pred = self._predict(
                    STMX[-slice_size:-slice_size+n, :],
                    STMy[-slice_size:-slice_size+n], # NOTE: multi dim y values possible?
                    STMX[-slice_size+n, :])

                MLE += math.log(1+abs(pred - STMy[-slice_size+n]))

            MLE = MLE/slice_size
            
            if (MLE < best_MLE):
                best_MLE = MLE
                best_size = slice_size
            
            slice_size = int(slice_size / 2)
        
        if(old_size != best_size):
            self.adaptions += 1

            if(len(self.STMX[0]) == 1):
                fig, ax = plt.subplots(2,2, sharex=True, sharey=True, num="Adaption #" + str(self.adaptions))
            
            
            print("ADAPTING: old size & error: ", old_size, old_error, "new size & error: ", best_size, best_MLE)
            
            discarded_X = STMX[0:-best_size, :]
            discarded_y = STMy[0:-best_size] # NOTE: multi dim y values possible?
            self.STMX = STMX[-best_size:, :].tolist()
            self.STMy = STMy[-best_size:].tolist()
            self.STMerror = best_MLE
            
            if(len(self.STMX[0]) == 1):
                ax[1][0].scatter(self.STMX, self.STMy, label="STM after adaption", s=100, alpha=.2, color='C1')

            
            original_discard_size = len(discarded_X)
            if(len(self.STMX[0]) == 1):
                ax[0][0].scatter(discarded_X, discarded_y, label="all discarded", s=100, alpha=.2, color='C2')



            discarded_X, discarded_y = self._cleanDiscarded(discarded_X, discarded_y)
            if(len(self.STMX[0]) == 1):
                ax[0][1].scatter(discarded_X, discarded_y, label="cleaned discarded -> LTM", s=100, alpha=.2, color='C3')

            
            if (discarded_X.size):
                self.LTMX += discarded_X.tolist()
                self.LTMy += discarded_y.tolist()
                if(len(self.STMX[0]) == 1):    
                    ax[1][1].scatter(self.LTMX, self.LTMy, label="LTM with new from STM", s=100, alpha=.2, color='C4')
                


                print("Added", len(discarded_X), "of", original_discard_size, "to LTM. ")
            else:
                print("All discarded Samples are dirty")

            if(len(self.STMX[0]) == 1):
                plt.figlegend()
                plt.tight_layout()
                plt.show(block=False)

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
        if (len(self.STMX) < self.n_neighbors):
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
        """
        print("LTM:")
        print(self.LTMX)
        print("STM:")
        print(self.STMX)
        """

if __name__ == "__main__":
    """
    X = np.arange(0, 1000, 5)
    np.random.shuffle(X)
    y = X**2
    X = np.reshape(X, (X.shape[0], -1))
    """
    generator = datagen.NormalBlopps(dims=1, abrupt_drift_rate=200)
    generator.prepare_for_use()
    data = [generator.next_sample(500), generator.next_sample(500), generator.next_sample(500)]
    X = []
    y = []
    for i in range(len(data)):
        X += data[i][0]
        y += data[i][1]
    X = np.array(X).astype('d')#/np.amax(X)
    y = np.array(y).astype('d')#/np.amax(y)
    X = np.reshape(X, (X.shape[0], -1))
    model = SAMKNNRegressor()
    model.fit(X, y)
    #print(model.predict(np.array([[3],[8],[15],[79]])))
    print("LTM size:", len(model.LTMX), "STM size:", len(model.STMX))
    if(X.shape[0] == 1):
        fig, ax = plt.subplots()
        ax.scatter(X, y, label="original", s=100, alpha=.1)
        ax.scatter(model.STMX, model.STMy, label="STM", s=100, alpha=.4)
        ax.scatter(model.LTMX, model.LTMy, label="LTM", s=100, alpha=.4)
        ax.legend()
        plt.show()
    model.print_model()
