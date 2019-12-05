import numpy as np
import sklearn
from sklearn.base import BaseEstimator

class ConcentrationFreeOutlierFactor(BaseEstimator):
    def __init__(self, rho=0.01, contamination=.1):
        self.rho = rho
        self.contamination = contamination
    
    def fit(self, X, y=None):
        pass
    
    def predict(self, X):
        raise NotImplementedError()
    
    def fit_predict(self, X):
        """ Predict outlier labels, similar to sklearn (-1 for outlier, 1 for inlier) """
        scores = self.score_samples(X)
        threshold = np.percentile(scores, (100 - self.contamination * 100.))
        return np.where(scores > threshold, -1, 1)
    
    def score_samples(self, X):
        """
        returns: raw CFOF scores. Bigger score indicates more 'outlying' sample
        """
        n, m = X.shape
        # Step 1. Calculate pairwise distances between points
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self._distance(X[i, :], X[j, :])
                D[i, j] = d
                D[j, i] = d
        # Step 2. Get neighbors indices
        neighbors = np.argsort(D)
        # Step 3. Calculate reverse neighbor distance
        NN = np.zeros((n, n))
        # min k such that i contains j in its neighbourhood
        for i in range(n):
            for j in range(n):
                NN[i, j] = np.where(neighbors[i] == j)[0][0]

        threshold = self.rho * n
        cfof = np.zeros((n,))
        for i in range(n):
            # now, find reverse neighbors set with minimal sufficient width
            for k in range(1, n):
                if len(np.where(NN[:, i] <= k)[0]) > threshold:
                    cfof[i] = k / n
                    break
        return cfof

    @staticmethod
    def _distance(a, b):
        """ Euclidean distance between two points. You're free to replace it with anything more relevant. """
        return np.sqrt(((a - b) ** 2).sum())