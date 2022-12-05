from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.neighbors import KernelDensity



class KDEClassifier(BaseEstimator, ClassifierMixin):
    """Bayesian generative classification based on KDE
    
    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        
    def fit(self, X, y):
        """
        Get the training sets for each class\n
        Calculate class priors P(y)\n
        Compute the likelihood P(x | y)\n
        So, for a new point x, the posterior P(y | x) ~ P(y)*P(x | y)

        Parameters
        ----------
        X: numpy nd-array
            data do be fit into the model
        y: numpy 1d-array
            labels of X
        len(X) == len(y)

        Ret 
            self to chain commands.
        """
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.likelihoods_ = [KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel).fit(Xi)
                        for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
        return self
        
    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X)
                             for model in self.likelihoods_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)
        
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]

    def get_params(self, deep=True):
        return super().get_params(deep=deep)
    
    def set_params(self, **params):
        return super().set_params(**params)
