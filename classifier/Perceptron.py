# -*- coding:utf-8 -*-

class Perceptron(object):
    """Perception classifier.

    Parameters
    ----------
    eta : float
        leaning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    ----------
    w_ : ld-array
        Weight after fitting.
    errors : list
        Number of misclassifications in every epoch
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ Fit training data:

        :param X: {array-like}, shape = [n_sample, n_features] Training vectors, where n_samples is the number of samples and
                n_features is the number of features.
        :param y: array-like, shape = [n_sample]
        --------
        :return: self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                predict = self.predict(xi)
                update = self.eta * (target - predict)
                self.w_[1:] += update * xi
                self.w_[0] = update
                errors += int(update != 0.0)
                #print("target: %d, predict: %d, update: %.3f" % (target, predict, update))
                #print("Wehght: %s" % self.w_)
            self.errors_.append(errors)
        return self

    def net_input(self, xi_vec):
        """Caculate net input"""
        return np.dot(xi_vec, self.w_[1:]) + self.w_[0]

    def predict(self, xi_vec):
        """Return class label after unit step"""
        return np.where(self.net_input(xi_vec) >= 0.0, 1, -1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    # df.tail()
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)

    # plot to show data
    plt.subplot(211)
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend(loc='upper left')
    #plt.show()

    # plot perception
    plt.subplot(212)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()
