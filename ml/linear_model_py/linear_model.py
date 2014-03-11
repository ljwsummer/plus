# -*-coding:utf-8-*-

__author__ = "Jiawang Liu (ljwsummer@gmail.com)"
__date__ = "2013/02/21 12:07"

import ConfigParser
import numpy as np

import reader

class Error(Exception):
    pass

class LinearModel(object):
    """base class of linear model"""

    def __init__(self):
        if self.__class__ is LinearModel:
            raise Error("this is a abstract class")
        self.costs = []

    def _load_data(self, data_file):
        if data_file != '':
            self.data = reader.Data(data_file)
            self.m = self.data.X.shape[0]
            self.n = self.data.X.shape[1]
            self.theta = np.mat(np.zeros((self.n, 1)))

    def _compute_cost(self):
        pass

    def _compute_gradient(self):
        pass

    def _gradient_descent(self):
        grad = self._compute_gradient(self.data.X, self.data.Y, self.theta)
        self.theta -= self.alpha * grad / self.m

    def train(self):
        self.theta = np.mat(np.zeros((self.n, 1)))
        for x in xrange(self.iters):
            self._gradient_descent()
            cost = self._compute_cost(self.data.X, self.data.Y, self.theta)
            self.costs.append(cost)

    def predict(self, X):
        delta = X.shape[1] - self.theta.shape[0]
        if delta < 0:
            theta = self.theta[:X.shape[1]]
        elif delta > 0:
            theta = np.vstack([self.theta, np.mat(np.zeros((delta, 1)))])
        else:
            theta = self.theta
        return X * theta

    def save_model(self, model_out):
        fp = open(model_out, 'w')
        t = [str(x) for x in self.theta.flat]
        print >> fp, ';'.join(t)
        fp.close()

    def load_model(self, model_in):
        fp = open(model_in)
        self.theta = np.matrix(fp.readline().strip())
        fp.close()

