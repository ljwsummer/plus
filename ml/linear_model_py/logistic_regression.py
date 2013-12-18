# -*-coding:utf-8-*-

__author__ = "Jiawang Liu (ljwsummer@gmail.com)"
__date__ = "2013/02/22 12:03"

import sys
import optparse
import math

import numpy as np

from linear_model import LinearModel

class LogisticReg(LinearModel):
  """logistic regression"""

  def __init__(self, options):
    LinearModel.__init__(self)
    self.alpha = options.alpha
    self.iters = options.num_iters
    self.lambd = options.lambd
    self.intercept = options.has_intercept
    self.model_in = options.model_in
    self.model_out = options.model_out
    self.training_file = options.training_file
    self.test_file = options.test_file
    self.results_file = options.results_file
    if self.test_file != '':
      self._load_data(self.test_file)
    else:
      self._load_data(self.training_file)

  def _sigmoid(self, z):
    return 1 / (1 + np.power(math.e, -z))

  def _compute_cost(self, X, Y, theta):
    m = X.shape[0]
    res = - (np.multiply(Y, np.log(self._sigmoid(X * theta)))
            + np.multiply(1 - Y, np.log(1 - self._sigmoid(X * theta))))
    idx = 1 if self.intercept else 0
    reg = self.lambd * sum(np.power(theta[idx:], 2)) / (2 * m)
    j = sum(res) / m + reg
    return j[0,0]

  def _compute_gradient(self, X, Y, theta):
    if self.intercept:
        ret = X.T * (self._sigmoid(X * theta) - Y) + np.vstack([np.matrix('0'), self.lambd * theta[1:]])
    else:
        ret = X.T * (self._sigmoid(X * theta) - Y) + self.lambd * theta
    return ret

  def predict(self, X, results_file):
    ret = LinearModel.predict(self, X)
    pred = self._sigmoid(ret)
    fp = open(results_file, 'w')
    for v in pred.flat:
      print >> fp, v
    fp.close()


def main():

  if len(sys.argv) <= 1:
    print >> sys.stderr, 'usage: ' + sys.argv[0] + ' -h'
    sys.exit(-1)

  parser = optparse.OptionParser()
  parser.add_option('-a', '--alpha', dest = 'alpha', type = 'float',
      default = 0.01, help = 'step size of gradient descent')
  parser.add_option('-n', '--num_iters', dest = 'num_iters', type = 'int',
      default = 10000, help = 'number of iterations')
  parser.add_option('-l', '--lambda', dest = 'lambd', type = 'float',
      default = 0.001, help = 'regularization factor')
  parser.add_option('-t', '--has_intercept', dest = 'has_intercept', type = 'int',
      default = 1, help = '1 / 0 means has intercept or not')
  parser.add_option('-i', '--model_in', dest = 'model_in', type = 'str',
      default = '', help = 'read in a model from this file')
  parser.add_option('-o', '--model_out', dest = 'model_out', type = 'str',
      default = '', help = 'write the model to this file')
  parser.add_option('-f', '--training_file', dest = 'training_file', type = 'str',
      default = '', help = 'file to be used for training')
  parser.add_option('-e', '--test_file', dest = 'test_file', type = 'str',
      default = '', help = 'file to be used for testing')
  parser.add_option('-r', '--results_file', dest = 'results_file', type = 'str',
      default = '', help = 'file to be used for writing predictions')

  (options, args) = parser.parse_args()

  logistic_reg = LogisticReg(options)

  if (options.test_file != ''):
    logistic_reg.load_model(options.model_in)
    logistic_reg.predict(logistic_reg.data.X, options.results_file)
  elif (options.training_file != ''):
    logistic_reg.train()
    logistic_reg.save_model(options.model_out)
  else:
    print >> sys.stderr, 'both training_file and test_file are null.'


if __name__ == '__main__':
    main()



