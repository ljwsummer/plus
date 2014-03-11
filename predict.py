#!/usr/bin/python
# -*-coding:utf-8-*-

import sys
import subprocess
import optparse


binmap = {
    'svm' : 'ml/svm/sofia-ml',
    'linear_regression' : 'ml/linear_model_py/linear_regression.py',
    'logistic_ression' : 'ml/linear_model_py/logistic_regression.py'
    }


def predict(options):
  cmds = []
  cmds.append(binmap[options.model])
  cmds.append('--test_file')
  cmds.append(options.test_file)
  cmds.append('--model_in')
  cmds.append(options.model_in)
  cmds.append('--results_file')
  cmds.append(options.results_file)

  print ' '.join(cmds)
  print subprocess.check_output(cmds)


def main():
  if len(sys.argv) <= 1:
    print >> sys.stderr, 'usage: ' + sys.argv[0] + ' -h'
    sys.exit(-1)

  parser = optparse.OptionParser(usage = 'usage: %prog -h')
  parser.add_option('-f', '--test_file', dest = 'test_file',
      default = '', help = 'file to be used for testing')
  parser.add_option('-i', '--model_in', dest = 'model_in',
      default = '', help = 'read in a model from this file')
  parser.add_option('-m', '--model', dest = 'model',
      default = '', help = 'choose training algorithm linear_regression/logistic_regression/svm')
  parser.add_option('-r', '--results_file', dest = 'results_file',
      default = '', help = 'file to be used for writing predictions')

  (options, args) = parser.parse_args()

  predict(options)


if __name__ == '__main__':
  main()


