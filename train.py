#!/usr/bin/python
# -*-coding:utf-8-*-

import sys
import subprocess
import os
import ConfigParser
import optparse


def train(model, config_file, optparser):
  config = ConfigParser.ConfigParser()
  config.optionxform = str
  config.read(config_file)

  options = config.items(model)

  cmds = []
  for name, value in options:
    if optparser.has_option('--' + name):
      value = optparser.get_option('--' + name)
    if name == 'bin':
      cmds.insert(0, value)
    else:
      cmds.append('--' + name)
      cmds.append(value)

  print ' '.join(cmds)
  print subprocess.check_output(cmds)


def main():
  if len(sys.argv) <= 1:
    print >> sys.stderr, 'usage: ' + sys.argv[0] + ' -h'
    sys.exit(-1)

  parser = optparse.OptionParser(usage = "usage: %prog -h")
  parser.add_option('-c', '--config', dest = 'config_file',
      default = 'conf/ml.cfg',
      help = 'config file')
  parser.add_option('-m', '--model', dest = 'model',
      default = 'svm',
      help = 'choose training algorithm linear_regression/logistic_regression/svm')

  (options, args) = parser.parse_args()

  train(options.model, options.config_file, parser)


if __name__ == '__main__':
  main()

