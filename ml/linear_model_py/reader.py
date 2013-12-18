# -*-coding:utf-8-*-

__author__ = "Jiawang Liu (ljwsummer@gmail.com)"
__date__ = "2012/12"

import sys
import numpy as np

class Error(Exception):
    pass

class Data(object):
    """read data"""

    def __init__(self, input_file):
        self.Y = np.matrix('')
        self.X = np.matrix('')
        self.qid = np.matrix('')
        self.append_file(input_file)

    def append_line(self, line):
        sp = line.split()
        appd = lambda m1, m2: m2 if m1.size == 0 else np.vstack([m1, m2])
        self.Y = appd(self.Y, np.matrix(sp[0]))
        start = 1
        if sp[1].startswith('qid'):
            qid = sp[1].split(':')[1]
            self.qid = appd(self.qid, np.matrix(qid))
            start = 2
        featv = []
        for n in xrange(start, len(sp)):
            idx, val = sp[n].split(':')
            idx = int(idx)
            if idx > len(featv):
                featv += ['0'] * (idx - len(featv))
            featv[idx - 1] = val
        if self.X.size == 0:
            self.X = np.matrix(','.join(featv))
        else:
            delta = len(featv) - self.X.shape[1]
            if delta > 0:
                temp = np.mat(np.zeros((self.X.shape[0], delta)))
                self.X = np.hstack([self.X, temp])
            elif delta < 0:
                featv += ['0'] * abs(delta)
            else:
                pass
            self.X = np.vstack([self.X, np.matrix(','.join(featv))])
        self._check()

    def append_file(self, input_file):
        fp = open(input_file)
        try:
            for line in fp:
                self.append_line(line)
        except Error as error:
            print >> sys.stderr, error
        finally:
            fp.close()

    def _check(self):
        if self.Y.size != self.qid.size or self.Y.size != self.X.shape[0]:
            raise Error('Exception: Input data format error.')



