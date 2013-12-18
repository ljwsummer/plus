# das
das = data analysis software? No. Actually **das** means the reverse of sad. To be honest, I feel very sad for something when I start this project.

## Introduction
A machine learning toolkit. 
It will provide various of machine learning algorithms. All of algorithms in this package will be writen by C/C++ or Python programming language. We have supported [**linear regression**](https://github.com/ljwsummer/das/blob/master/ml/linear_model_py/linear_regression.py) and [**logistic regression**](https://github.com/ljwsummer/das/blob/master/ml/linear_model_py/logistic_regression.py), both of them were implemented by Python. You can also add your own algorithms in das, here is an example: https://github.com/ljwsummer/das/tree/master/ml/svm

## Dependency
* NumPy 1.7.0-1. [*HomePage*](https://github.com/numpy/numpy)
* Python 2.7.2

## Quick Start
see help: ./train.py -h

* use linear regression:
`./train.py -c conf/ml.cfg -m linear_regression`
* use logistic regression:
`./train.py -c conf/ml.cfg -m logistic_regression`
* use svm regression:
`./train.py -c conf/ml.cfg -m svm`

If you want to tune the parameters, you can find them in configuration file(conf/ml.cfg).

