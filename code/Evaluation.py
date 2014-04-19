from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import pylab as pl

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics


def Evaluate(x_train, y_train, x_test, y_test, select_chi2=None):

  # Display progress logs on stdout
  logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s %(levelname)s %(message)s')

  if select_chi2:
      print("Extracting %d best features by a chi-squared test" % select_chi2)
      t0 = time()
      ch2 = SelectKBest(chi2, k=select_chi2)
      x_train = ch2.fit_transform(x_train, y_train)
      x_test = ch2.transform(x_test)
      print("done in %fs" % (time() - t0))
      print()

  results = []
  for clf, name in (
          (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
          (Perceptron(n_iter=50), "Perceptron"),
          (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
          (KNeighborsClassifier(n_neighbors=10), "kNN")):
      print('=' * 80)
      print(name)
      results.append(benchmark(clf, x_train, y_train, x_test, y_test))

  for penalty in ["l2", "l1"]:
      print('=' * 80)
      print("%s penalty" % penalty.upper())
      # Train Liblinear model
      results.append(benchmark(LinearSVC(loss='l2', penalty=penalty, dual=False, tol=1e-3), x_train, y_train, x_test, y_test))

      # Train SGD model
      results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty=penalty), x_train, y_train, x_test, y_test))

  # Train SGD with Elastic Net penalty
  print('=' * 80)
  print("Elastic-Net penalty")
  results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"), x_train, y_train, x_test, y_test))

  # Train NearestCentroid without threshold
  print('=' * 80)
  print("NearestCentroid (aka Rocchio classifier)")
  results.append(benchmark(NearestCentroid(),x_train, y_train, x_test, y_test))

  #TODO: Normalize Data so we can use this
  # Train sparse Naive Bayes classifiers
  print('=' * 80)
  print("Naive Bayes")
  #results.append(benchmark(MultinomialNB(alpha=.01),x_train, y_train, x_test, y_test))
  #results.append(benchmark(BernoulliNB(alpha=.01),x_train, y_train, x_test, y_test))



def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

def benchmark(clf, x_train, y_train, x_test, y_test):
    categories = [str(el) for el in np.unique(y_train)]
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(x_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(x_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.f1_score(y_test, pred)
    print("f1-score:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        # if opts.print_top10 and feature_names is not None:
        #     print("top 10 keywords per class:")
        #     for i, category in enumerate(categories):
        #         top10 = np.argsort(clf.coef_[i])[-10:]
        #         print(trim("%s: %s"
        #               % (category, " ".join(feature_names[top10]))))
        # print()

    # if opts.print_report:
    if True:
        print("classification report:")
        print(metrics.classification_report(y_test, pred, target_names=categories))

    #if opts.print_cm:
    if False:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time
    print('=' * 80)
    print("LinearSVC with L1-based feature selection")
    results.append(benchmark(L1LinearSVC()))


    # make some plots

    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    pl.figure(figsize=(12,8))
    pl.title("Score")
    pl.barh(indices, score, .2, label="score", color='r')
    pl.barh(indices + .3, training_time, .2, label="training time", color='g')
    pl.barh(indices + .6, test_time, .2, label="test time", color='b')
    pl.yticks(())
    pl.legend(loc='best')
    pl.subplots_adjust(left=.25)
    pl.subplots_adjust(top=.95)
    pl.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        pl.text(-.3, i, c)

    pl.show()




class L1LinearSVC(LinearSVC):

    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(penalty="l1",
                                      dual=False, tol=1e-3)
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)


