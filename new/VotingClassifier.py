"""
I couldn't find a Ensemble classifier that fits my needs, so here's my own implementation.

Main motive for the following Classifier:
    1. Be able to contain pre-trained estimators.
    2. Be able to take different Input features per estimator during predicting/scoring/training.

I'm building this on top of sklearn.base and will only change few functionality to suit my needs.

Everywhere, `X` should be a list of (numpy-array/csr-matrix) where each element is of shape (n_samples, n_features)
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError, UndefinedMetricWarning
import numpy as np


class VotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clfs, voting='soft', weights=None,
                 refit=False):
        self.clfs = clfs
        self.voting = voting
        self.weights = weights
        self.refit = refit

        if self.voting not in ('soft', 'hard'):
            print(UndefinedMetricWarning("`voting` should be either soft or hard, using soft-voting."))
            self.voting = 'soft'

        if self.weights and len(self.weights) != len(self.clfs):
            raise ValueError("estimator weights and estimators don't match.")

    @staticmethod
    def _is_fitted(clf, attributes='n_classes_'):
        """
        Since we're exclusively writing codes for classifiers, if `classes_` attributed is undefined
        then that implies the classifier has not be fitted yet.
        Can pass different single argument.
        """
        return hasattr(clf, attributes)

    @staticmethod
    def _check_for_samples(x):
        """"
        Ensures that n_samples stays same across entire dimensions.
        Assumes that `x` is a list of array-like elements where first dim is n_samples.
        """
        n_samples = x[0].shape[0]
        for single_data in x:
            if single_data.shape[0] != n_samples:
                raise ValueError("n_samples should be same across all data.")

    def _check_for_fitted(self):
        """
        Internal method to check if we need to throw `NotFittedError`.
        """
        all_fitted = True
        for clf in self.clfs:
            if not self._is_fitted(clf):
                all_fitted = False
                break

        if self.refit or all_fitted:
            raise NotFittedError

    def fit(self, X, y, sample_weight=None):
        """

        :param X: <array-like>, shape:(n_estimators, n_samples, n_features).
            For each estimator, give (n_samples, n_features) to fit that estimator. Starts at 0 index.
            For the number in first dimension, find the corresponding classifier from `clfs`. This enables us to fit
            estimator with a unique n_features.

            This MUST be a list of np arrays/sparse matrices.

        :param y: <array-like>, shape: (n_samples,) or (n_estimators, n_samples).
            If the shape is (n_samples,) then it assumes that all estimators are being fed the same amount of samples.
            For different n_samples per estimator, shape will match the latter.

            This MUST either be a np-array/sparse-matrix or a list of np-array/sparse-matrix.

        :param sample_weight: <array-like>, shape: (n_samples,).
            They are passed to the individual estimators via the `sample_weight` argument.
        :return: None
        """
        if not self.refit:
            print("`refit` set to False, not refitting the base estimators")
            return

        if len(X) != 3 or len(y) not in (1, 2):
            raise ValueError("Shape of X should be (n_estimators, n_samples, n_features)"
                             "Shape of Y should either be (n_estimators, n_samples) or (n_samples,)")
        if not isinstance(y, list):  # make shape (n_estimators, n_samples)
            y = [y for _ in range(len(X))]

        self._check_for_fitted()
        self._check_for_samples(X)
        self._check_for_samples(y)

        [[clf.fit(x, y, sample_weight) for (x, y) in zip(X, y)] for clf in self.clfs]

    def predict(self, X):
        """
        Given X, predict the outputs.
        :param X: <array-like>, shape:(n_estimators, n_samples, n_features), List<np-array/csr-matrix>
        :return: predictions: <np.array>, shape(n_samples, )
        """

        self._check_for_fitted()
        self._check_for_samples(X)

        if len(X) != len(self.clfs):
            ValueError("`X` should have shape as: (n_estimators, n_samples, n_features), first dimension didn't match")

        predictions = []
        if self.voting == 'soft':
            [predictions.append(clf.predict_proba(x)) for (clf, x) in zip(self.clfs, X)]
            predictions = np.sum(np.array(predictions), axis=0)
            return np.argmax(predictions, axis=1)
        elif self.voting == 'hard':
            [predictions.append(clf.predict(x)) for (clf, x) in zip(self.clfs, X)]
            predictions = np.array(predictions)
            predictions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),
                                              axis=0, arr=predictions)
            return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities.
        :param X: <array-like>, shape:(n_estimators, n_samples, n_features), List<np-array/csr-matrix>
        :return: predictions: np.array, shape=(n_samples, )
        """

        self._check_for_fitted()
        self._check_for_samples(X)

        if len(X) != len(self.clfs):
            ValueError("`X` should have shape as: (n_estimators, n_samples, n_features), first dimension didn't match")

        predictions = []
        [predictions.append(clf.predict(x)) for (clf, x) in zip(self.clfs, X)]

        return np.average(np.array(predictions), axis=0)

    def score(self, X, y, **kwargs):
        """
        Return mean accuracy, given data and labels.
        :param X: same as `fit` method.
        :param y: same as `fit` method.
        :param kwargs: added to match the signature.
        :return: scores: float, a value between 0.0 and 1.0.
        """

        if len(X) != 3 or len(y) not in (1, 2):
            raise ValueError("Shape of X should be (n_estimators, n_samples, n_features)"
                             "Shape of Y should either be (n_estimators, n_samples) or (n_samples,)")
        if not isinstance(y, list):  # make shape (n_estimators, n_samples)
            y = [y for _ in range(len(X))]

        self._check_for_fitted()
        self._check_for_samples(X)
        self._check_for_samples(y)

        scores = []
        [scores.append(clf.score(x, y, sample_weight=self.weights)) for (clf, x, y) in zip(self.clfs, X, y)]
        return np.average(np.array(scores))
