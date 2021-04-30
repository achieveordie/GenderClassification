"""
I couldn't find a Ensemble classifier that fits my needs, so here's my own implementation.

Main motive for the following Classifier:
    1. Be able to contain pre-trained estimators.
    2. Be able to take different Input features per estimator during predicting/scoring/training.

I'm building this on top of sklearn.base and will only change few functionality to suit my needs.
"""

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import NotFittedError, UndefinedMetricWarning, DataDimensionalityWarning


class VotingClassifier(BaseEstimator, ClassifierMixin, clone):
    def __init__(self, clfs, voting='soft', weights=None,
                 use_clones=False, refit=False):
        self.clfs = clfs
        self.voting = voting
        self.weights = weights
        self.use_clones = use_clones
        self.refit = refit

        if self.voting not in ('soft', 'hard'):
            print(UndefinedMetricWarning("`voting` should be either soft or hard, using soft-voting."))
            self.voting = 'soft'

        if self.weights and len(self.weights) != len(self.clfs):
            raise ValueError("estimator weights and estimators don't match.")

    def fit(self, X, y, sample_weight=None):
        """

        :param X: <array-like>, shape:(n_estimators, n_samples, n_features).
            For each estimator, give (n_samples, n_features) to fit that estimator. Starts at 0 index.
            For the number in first dimension, find the corresponding classifier from `clfs`. This enables us to fit
            estimator with a unique n_features.
        :param y: <array-like>, shape: (n_samples,) or (n_estimators, n_samples).
            If the shape is (n_samples,) then it assumes that all estimators are being fed the same amount of samples.
            For different n_samples per estimator, shape will match the latter.
        :param sample_weight: <array-like>, shape: (n_samples,).
            They are passed to the individual estimators via the `sample_weight` argument.
        :return:
        """
        if not self.refit:
            print("`refit` set to False, not refitting the base estimators")
            return

        if len(X.shape) != 3 or len(y.shape) not in (1, 2):
            raise DataDimensionalityWarning("Shape of X should be (n_estimators, n_samples, n_features)"
                                            "Shape of Y should either be (n_estimators, n_samples) or (n_samples,)")




