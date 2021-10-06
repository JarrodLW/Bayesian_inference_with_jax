import numpy as np


class LinearRegressor():

    def __init__(self):
        self.m = None
        self.c = None

    def fit(self, X, y):
        z = np.polyfit(X, y, 1)
        self.m = z[0]
        self.c = z[1]

    def transform(self, Xsamples):
        out = self.m * Xsamples + self.c

        return out
