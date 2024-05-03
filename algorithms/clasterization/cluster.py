import numpy as np
from sklearn.cluster import KMeans

class Cluster:
    def __init__(self, method=KMeans):
        self.model = method()

    def calculate(self, hsi, dist=False):
        rows, cols, deep = hsi.shape
        hsi_table = hsi.reshape(-1, deep)
        results = self.model.fit_predict(hsi_table).reshape(rows, cols)

        return results if not dist else (results, self.model.transform(hsi_table))