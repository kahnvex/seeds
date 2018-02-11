import numpy as np


class MeanEnsemble:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])

        return np.mean(predictions, axis=0)
