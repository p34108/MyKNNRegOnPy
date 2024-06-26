import numpy as np
import pandas as pd


class MyKNNReg:
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.train_size = None
        self.size = None
        self.X, self.y = None, None
        self.metric = metric
        self.weight = weight

    def __str__(self):
        return self.train_size

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.train_size = X.shape
        self.size = X.shape[0]

    def predict(self, X):
        result = X.apply(self.calculating_the_distance, axis=1)
        return result

    def calculating_the_distance(self, value):
        data = {f'col{i}': val for i, val in enumerate(list(value))}
        X = pd.DataFrame(data, index=[0])
        X = pd.concat([X] * self.size, ignore_index=True)
        X.columns = list(self.X.columns)
        if self.metric == 'euclidean':
            evD = self.euclidean_distance(self.X, X)
        elif self.metric == 'chebyshev':
            evD = self.chebyshev_distance(self.X, X)
        elif self.metric == 'manhattan':
            evD = self.manhattan_distance(self.X, X)
        else:
            evD = self.cosine_distance(self.X, X)
        evD.columns = ['bD']
        evD['y'] = self.y.reset_index(drop=True)
        if self.weight == 'uniform':
            result = evD.sort_values(by='bD').reset_index(drop=True).iloc[:self.k]['y']
            return result.mean()
        elif self.weight == 'rank':
            weights_array = []
            test = evD.sort_values(by='bD').reset_index(drop=True).iloc[:self.k]
            test = test.reset_index()
            test['index'] = test['index'] + 1
            for i in range(test.shape[0]):
                numerator = np.vectorize(self.rank_or_distance_averaging)(test.iloc[i]['index'])
                denominator = (np.vectorize(self.rank_or_distance_averaging)(test['index'])).sum()
                Q_i = numerator / denominator
                weights_array.append(Q_i)
            return np.array(weights_array) @ test['y']
        else:
            weights_array = []
            test = evD.sort_values(by='bD').reset_index(drop=True).iloc[:self.k]
            for i in range(test.shape[0]):
                numerator = np.vectorize(self.rank_or_distance_averaging)(test.iloc[i]['bD'])
                denominator = (np.vectorize(self.rank_or_distance_averaging)(test['bD'])).sum()
                Q_i = numerator / denominator
                weights_array.append(Q_i)
            return np.array(weights_array) @ test['y']

    def rank_or_distance_averaging(self, stroka):
        stroka = (1 / stroka)
        return stroka

    def euclidean_distance(self, X_true, X):
        X_result = pd.DataFrame(np.sqrt(((X_true.reset_index(drop=True) - X) ** 2).sum(axis=1)))
        return X_result

    def chebyshev_distance(self, X_true, X):
        X_result = pd.DataFrame((X_true.reset_index(drop=True) - X).abs().max(axis=1))
        X_result = X_result.reset_index(drop=True)
        return X_result

    def manhattan_distance(self, X_true, X):
        X_result = pd.DataFrame((X_true.reset_index(drop=True) - X).abs().sum(axis=1))
        X_result = X_result.reset_index(drop=True)
        return X_result

    def cosine_distance(self, X_true, X):
        X_result = pd.DataFrame(1 - ((X_true.reset_index(drop=True) * X).sum(axis=1) / (
                np.sqrt((X ** 2).sum(axis=1)) * np.sqrt((X_true.reset_index(drop=True) ** 2).sum(axis=1)))))
        X_result = X_result.reset_index(drop=True)
        return X_result