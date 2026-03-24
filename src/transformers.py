from sklearn.base import BaseEstimator, TransformerMixin
# Custom transformer to handle correlation filtering
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        corr_matrix = np.corrcoef(X, rowvar=False)
        upper = np.triu(np.abs(corr_matrix), k=1)  # upper triangle only
        self.drop_cols_ = [
            i for i in range(upper.shape[1])
            if any(upper[:, i] > self.threshold)    # drop if correlated with any earlier col
        ]
        return self

    def transform(self, X, y=None):
        return np.delete(X, self.drop_cols_, axis=1)

    def get_support(self):
        # for recovering gene names later
        return self.drop_cols_
