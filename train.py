from sklearn.linear_model import LogisticRegression

def train_model():
    X = [[1], [2], [3], [4], [5], [6]]
    y = [0, 0, 0, 1, 1, 1]

    model = LogisticRegression()
    model.fit(X, y)

    return model

