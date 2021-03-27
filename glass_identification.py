import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression


class BackProp:
    def __init__(self, hidden_layer_size, update_rule='standard', learning_rate=0.1, max_iter=1000):
        self.hidden_layer_size = hidden_layer_size
        self.update_rule = update_rule
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.labels = None
        self.W_1 = None
        self.b_1 = None
        self.W_2 = None
        self.b_2 = None
        self.fitted = False

    def fit(self, X, y):
        sample_size, input_layer_size = X.shape
        output_layer_size = y.shape[1]

        self.labels = np.array(list(set([tuple(label) for label in y])))
        self.fitted = True

        # Xavier Initialization
        self.W_1 = np.random.randn(input_layer_size, self.hidden_layer_size) / np.sqrt(input_layer_size)
        self.b_1 = np.zeros((1, self.hidden_layer_size))
        self.W_2 = np.random.randn(self.hidden_layer_size, output_layer_size) / np.sqrt(self.hidden_layer_size)
        self.b_2 = np.zeros((1, output_layer_size))

        if self.update_rule == 'standard':

            for _ in range(self.max_iter):
                for i in range(sample_size):
                    z_1 = expit(np.dot(np.c_[X[i].reshape(1, -1), -1], np.r_[self.W_1, self.b_1])).ravel()
                    z_2 = expit(np.dot(np.c_[z_1.reshape(1, -1), -1], np.r_[self.W_2, self.b_2])).ravel()

                    g = z_2 * (np.ones(output_layer_size) - z_2) * (y[i] - z_2)
                    e = z_1 * (np.ones(self.hidden_layer_size) - z_1) * np.dot(self.W_2, g)

                    self.W_2 += self.learning_rate * np.dot(z_1.reshape(-1, 1), g.reshape(1, -1))
                    self.b_2 -= self.learning_rate * g
                    self.W_1 += self.learning_rate * np.dot(X[i].reshape(-1, 1), e.reshape(1, -1))
                    self.b_1 -= self.learning_rate * e

        elif self.update_rule == 'cumulative':

            for _ in range(self.max_iter):
                z_1 = expit(np.dot(np.c_[X, -np.ones(sample_size).reshape(-1, 1)], np.r_[self.W_1, self.b_1]))
                z_2 = expit(np.dot(np.c_[z_1, -np.ones(sample_size).reshape(-1, 1)], np.r_[self.W_2, self.b_2]))

                for i in range(sample_size):
                    g = z_2[i] * (np.ones(output_layer_size) - z_2[i]) * (y[i] - z_2[i])
                    e = z_1[i] * (np.ones(self.hidden_layer_size) - z_1[i]) * np.dot(self.W_2, g)

                    self.W_2 += self.learning_rate * np.dot(z_1[i].reshape(-1, 1), g.reshape(1, -1))
                    self.b_2 -= self.learning_rate * g
                    self.W_1 += self.learning_rate * np.dot(X[i].reshape(-1, 1), e.reshape(1, -1))
                    self.b_1 -= self.learning_rate * e

        else:
            print('Error: invalid parameters')

    def predict(self, X):
        if self.fitted:
            test_sample_size = X.shape[0]

            z_1 = expit(np.dot(np.c_[X, -np.ones(test_sample_size).reshape(-1, 1)], np.r_[self.W_1, self.b_1]))
            z_2 = expit(np.dot(np.c_[z_1, -np.ones(test_sample_size).reshape(-1, 1)], np.r_[self.W_2, self.b_2]))

            for i in range(test_sample_size):
                losses = {}
                for label in self.labels:
                    losses[tuple(label)] = np.linalg.norm(z_2[i] - label)
                z_2[i] = sorted(zip(losses.values(), losses.keys()))[0][1]

            return z_2.astype(int)
        else:
            print('Error: unfitted model')

    def score(self, X, y):
        if self.fitted:
            correct_pred = 0
            y_pred = self.predict(X)
            for i in range(len(y)):
                if not any(y_pred[i] - y[i]):
                    correct_pred += 1
            return correct_pred / len(y)
        else:
            print('Error: unfitted model')


if __name__ == '__main__':
    data = pd.read_csv('data/Glass Identification.csv')
    X = scale(data.iloc[:, :-1].values)
    y = data.Type.values
    y_one_hot = pd.get_dummies(data.Type).values

    classifier = LogisticRegression()
    classifier.fit(X, y)
    score = classifier.score(X, y)
    print('Accuracy (logistic regression):', score)

    for hidden_layer_size in range(2, 6):
        classifier = BackProp(hidden_layer_size)
        classifier.fit(X, y_one_hot)
        score = classifier.score(X, y_one_hot)
        print('Accuracy (BP, hidden layer size = %d):' % hidden_layer_size, score)
