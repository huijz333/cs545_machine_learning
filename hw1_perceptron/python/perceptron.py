import numpy as np


class Perceptron(object):
    def __init__(self,
                 alpha=0.0001,
                 eta0=1.0,
                 n_iter=10,
                 grad_descent='stochastic',
                 batch_size=None,
                 max_iter=None,
                 tol=None,
                 shuffle=False,
                 rand_state=True):

        self.alpha = alpha
        self.eta0 = eta0
        self.n_iter = n_iter
        self.grad_descent = grad_descent
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.rand_state = rand_state
        self.w = None
        return None

    def fit(self, X, y):
        if not self.w:  # Initialize the weights they haven't been already
            self.w = self.init_weights(X.shape)

        if self.grad_descent == 'stochastic':
            self.__fit_stochastic(X, y)
        elif self.grad_descent == 'batch':
            self.__fit_batch(X, y)
        else:  # True gradient descent
            self.__fit_tgd(X, y)
        return None

    def init_weights(self, input_shape):
        rows, cols = input_shape[1], input_shape[2]
        if self.rand_state:
            w = np.random.uniform(-0.05, 0.05, rows * cols + 1)
        else:
            w = np.zeros(rows * cols + 1)
        return w

    def __fit_stochastic(self, X, y):
        samples = X.shape[0]
        ind = np.arange(samples)
        for _ in range(self.n_iter):
            if self.shuffle:    # shuffle samples
                ind = np.random.shuffle(ind)
            for i in ind:
                output = self.w[0] + np.dot(X[i].flatten(), self.w[1:])
                if output > 0:      # Postive prediction
                    if y[i] != 0:   # False positive
                        self.w[0] -= self.eta0
                        self.w[1:] -= self.eta0 * X[i].flatten()
                else:               # Negative prediction
                    if y[i] == 0:   # Miss detection
                        self.w[0] += self.eta0
                        self.w[1:] += self.eta0 * X[i].flatten()

        for i in range(100):
            output = self.w[0] + np.dot(X[i].flatten(), self.w[1:])
            if output > 0:      # Postive prediction
                print('Positive prediction:', end=' ')
                if y[i] != 0:   # False positive
                    print('INCORRECT')
                else:
                    print('correct')
            else:               # Negative prediction
                print('Negative prediction:', end=' ')
                if y[i] == 0:   # Miss detection
                    print('INCORRECT')
                else:
                    print('correct')

        return None

    # def __fit_batch(self, X, y):
    #     return None

    # def __fit_tgd(self, X, y):
    #     return None

    def __update_weights(self):
        return None

    def test(self):
        return None

#     def confusion_matrix(self):
#         return None
