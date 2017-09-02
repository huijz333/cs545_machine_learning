import numpy as np


class Perceptron(object):
    def __init__(self,
                 rand_init_weights='True'):
        self.w = None

    def __init_weights(self):
        num_weights = self.n_in * self.n_out
        w = np.random.uniform(-0.05, 0.05, num_weights)
        w = w.reshape(self.n_in, self.n_out)
        return w

    def __update_weights(self):
        return 0

    def fit(self, X, y, n_epochs, decent='stochastic'):
        if self.w == None:
            self.w = __init_weights(shape(X,y))  # Fix this!!!!
        return 0

    def test(self):
        return 0

#     def conf_matrix(self):
#         return 0
