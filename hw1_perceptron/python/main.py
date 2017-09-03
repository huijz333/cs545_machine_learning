# import matplotlib.pyplot as plt
# import numpy as np
import time
import mnist
import perceptron


start = time.time()
p = perceptron.Perceptron(alpha=0.0001,
                          eta0=1.0,
                          n_iter=10,
                          grad_descent='stochastic',
                          batch_size=1,
                          max_iter=None,
                          tol=None,
                          shuffle=False,
                          rand_state=True)

images, labels = mnist.retrieve('training')
p.fit(images, labels)

# mnist.show_digit(images[80], label='not 0')

stop = time.time()
print('time: {} seconds'.format(stop-start))
