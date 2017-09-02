# import matplotlib.pyplot as plt
# import numpy as np
import time
import mnist
import perceptron


start = time.time()

images, labels = mnist.retrieve('training')
# ind = 2
# mnist.show_digit(images[ind], labels[ind])

stop = time.time()
print('time: {} seconds'.format(stop-start))
