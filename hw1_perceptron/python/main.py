# import matplotlib.pyplot as plt
# import numpy as np
import time
import load_mnist


start = time.time()

images, labels = load_mnist.retrieve('training')
print(images.shape)
print(labels.shape)

stop = time.time()
print('time: {} seconds'.format(stop-start))
