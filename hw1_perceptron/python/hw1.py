eta = 0.005

import matplotlib.pyplot as plt
import numpy as np
import time 
#from pudb import set_trace; set_trace()
from PIL import Image
from decimal import *

getcontext().prec = 3
start = time.clock()

filename = 'mnist_train.csv';
data = np.loadtxt(filename, delimiter=',', unpack=True)
print(data[0][0:10])

end = time.clock()
print('Processing time:', Decimal(end) - Decimal(start), 'seconds')
