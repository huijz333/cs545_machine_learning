# load data from file example
import matplotlib.pyplot as plt
import numpy as np

#from pudb import set_trace; set_trace()

filename = 'test_file.csv'

# Part 1
'''
import csv

target = [0]*10000
data = []

#filename = 'sample_file.csv'
with open(filename, 'r') as csvfile:
        digits = csv.reader(csvfile, delimiter=',')
        for row in digits:
                target.append(int(row[0]))
                data.append(int(row[1:]))

#plt.plot(x, y, label='Loaded from file!')
'''
# Part 2
x = np.loadtxt(filename, delimiter=',', unpack=True)
print(x)
print(x[0,2])
#plt.plot(x, y, label='Loaded from file!')

#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Interesting Graph\nCheck it out!')
#plt.legend()
#plt.show()
