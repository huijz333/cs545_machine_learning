# Find a more elegant solution to this

# To time a block of code, bookend it with the following
from decimal import *
import time
getcontext().prec = 3
start = time.clock()




end = time.clock()
print(4)
print("Processing time:", Decimal(end) - Decimal(start), "seconds")
