from PIL import Image
import numpy as np

w, h = 28, 28
data = np.zeros((h, w), dtype=np.uint8)
data[14:27, 14] = 255 
img = Image.fromarray(data, 'L')
img = img.resize((256, 256), 0)
#img.save('my.png')
img.show()
