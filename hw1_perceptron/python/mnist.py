import numpy as np
import struct
import gzip
import urllib.request
from pathlib import Path


def retrieve(samp_type='training'):
    '''
        Retrieves data from current directory or downloads them

        Parameters: TYPE: string - 'training' or 'testing'

        Returns: TYPE: numpy.ndarray - normalized 28x28 greyscale images
                 TYPE: numpy.ndarray - lables
    '''
    if samp_type.lower() == 'training':
        image_file = 'train-images-idx3-ubyte.gz'
        label_file = 'train-labels-idx1-ubyte.gz'
    else:  # Sample type is 'testing'
        image_file = 't10k-images-idx3-ubyte.gz'
        label_file = 't10k-labels-idx1-ubyte.gz'

    image_path = Path(image_file)
    label_path = Path(label_file)
    if not (image_path.is_file() and label_path.is_file()):
        # Files do not exist in directory. We need to download the files.
        download_file(label_file)
        download_file(image_file)
    return load_data(label_file, image_file)


def download_file(zip_file_name):
    '''
        Downloads image and label files if the do not already exist

        Parameters: TYPE: string - file name to be downloaded

        Returns: None
    '''
    url = 'http://yann.lecun.com/exdb/mnist/{}'.format(zip_file_name)
    print('Downloading {}'.format(zip_file_name))
    f_zip, h = urllib.request.urlretrieve(url, zip_file_name)


def load_data(label_file, image_file):
    '''
        Loads data from .gz file into numpy format

        Parameters: TYPE: string - image file name
                    TYPE: string - label file name

        Returns: TYPE: numpy.ndarray - label data
                 TYPE: numpy.ndarray - image data
    '''
    with gzip.open(label_file, 'rb') as flbl:
        byte = flbl.read(8)
        _, size = struct.unpack(">II", byte)
        buf = flbl.read(size)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    with gzip.open(image_file, 'rb') as fimg:
        byte = fimg.read(16)
        _, size, rows, cols = struct.unpack(">IIII", byte)
        buf = fimg.read(rows * cols * size)
        images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        images = np.divide(images, np.max(images))
        images = images.reshape(size, rows, cols)
    return labels, images
