import argparse

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.datasets import cifar10
import pickle
import scipy
import numpy as np

# Load the data set
#(X, Y), (X_test, Y_test) = cifar10.load_data()
#X,Y =shuffle(X,Y)
#Y = to_categorical(Y,10)
#Y_test=to_categorical(Y_test,10)


#parser = argparse.ArgumentParser(description='Decide if an image is a picture of a bird')
#parser.add_argument('image', type=str, help='The image image file to check')
#args = parser.parse_args()



# Make sure the data is normalized
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)


