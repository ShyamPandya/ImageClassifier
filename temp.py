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

# Define our network architecture:

# Input is a 32x32 image with 3 color channels (red, green and blue)
network=input_data(shape=[None,32,32,3],data_preprocessing=img_prep,data_augmentation=img_aug)

#Step 1: Convolution
network=conv_2d(network,32,3,activation='relu')

#Step 2: Max Pooling
network=max_pool_2d(network,2)

# Step 3: Convolution again
network = conv_2d(network, 64, 3, activation='relu')

# Step 4: Convolution yet again
network = conv_2d(network, 64, 3, activation='relu')

# Step 5: Max pooling again
network = max_pool_2d(network, 2)

# Step 6: Fully-connected 512 node neural network
network = fully_connected(network, 512, activation='relu')

# Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.5)

# Step 8: Fully-connected neural network with two outputs (0=isn't a bird, 1=is a bird) to make the final prediction
network = fully_connected(network, 10, activation='softmax')

# Tell tflearn how we want to train the network
network=regression(network,optimizer='adam',loss='categorical_crossentropy',learning_rate=0.001)

#Wrap the network in a model object
model=tflearn.DNN(network,tensorboard_verbose=0, checkpoint_path='bird-classifier.tfl.ckpt')


# Train it! We'll do 100 training passes and monitor it as it goes.
#model.fit(X,Y,n_epoch=50,shuffle=True,validation_set=(X_test, Y_test),show_metric=True, batch_size=96,
#         snapshot_epoch=True,
#run_id='bird-classifier')

# Save model when training is complete to a file
#model.save("bird-classifier.tfl")
#print("Network trained and saved as bird-classifier.tfl!")
model.load('bird-classifier.tfl')


# Load the image file
img1 = scipy.ndimage.imread("plane.jpeg", mode="RGB")

# Scale it to 32x32
img1 = scipy.misc.imresize(img1, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')

# Predict
prediction = model.predict([img1])
print(prediction)
#is_bird1=prediction[0][0]
# Check the result.
is_bird1 = np.argmax(prediction[0][2])

if is_bird1:
    print("1)That's a bird!")
else:
    print("1)That's not a bird!")


