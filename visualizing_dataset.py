import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from keras import datasets, layers, models



mnist_dataset = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()
# train_labels: each label is an integer between 0 and 9

class_names_for_dataset = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images.shape
#print(len(train_labels)) is 60000
#print(len(test_labels)) is 10000

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

#Preprocessing
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# normalizing values to [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# check data is in correct format and ready to build and train model
plt.figure(figsize=(10, 10))
for image in range(25):
    plt.subplot(5, 5, image+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[image], cmap=plt.cm.binary)
    plt.xlabel(class_names_for_dataset[train_labels[image]])
plt.show()





