import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data() # using keras to load the Fashion MNIST dataset

# Normalize the training and testing images 
train_images = train_images / 255.0
test_images = test_images / 255.0

# The last 12000 samples of the training data will be used as a validation set
validation_images = train_images[-12000:]
train_images = train_images[:-12000]

validation_labels = train_labels[-12000:]
train_labels = train_labels[:-12000]

"""
Creating the Convolutional Neural Network model with the following conditions:
1. 2D Convolutional layer, 28 filters, 3x3 window size, ReLU activation
2. 2x2 max pooling
3. 2D Convolutional, 56 filters, 3x3 window size, ReLU activation
4. Fully-connected layer, 56 nodes, ReLU activation - this is done using Dense(...)
5. Fully-connected layer, 10 nodes, softmax activation 
"""

cnn_model = Sequential([
    Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(56, (3, 3), activation='relu'),
    Flatten(), # this is used to convert all the 2D arrays from pooled feature maps into a single long continuous linear vector. The flattened matrix is an input to the fully connected layer to classify the image.
    Dense(56, activation='relu'),
    Dense(10, activation='softmax')
])

# Compiling the Convolutional Neural Network model to check for format errors, define the loss function, define the optimizer and the metrics
cnn_model.compile(
    optimizer=Adam(),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

print("The number of trainable parameters in this model are: ", cnn_model.count_params()) # The number of trainable parameters in this model is 394530

# Fitting the model 
history = cnn_model.fit(
    train_images[..., np.newaxis], # adds a new dimension to the input images so instead of (28, 28), it is (28, 28, 1)
    train_labels,
    epochs=10, # specifying the number of epochs
    batch_size=32, # specifying the batch size 
    validation_data=(validation_images[..., np.newaxis], validation_labels) # specifying the validation data for the model
)

# plotting the training and validation accuracy as line plots on the same set of axes
plt.figure(figsize=(10, 10))
plt.plot(history.history['accuracy'], label='Training Accuracy of Model', linestyle='-')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy of Model', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy over Epochs')
plt.grid(True)
plt.show()

# Evaluate the accuracy on the test set 
test_set_loss, test_set_accuracy = cnn_model.evaluate(test_images[..., np.newaxis], test_labels, verbose=0)

print("The test accuracy of this model is: ", test_set_accuracy)

# Checking where the model misclassifies 
predicted_labels = cnn_model.predict(test_images[..., np.newaxis]).argmax(axis=1)
misclassified_indices = np.where(predicted_labels != test_labels)[0]

# Showing examples of where the model misclassifies
for class_label in range(10):
    misclassified_class_indices = misclassified_indices[test_labels[misclassified_indices] == class_label]
    if len(misclassified_class_indices) > 0:
        misclassified_eg_index = misclassified_class_indices[0]
        misclassified_eg = test_images[misclassified_eg_index]
        plt.figure()
        plt.imshow(misclassified_eg, cmap='gray')
        plt.title(f'Actual Label: {test_labels[misclassified_eg_index]}, Predicted Label: {predicted_labels[misclassified_eg_index]}')
        plt.show()
