# Patel, Hevin Dharmeshbhai
# 1002_036_919
# 2023_04_02
# Assignment_03_01

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models

def confusion_matrix(y_true, y_pred, n_classes=10):
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)
    n_test_samples = np.size(y_true, 0)
    #... for each sample
    for k in range(n_test_samples):
        expected = y_true[k]
        predicted = y_pred[k]
        confusion_matrix[expected, predicted] += 1
    return confusion_matrix 

    
def train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=1, batch_size=4):

  tf.keras.utils.set_random_seed(5368) # do not remove this line

  input_shape = X_train.shape[1:]

  # Neural Network Architecture
  model = models.Sequential()
  model.add(layers.Conv2D(8, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001), input_shape=input_shape))
  model.add(layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
  model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(layers.Flatten())
  model.add(layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
  model.add(layers.Dense(10, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
  model.add(layers.Activation('softmax'))

  # Training Neural Network
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

  # Predicting output for testing data
  predicted_y = model.predict(X_test)
  predicted_y_class = np.argmax(predicted_y,axis=1)

  # Confusion Matrix
  actual_y = np.argmax(Y_test, axis=1)
  cm = confusion_matrix(actual_y, predicted_y_class)
  plt.matshow(cm)
  plt.colorbar()
  plt.xlabel('Predicted label')
  plt.ylabel('True label')
  plt.savefig('confusion_matrix.png')

  # Save the model
  model.save('model.h5')

  # Return list of output parameters
  return [model, history, cm, predicted_y_class]