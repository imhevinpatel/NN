# Patel, Hevin Dharmeshbhai
# 1002_036_919
# 2023_04_16
# Assignment_04_01
import tensorflow as tf
import numpy as np
import keras


class CNN(object):


    def __init__(self):
        #builds the neural network model's linear stack of layers.
        self.model = keras.models.Sequential()
        self.metric = []
        
        #self.loss and self.optimizer are initialized to None
        self.loss = None
        self.optimizer = None

    def add_input_layer(self, shape=(2,), name=""):
        #creates an input layer with a specified input_shape and name.
        
        added_layer = keras.layers.InputLayer(input_shape=shape, name=name)
        self.model.add(added_layer)
        return None

    def append_dense_layer(self, num_nodes, activation="relu", name="", trainable=True):
        #creates a dense layer with a specified number of num_nodes, activation function, name, and trainable parameter.
       
        added_layer = keras.layers.Dense(num_nodes, activation=activation, name=name, trainable=trainable)
        self.model.add(added_layer)
        return None

    def append_conv2d_layer(self, num_of_filters, kernel_size=3, padding='same', strides=1,
                            activation="relu", name="", trainable=True):
        #create a convolutional layer using keras.layers.Conv2D()
        
        layer = keras.layers.Conv2D(filters=num_of_filters, kernel_size=kernel_size, padding=padding, strides=strides,
                                    activation=activation, name=name, trainable=trainable)
        self.model.add(layer)
        return layer

    def append_maxpooling2d_layer(self, pool_size=2, padding="same", strides=2, name=""):
        #create a max pooling layer using keras.layers.MaxPooling2D().#arguments meanings are same as previous explaination.
        
        layer = keras.layers.MaxPooling2D(pool_size=pool_size, padding=padding, strides=strides, name=name)
        self.model.add(layer)
        return layer

    def append_flatten_layer(self, name=""):
        #adds a flattening layer to a Keras-powered neural network.
        
        layer = keras.layers.Flatten(name=name)
        self.model.add(layer)
        return layer

    def set_training_flag(self, layer_numbers=[], layer_names="", trainable_flag=True):
        #method for setting the trainable flag
        
        if type(layer_numbers) == list:
            self.model.get_layer(layer_number=layer_numbers, layer_name=layer_names).trainable = trainable_flag
        else:
            for l in layer_numbers:
                self.model.get_layer(layer_number=l, layer_name=layer_names[l]).trainable = trainable_flag
        return None
        

    def get_weights_without_biases(self, layer_number=None, layer_name=""):
        #retrieving the weight matrix (without biases) for a specific layer in a Keras-implemented neural network.
        
        lyer_number = 0
       
        #If any weights are present in the provided layer, the method should first check for them. The procedure must return None if the layer is weightless.
        
        if layer_number == None:
            if len(self.model.get_layer(name=layer_name).get_weights()) <= 0:
                return None
            else:
                return self.model.get_layer(name=layer_name).get_weights()[0]
        else:
            if len(self.model.layers[layer_number - 1].get_weights()) <= 0 or layer_number == 0:
                return None
            elif layer_number == (lyer_number-1):
                return self.model.layers[layer_number].get_weights()[0]
            else:
                return self.model.layers[layer_number - 1].get_weights()[0]

    def get_biases(self, layer_number=None, layer_name=""):
        #function is used to retrieve the biases for a specified layer in a Keras-implemented neural network

        if layer_number == None:
            if len(self.model.get_layer(name=layer_name).get_weights()) <= 0:
                return None
            else:
                return self.model.get_layer(name=layer_name).get_weights()[1]

        else:
            if len(self.model.get_layer(index=layer_number - 1).get_weights()) <= 0 or layer_number == 0:
                return None
            elif layer_number == -1:
                return self.model.layers[layer_number].get_weights()[1]
            else:
                return self.model.layers[layer_number - 1].get_weights()[1]

    def set_weights_without_biases(self, weights, layer_number=None, layer_name=""):
        #The function set_weights_without_biases accepts a weight matrix (without biases) as input and, if necessary, a layer number or name to identify the layer whose weights should be adjusted.
        if layer_number == 0:
            return None

        elif layer_number == None:
            keras.backend.set_value(self.model.get_layer(name=layer_name).weights[0], weights)

        elif layer_number != None:
            keras.backend.set_value(self.model.get_layer(index=layer_number - 1).weights[0], weights)


    def set_biases(self, biases, layer_number=None, layer_name=""):
        #function sets biases for each layer
        if layer_number != None:
            keras.backend.set_value(self.model.get_layer(index=layer_number - 1).weights[1], biases)

        elif layer_number == None :
            keras.backend.set_value(self.model.get_layer(name=layer_name).weights[1], biases)

    def remove_last_layer(self):
        #function removes the last layer from the model and returns the removed layer
        ll = self.model.pop()
        self.model = keras.Sequential(self.model.layers)
        return ll

    def load_a_model(self, model_name="", model_file_name=""):
        #initializes a new Keras Sequential model and adds layers from an existing model, which can be either VGG16, VGG19 or a custom model loaded from a file.   
        self.model = keras.models.Sequential()

        if model_name == "VGG16":
            base_model = keras.applications.vgg16.VGG16()
        elif model_name == "VGG19":
            base_model = keras.applications.vgg19.VGG19()
        else:
            base_model = keras.models.load_model(model_file_name)

        for each_layer in base_model.layers:
            self.model.add(each_layer)

    def save_model(self, model_file_name=""):
        self.model.save(model_file_name)

    def set_loss_function(self, loss="SparseCategoricalCrossentropy"):
        self.loss = loss

    def set_metric(self, metric):
        self.metric = metric

    def set_optimizer(self, optimizer="SGD", learning_rate=0.01, momentum=0.0):
        #optimizer is set to "SGD" then the method sets the optimizer to keras.optimizers.SGD with the given learning_rate and momentum parameters.
        
        if optimizer.lower() == "sgd":
            self.optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        
        #If optimizer is set to "RMSprop", the method sets the optimizer to keras.optimizers.RMSprop with the given learning_rate parameter
        
        elif optimizer.lower() == "rmsprop":
            self.optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        
        # If optimizer is set to "Adagrad", the method sets the optimizer to keras.optimizers.Adagrad with the given learning_rate parameter
        elif optimizer.lower() == "adagrad":
            self.optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)
        
        #If optimizer is set to "Adagrad", the method sets the optimizer to keras.optimizers.Adagrad with the given learning_rate parameter
        else:
            pass

    def predict(self, X):
        X = X.astype('float32')
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.evaluate(x=X, y=y)

    def train(self, X_train, y_train, batch_size, num_epochs):
        #the method compiles the model using the specified optimizer, loss function, and metric.
        
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metric)

        p = self.model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=num_epochs, verbose=2, shuffle=True)

        return p.history['loss']
