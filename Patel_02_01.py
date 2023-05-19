# Patel, Hevin Dharmeshbhai
# 1002_036_919
# 2023_03_19
# Assignment_02_01
import numpy as np
import tensorflow as tf
#converting trainingdata into batchsize
def generate_batches(X, y, batch_size=32):
    for i in range(0, X.shape[0], batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]
    # if there's any data left, yield it
    if X.shape[0] % batch_size != 0:
        yield X[-(X.shape[0] % batch_size):], y[-(X.shape[0] % batch_size):]

def multi_layer_nn_tensorflow(X_train,Y_train,layers,activations,alpha,batch_size,epochs=1,loss="svm",validation_split=[0.8,1.0],weights=None,seed=2):
  epoch_error_lst = []
  weight_matrix_list = []
  #weights initialization
  if weights is None:
    for i in range(len(layers)):
      np.random.seed(seed)
      #weightmatrices for first layer
      if i==0:
        w = tf.Variable(np.random.randn(X_train.shape[1]+1, layers[i]), dtype = np.float32)
        weight_matrix_list.append(w)
      #weight matrices for other layer
      else:
        w = tf.Variable(np.random.randn(layers[i-1]+1, layers[i]), dtype = np.float32)
        weight_matrix_list.append(w)
  #consider weights from function parameter      
  else:
    weight_matrix_list = [tf.Variable(w, dtype = np.float32) for w in weights]
  #converting training data into tensor
  X_train = tf.convert_to_tensor(X_train,dtype=tf.float32)
  #splitting data into training and validation set
  start = int(validation_split[0] * X_train.shape[0])
  end = int(validation_split[1] * X_train.shape[0])
  train_x, train_y, val_x, val_y = np.concatenate((X_train[:start], X_train[end:])), np.concatenate((Y_train[:start], Y_train[end:])), X_train[start:end], Y_train[start:end]
  train_x = tf.concat([tf.ones((tf.shape(train_x)[0], 1), dtype=train_x.dtype), train_x], axis=1)
  val_x = tf.concat([tf.ones((tf.shape(val_x)[0], 1), dtype=val_x.dtype), val_x], axis=1)
  
  #train the network
  def train_network(X_train, Y_train, layers, activations, loss, weights):
    a = [X_train]
    for i in range(len(layers)):
        #matrix multiplication for xtarin and weights
        z = tf.matmul(a[-1], weights[i])
        #passing the output into linear activation function
        if activations[i].lower() == "linear":
          activated_output = z
        #passing the output into sigmoid activation function
        if activations[i].lower() == "sigmoid":
          activated_output = tf.nn.sigmoid(z)
        #passing the output into relu activation function
        if activations[i].lower() == "relu":
          activated_output = tf.nn.relu(z)
        a.append(activated_output)
        #add bias layer to the output if it is not the last layer
        if i != (len(layers)-1):
          activated_output = tf.concat([tf.ones((tf.shape(activated_output)[0], 1), dtype=activated_output.dtype), activated_output], axis=1)
          a.append(activated_output)
    network_output = a[-1]
    # Calculate SVM Loss
    if loss.lower() == "svm":
      error = tf.reduce_mean(tf.maximum(0.0, 1.0 - Y_train * network_output))
    # Calculate MSE Loss
    if loss.lower() == "mse":
      error = tf.reduce_mean(tf.square(network_output - Y_train))
    # Calculate cross_entropy Loss
    if loss.lower() == "cross_entropy":
      error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network_output, labels=Y_train))

    return [network_output, error]
  #training the multilayer nueral network
  for each_epoch in range(epochs):
    for train_x_batch, train_y_batch in generate_batches(train_x, train_y, batch_size):
      with tf.GradientTape() as tape:
        tape.watch(weight_matrix_list)
        #calculating loss for each loss
        batch_loss = train_network(train_x_batch, train_y_batch, layers, activations, loss, weight_matrix_list)[1]
        #calculate derivative of error with respect to weight
      gradient = tape.gradient(batch_loss,weight_matrix_list)
      #update weights for each layer 
      for i in range(len(weight_matrix_list)):
        weight_matrix_list[i] = weight_matrix_list[i]-(alpha * gradient[i])
        #calculating validation error using updated weight
    validation_error = train_network(val_x, val_y, layers, activations, loss, weight_matrix_list)[1]
    epoch_error_lst.append(validation_error)
  #final output of the multilayer neural network
  neural_network_output = train_network(val_x, val_y, layers, activations, loss, weight_matrix_list)[0]
  return [weight_matrix_list, epoch_error_lst, neural_network_output]