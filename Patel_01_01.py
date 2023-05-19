# Patel, Hevin Dharmeshbhai
# 1002_036_919
# 2023_02_27
# Assignment_01_01
import numpy as np 
#calculate mean square error for training data and predicted data
def Mean_Square_Error_M1(Y_pred, Y_true):
  return np.mean((Y_pred - Y_true)**2)
#calculate sigmoid
def Sigmoid_M2(x):
  return 1 / (1 + np.exp(-x))
#adding bias layer to testing and training data
def add_bias(X):
  return np.concatenate((np.ones((1, X.shape[1])), X))
#calculate mean square error 
def Mean_Square_Error_M1_Activation(WT, TrainS_N, TrainY_N, layers):
  Activation_NO = []
  for i in range(len(layers)):
    if len(Activation_NO) == 0:
      Activation_NO.append(TrainS_N)
      X = np.dot(WT[i], TrainS_N)
      Activation_NO.append(Sigmoid_M2(X))
    else:
      X = np.dot(WT[i], add_bias(Activation_NO[-1]))
      Activation_NO.append(Sigmoid_M2(X))
  return Mean_Square_Error_M1(Activation_NO[-1], TrainY_N)

def multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h=0.00001,seed=2):
    WT = [] # LIst containing the weight matrix of each layer.
    Mean_Square_Error_M1_Per_Epoch = [] # List containing the Mean_Square_Error_M1 per epoch.
    for i in range (len(layers)):
      np.random.seed(seed)
      if i == 0:
        w= np.random.randn(layers[i],X_train.shape[0]+1)
        WT.append(w)
      else:
        w=np.random.randn(layers[i],layers[i-1]+1)
        WT.append(w)
    #adding bias to training and testing data    
    TrainX_N = add_bias(X_train)
    TestX_N = add_bias(X_test)
    Temporary_W = []
    for i in WT:
      a = np.copy(i)
      Temporary_W.append(a)
    #training multilayer neural network
    for i in range(epochs):
      for i in range(X_train.shape[1]):
        TrainS_N = TrainX_N[:,i:i+1]
        TrainY_N = Y_train[:,i:i+1]
        for i, ele1 in enumerate(WT):
          for j, ele2 in enumerate(ele1):
            for k, w in enumerate(ele2):
              #adding step size and calculating MSE
              WT[i][j][k] = WT[i][j][k] + h
              a = Mean_Square_Error_M1_Activation(WT, TrainS_N, TrainY_N, layers)
              WT[i][j][k] = WT[i][j][k] - h
              #subtracting step size and calculating MSE
              WT[i][j][k] = WT[i][j][k] - h
              b = Mean_Square_Error_M1_Activation(WT, TrainS_N, TrainY_N, layers)
              WT[i][j][k] = WT[i][j][k] + h
              #calculating derivatives
              dMean_Square_Error_M1_W = (a - b)/(2*h)
              #calculating new weights
              Wnew = WT[i][j][k] - alpha * dMean_Square_Error_M1_W
              Temporary_W[i][j][k] = Wnew
              
        WT=[]
        for i in Temporary_W:
          wt_layer= np.copy(i)
          WT.append(wt_layer)
      #calculating mean square error for testing data
      Mean_Square_Error_M1_List = []
      for i in range(X_test.shape[1]):
        X_test_sample = TestX_N[:,i:i+1]
        Y_test_sample = Y_test[:,i:i+1]
        Mean_Square_Error_M1_List.append(Mean_Square_Error_M1_Activation(Temporary_W, X_test_sample, Y_test_sample, layers))
      Mean_Square_Error_M1_Per_Epoch.append(np.mean(Mean_Square_Error_M1_List))
      
    #calculating final output
    Final_result_lyr = []
    for i in range(len(layers)):
      if len(Final_result_lyr) == 0:
        Y = np.dot(Temporary_W[i], TestX_N)
        Final_result_lyr.append(Sigmoid_M2(Y))
      else:
        Y = np.dot(Temporary_W[i], add_bias(Final_result_lyr[-1]))
        Final_result_lyr.append(Sigmoid_M2(Y))
    #output = np.array(output).reshape(Y_test.shape)
    WT = Temporary_W
    return [WT, Mean_Square_Error_M1_Per_Epoch, Final_result_lyr[-1]]