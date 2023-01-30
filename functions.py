import numpy as np
import h5py
import matplotlib.pyplot as plt

# %matplotlib inline
# plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

# %load_ext autoreload
# %autoreload 2


#   Sigmoid function
def sigmoid(Z):             #   Z -- numpy array of any shape
    A = 1/(1+np.exp(-Z)) 
    cache = Z              

    return A, cache

#   ReLu function
def relu(Z):
    A = np.maximum(0,Z)     #   A -- post-activation parameter
    assert(A.shape == Z.shape)
    cache = Z 

    return A, cache

#   ReLu backward propagation
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)    # just converting dz to a correct object
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)

    return dZ

#   Sigmoid backward propagation
def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    
    return dZ

#   FUNCTION: initialize_parameters
def initialize_parameters(n_x, n_h, n_y):   #   n_x -- size.input layer, n_h -- size.hidden layer, n_y -- size.output layer
    np.random.seed(1)                       #   set random numbers 
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {"W1": W1,                 #   creating dict for storing W and b
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

#   FUNCTION: initialize_parameters_deep
def initialize_parameters_deep(layer_dims):
    np.random.seed(1)                       #   set random numbers 
    parameters = {}
    L = len(layer_dims)                     # number of layers in the network
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

#   FUNCTION: linear_forward
def linear_forward(A, W, b):
    Z = np.dot(W,A) + b
    cache = (A, W, b)
    assert(Z.shape == (W.shape[0], A.shape[1]))

    return Z, cache

#   FUNCTION: linear_activation_forward
def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev,W,b)        #   FUNCTION: linear_forward
        A, activation_cache = sigmoid(Z)                    #   Sigmoid function

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev,W,b)        #   FUNCTION: linear_forward
        A, activation_cache = relu(Z)                       #   ReLu function

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

#   FUNCTION: L_model_forward
def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")        #   FUNCTION: linear_activation_forward
        caches.append(cache)

    AL, cache = linear_activation_forward(A,parameters['W' + str(L)], parameters['b' + str(L)], activation =  "sigmoid")              #   FUNCTION: linear_activation_forward
    caches.append(cache)
    assert(AL.shape == (1,X.shape[1]))

    return AL, caches

#   FUNCTION: compute_cost
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = np.sum(np.multiply(np.log(AL),Y) + np.multiply((1-Y),(np.log(1-AL))))/-m
    #another method:
    # cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    cost = np.squeeze(cost)                 #   e.g. this turns [[17]] into 17
    assert(cost.shape == ())

    return cost

#   FUNCTION: linear_backward
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ, axis=1,keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

#   FUNCTION: linear_activation_backward
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)                #   ReLu backward propagation
        dA_prev, dW, db = linear_backward(dZ, linear_cache)     #   FUNCTION: linear_backward
        
    elif activation == "sigmoid":
        dZ =  sigmoid_backward(dA, activation_cache)            #   Sigmoid backward propagation
        dA_prev, dW, db = linear_backward(dZ, linear_cache)     #   FUNCTION: linear_backward
    
    return dA_prev, dW, db

#   FUNCTION: L_model_backward
def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))       #   Initializing the backpropagation
    # Lth layer (SIGMOID -> LINEAR)
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")      #   FUNCTION: linear_activation_backward
    
    # Loop lth layer: (RELU -> LINEAR)
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")    #   FUNCTION: linear_activation_backward
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    
    return grads

#   FUNCTION: update_parameters
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW"+ str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db"+ str(l+1)]
    
    return parameters

#   load_data function
def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])    # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])    # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])       # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])       # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

#   FUNCTION: predict
def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2 
    p = np.zeros((1,m))
    probas, caches = L_model_forward(X, parameters)             #   FUNCTION: L_model_forward
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    print("Accuracy: "  + str(np.sum((p == y)/m)))

    return p

#   FUNCTION: L_layer_model
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    np.random.seed(1)
    costs = [] 
    parameters = initialize_parameters_deep(layers_dims)                #   FUNCTION: initialize_parameters_deep
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)                     #   FUNCTION: L_model_forward
        cost = compute_cost(AL, Y)                                      #   FUNCTION: compute_cost
        grads = L_model_backward(AL, Y, caches)                         #   FUNCTION: L_model_backward
        parameters = update_parameters(parameters, grads, learning_rate) #  FUNCTION: update_parameters
        
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs
