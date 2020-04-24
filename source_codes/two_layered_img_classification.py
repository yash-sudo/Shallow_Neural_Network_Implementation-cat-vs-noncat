#Importing the librarries/Modules
import numpy as np
import matplotlib.pyplot as plt
import h5py
from helper_functions  import *  #Importing all the helper functions defined earlier in helper_functions file
#This includes initialize_parameters_deep(),L_model_backward(),L_model_forward(),update_parameters
#initialize_parameters(),linear_activation_backward(),linear_activation_forward() & predict()

"""
Fetching the data.
Here the data is in a h5 file. This h5 file has Columns of Name "train_set_x","train_set_y","test_set_x","test_set_y" & "classes"
 """
def load_data():
    """
    Function returning training and test datasets along with the classes.
    """
    train_dataset = h5py.File('train_catvnoncat.h5', "r")  # Change the directory as per your system
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r") # Change the directory as per your system
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
#Function definition to extract data from h5 completed 

# Calling the function and unpacking the returned datasets
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

#Uncomment to Visualize The data
"""

plt.imshow(train_x_orig[109])   # A cat picture in training dataset at 109th index
plt.show()

"""

#Preprocessing the Image data

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

#Done preprocesing

#Now heading towards Model
 
#Setting the Hyperparameters
n_x = 12288     # num_px * num_px * 3
n_h = 7                     #No of hidden units in 1st hidden layer
n_y = 1             #No of nodes in output layer.
layers_dims = (n_x, n_h, n_y)




def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    # Initialize parameters dictionary, by calling initialize_parameters() functions from the helper_functions module.
    
    parameters = initialize_parameters(n_x,n_h,n_y) 
    
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W, b, Activation_name". Output: "A, cache".
        
        A1, cache1 = linear_activation_forward(X,W1,b1,"relu")
        A2, cache2 = linear_activation_forward(A1,W2,b2,"sigmoid")
 
        
        # Compute cost
        cost = 1/m*(np.sum(-Y*np.log(A2)-(1-Y)*np.log(1-A2),axis=1))
        
        # Initializing backward propagation
        dA2 = -(np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Backward propagation. Inputs: "dA, cache, Activation_Name". Outputs: "dA, dW, db; also dA0 (not used)".
        dA1, dW2, db2 =linear_activation_backward(dA2,cache2,"sigmoid")  

        dA0, dW1, db1 = linear_activation_backward(dA1,cache1,"relu")
        
        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        
        # Update parameters.
        
        parameters = update_parameters(parameters,grads,learning_rate)
        

        # Retrieve W1, b1, W2, b2 from parameters(updated)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
       
    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

#Now Training the Model
#set the hyperparameters
parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)

#Check Accuracy over training as well as test datasets
predictions_train_two_layered = predict(train_x, train_y, parameters)
predictions_test_two_layered = predict(test_x, test_y, parameters)

"""
MODEL completed with a accuracy of 99.521% over training dataset 
&
an accuracy of ~70% over test datasets
"""