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

layers_dims = [12288, 20, 7, 5, 1] #  4-layered model
# Set size as per your model's best fit. Here Number of units in ith layer is layer_dims[i] 

#Nown defining the Model
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    costs = []                                 # keep track of cost
    m=X.shape[1]
    
    
    parameters = initialize_parameters_deep(layers_dims)
    
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        
        AL, caches = L_model_forward(X,parameters)  
        
        
        # Compute cost.
        
        cost = 1/m*(np.sum(-Y*np.log(AL)-(1-Y)*np.log(1-AL),axis=1))
        
        
        # Backward propagation.
        
        grads = L_model_backward(AL,Y,caches) 
        
 
        # Update parameters.
        
        parameters = update_parameters(parameters,grads,learning_rate)
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
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
parameters = L_layer_model(train_x, train_y, layers_dims , num_iterations = 2500, print_cost=True)

#Check Accuracy over training as well as test datasets
pred_train_L_layered = predict(train_x, train_y, parameters)
pred_test_L_layered = predict(test_x, test_y, parameters)

