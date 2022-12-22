import numpy as np

def load():
    """ Load training dataset and convert 'b' into 0 and 's' into 1
        Remove the 24th categorical feature
        Return the dataset and the removed feature """
   
    X = np.genfromtxt("X.csv", delimiter=",")
    Y = np.genfromtxt("Y.csv", delimiter=",")
    
    
    Fx = np.genfromtxt("FX.csv", delimiter=",")
    Fy = np.genfromtxt("FY.csv", delimiter=",")
    
    return X.T, Y.T, Fx.T, Fy.T

def load_new():
    """ Load training dataset and convert 'b' into 0 and 's' into 1
        Remove the 24th categorical feature
        Return the dataset and the removed feature """
   
    X_new = np.genfromtxt("X_new.csv", delimiter=",")
    Y_new = np.genfromtxt("Y_new.csv", delimiter=",")
    
    
    Fx_new = np.genfromtxt("FX_new.csv", delimiter=",")
    Fy_new = np.genfromtxt("FY_new.csv", delimiter=",")
    
    return X_new.T, Y_new.T, Fx_new.T, Fy_new.T

