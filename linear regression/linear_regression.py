import numpy as np
from feature_engineering import *

import matplotlib.pyplot as plt

######################### UTILITIES LINEAR REGRESSION ########################
def least_squares(y, tx):
    """ The Least Squares algorithm (LS) 
        Args:
            y: shape=(N, ) (N number of events)
            tx: shape=(N, D) (D number of features)
        Returns:
            w: shape=(D, ) optimal weights
            mse: scalar(float) """
    w = np.linalg.solve(np.dot(tx.T,tx), np.dot(tx.T,y))
    mse = compute_mse(y, tx, w)
    
    return w, mse

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """ Generate a minibatch iterator for a dataset.
        Takes as input two iterables (here the output desired values 'y' and
        the input data 'tx')
        Outputs an iterator which gives mini-batches of `batch_size` matching 
        elements from `y` and `tx`. Data can be randomly shuffled to avoid 
        ordering in the original data messing with the randomness of the minibatches.
        Example of use :
            for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
                <DO-SOMETHING> """
    
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

    return

def build_k_indices(y, k_fold, seed):
    """ Build k indices for k-fold 
        Args:
            y: shape=(N,)
            k_fold: K in K-fold, i.e. the fold num
            seed: the random seed
        Returns:
            ret: shape=(k_fold, N/k_fold) with the data indices for each fold """
    num_row = y.shape[0]
    interval = int(num_row / k_fold) # Here it computes the number of intervals
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    
    return np.array(k_indices)

def compute_mse(y, tx, w):
    """ Calculate the mse for the vector e = y - tx.dot(w) """
    
    return 1 / 2 * np.mean((y-tx.dot(w)) ** 2)

def compute_mae(y, tx, w):
    """ Calculate the mae for vector e = y - tx.dot(w) """
    
    return np.mean(np.abs(y-tx.dot(w)))

def compute_loss(y, tx, w):
    """ Calculate the loss using either MSE or MAE 
        Args:
            y: shape=(N, )
            tx: shape=(N, D)
            w: shape=(2,). The vector of model parameters.
        Returns:
            loss: scalar(float), corresponding to the input parameters w """
    
    return compute_mse(y, tx, w)


def cross_validation_tx_LS(y, tx, k_indices, k):
    """ Return the loss of ridge regression for a fold corresponding to k_indices
        The regression matrix tx is given as an input 
        Args:
            y: shape=(N, ) (N number of events)
            tx: shape=(N, D) (D number of features)
            k_indices: 2D array returned by build_k_indices()
            k: scalar, the k-th fold
            lambda_: scalar, cf. ridge_regression()
        Returns:
            loss_tr: scalar(float), rmse = sqrt(2 mse) of the training set
            loss_te: scalar(float), rmse = sqrt(2 mse) of the testing set """

    # get k'th subgroup in test, others in train
    k_fold = len(k_indices)
    
    y_test = y[k_indices[k]]
    tx_test = tx[k_indices[k],:]
    
    ind_train = []
    ind_train = np.append(ind_train, k_indices[np.arange(k_fold)!=k])
    ind_train = [int(ind_train[i]) for i in range(len(ind_train))]
                 
    y_train = y[ind_train]
    tx_train = tx[ind_train,:]
    
    # ridge regression
    w_k = least_squares(y_train, tx_train)[0]
    
    # calculate the loss for train and test data
    loss_tr = np.sqrt(2*compute_mse(y_train, tx_train, w_k))
    loss_te = np.sqrt(2*compute_mse(y_test, tx_test, w_k))
    
    return loss_tr, loss_te, w_k

def cross_validation_demo_tx_LS(y, tx, k_fold):
    """ Cross validation over regularisation parameter lambda 
        The regression matrix tx is given as an input 
        Args:
            y: shape=(N, ) (N number of events)
            tx: shape=(N, D) (D number of features)
            degree: integer, degree of the polynomial expansion
            k_fold: integer, the number of folds
            lambdas: shape = (p, ) where p is the number of values of lambda to test
        Returns:
            best_lambda : scalar, value of the best lambda
            best_rmse : scalar, the associated root mean squared error for the best lambda """
    
    seed = 1
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    
    r_tr = []
    r_te = []
    ww = []
    ww = np.asarray(ww)
    
    for k in range(k_fold): # we do this to perform the training using all the data
        tr, te, w_k = cross_validation_tx_LS(y, tx, k_indices, k) 
        r_tr.append(tr)  
        r_te.append(te)
        if k==0:
            ww = w_k
        else:
            ww = np.c_[ww, w_k]
    
    rmse_tr = np.mean(r_tr)
    rmse_te = np.mean(r_te)
    w = np.mean(ww, axis=1)
    
    return rmse_tr, rmse_te, w


def LEAST_SQUARES_REGRESSION(dd_min, dd_max, F, M, k_fold):
    """ Lest Squares Regression, where the features taken into considertion
        are the clearance parameterf for the d time steps prior to the one
        we are predicting. 
        d is the delay parameter and it iterates in the for loop from dd_min to dd_max.
        Args:
            dd_min: scalar (minimum numer of the delay parameter)
            dd_max: scalar (maximum numer of the delay parameter)
            F     : shape(500, 2001) (Aerodynamic force)
            M     : shape(500, 2001) (Clearance matrix)
            k_fold: integer, the number of folds
            
        Returns:
            losses: shape(dd_max-dd_min+1, 2001) (Mean Squared Errors)
            losses_rel: shape(dd_max-dd_min+1, 2001) (Relative Errors) """
    
    N = M.shape[1]
    D = 2001
    f = int(N/D) # number of features associated to each time step
    losses = np.zeros((dd_max-dd_min+1, D))
    losses_rel = np.zeros((dd_max-dd_min+1, D))
    
    for d in range(dd_min, dd_max+1):
        
        F_tilde = np.zeros(F.shape)
        # loss = 0
        
        for j in range(D):
            
            if (j+1) - d < 0:
                tx = M[:, : f*(j+1)]
            else:
                tx = M[:, f*(j+1-d) : f*(j+1)]
            
            tx = np.c_[np.ones(tx.shape[0]), tx]
                
            y = F[:, j]
            # tx = standardize(tx)[0]
            rmse_tr, rmse_te, w = cross_validation_demo_tx_LS(y, tx, k_fold)
            prediction = np.dot(tx, w)
            F_tilde[:, j] = prediction
            curr_loss = compute_loss(y, tx, w)
            curr_loss_rel = np.mean(np.abs((prediction - F[:, j])/F[:, j]))
            # loss += curr_loss
            losses[d-dd_min, j] = curr_loss
            losses_rel[d-dd_min, j] = curr_loss_rel
            
        print(d)
        
    return losses, losses_rel

def AUTOREGRESSIVE_TRUE(dd_min, dd_max, X, Y, Fx, Fy, k_fold):
    """ Lest Squares Regression, where the features taken into considertion
        are the clearance parameters and the true forces for the d time steps 
        prior to the one we are predicting.
        d is the delay parameter and it iterates in the for loop from dd_min to dd_max
        Args:
            dd_min: scalar (minimum numer of the delay parameter)
            dd_max: scalar (maximum numer of the delay parameter)
            X     : shape(500, 2001) (X-coordinates)
            Y     : shape(500, 2001) (Y-coordinates)
            Fx    : shape(500, 2001) (Aerodynamic force along x-direction)
            Fy    : shape(500, 2001) (Aerodynamic force along y-direction)
            k_fold: integer, the number of folds
            
        Returns:
            forces_x: list(dd_max-dd_min+1, [500, 2001]) (Predicted forces along x)
            forces_y: list(dd_max-dd_min+1, [500, 2001]) (Predicted forces along y)
            losses_x: shape(dd_max-dd_min+1, 2001) (Mean Squared Errors along x)
            losses_y: shape(dd_max-dd_min+1, 2001) (Mean Squared Errors along y)
            losses_rel_x: shape(dd_max-dd_min+1, 2001) (Relative Errors along x)
            losses_rel_y: shape(dd_max-dd_min+1, 2001) (Relative Errors along y) """
    
    N = Fx.shape[0]
    D = 2001
    losses_x = np.zeros((dd_max-dd_min+1, D))
    losses_y = np.zeros((dd_max-dd_min+1, D))
    losses_rel_x = np.zeros((dd_max-dd_min+1, D))
    losses_rel_y = np.zeros((dd_max-dd_min+1, D))
    forces_x = []
    forces_y = []
    
    for d in range(dd_min, dd_max+1):
        
        F_tilde_x = np.zeros(Fx.shape)
        F_tilde_y = np.zeros(Fy.shape)
        
        for j in range(D):
            
            if j - d <= 0:
                tx_x = Fx[:, :j]
                tx_y = Fy[:, :j]
            else:
                tx_x = Fx[:, (j-d):j]
                tx_y = Fy[:, (j-d):j]
            
            if (d == 0) | (j == 0):
                tx = tx_x # or tx = tx_y tanto sono nulli
            else:
                tx = np.ravel([tx_x, tx_y], 'F').reshape(N, 2*min(j,d))
            
            tx = np.c_[tx, X[:,j], Y[:,j]]
            tx = np.c_[np.ones(tx.shape[0]), tx]
            
            yx = Fx[:, j]
            yy = Fy[:, j]
            
            # learning Fx
            rmse_tr_x, rmse_te_x, w_x = cross_validation_demo_tx_LS(yx, tx, k_fold)
            prediction_x = np.dot(tx, w_x)
            F_tilde_x[:, j] = prediction_x
            
            # learning Fy
            rmse_tr_y, rmse_te_y, w_y = cross_validation_demo_tx_LS(yy, tx, k_fold)
            prediction_y = np.dot(tx, w_y)
            F_tilde_y[:, j] = prediction_y
            
            curr_loss_x = compute_loss(yx, tx, w_x)
            curr_loss_y = compute_loss(yy, tx, w_y)
            curr_loss_rel_x = np.mean(np.abs((prediction_x - Fx[:, j])/Fx[:, j]))
            curr_loss_rel_y = np.mean(np.abs((prediction_y - Fy[:, j])/Fy[:, j]))
            
            losses_x[d-dd_min, j] = curr_loss_x
            losses_y[d-dd_min, j] = curr_loss_y
            losses_rel_x[d-dd_min, j] = curr_loss_rel_x
            losses_rel_y[d-dd_min, j] = curr_loss_rel_y
            
        forces_x.append(F_tilde_x)
        forces_y.append(F_tilde_y)
        
        print(d)
            
    return forces_x, forces_y, losses_x, losses_y, losses_rel_x, losses_rel_y

