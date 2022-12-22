import numpy as np

#%%
def standardize(dataset):
    """Standardize the dataset
       Args:
           train: shape=(N, D+2) (N number of events, D number of features)
       Returns:
           ret: shape=(N, D+2) """
               
    ret = dataset.copy()
    mean = np.mean(ret, axis=0)
    std = np.std(ret, axis=0)
    ret = (ret - mean)/std
    
    return ret, mean, std


#%%
def velocity(dataset):
    
    N = dataset.shape[0]
    D = dataset.shape[1]
    
    ret = np.zeros((N, D-1))
    for i in range(N):
        ret[i] = [dataset[i,j] - dataset[i,j-1] for j in range(1,D)]
        
    return ret
#%%
def acceleration(dataset):
    
    N = dataset.shape[0]
    D = dataset.shape[1]
    
    ret = np.zeros((N, D-1))
    for i in range(N):
        ret[i] = [(dataset[i,j] - dataset[i,j-1])**2 for j in range(1,D)]
        
    return ret

#%%
def velocity_module(X, Y):
    
    N = X.shape[0]
    D = X.shape[1]
    
    ret = np.zeros((N, D-1))
    for i in range(N):
        ret[i] = [np.sqrt((X[i,j]-X[i,j-1])**2 + (Y[i,j]-Y[i,j-1])**2) for j in range(1,D)]
    
    return ret

#%%
def acceleration_module(X, Y):
    
    N = X.shape[0]
    D = X.shape[1]
    
    ret = np.zeros((N, D-1))
    for i in range(N):
        ret[i] = [np.sqrt((X[i,j]-X[i,j-1])**2 + (Y[i,j]-Y[i,j-1])**2) for j in range(1,D)]
    
    return ret

#%%
def distance_from_rotor(X, Y):
    R = 0.5
    r = np.sqrt(X**2 + Y**2)
    
    return R - r
   