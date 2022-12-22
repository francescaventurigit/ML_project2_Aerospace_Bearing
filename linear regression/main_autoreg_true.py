import numpy as np
from loading_datasets import *
from feature_engineering import *
from linear_regression import *
import tqdm
import scipy

#%% loading data
X, Y, Fx, Fy = load()
N = X.shape[0]
D = Y.shape[1]

#%% setting parameters
dd_min = 0
dd_max = 200
F = Fx
M = distance_from_rotor(X, Y)
k_fold = 5

#%% LS AUTOREGRESSIVE
forces_x, forces_y, losses_x, losses_y, losses_rel_x, losses_rel_y = AUTOREGRESSIVE_TRUE(dd_min, dd_max, X, Y, Fx, Fy, k_fold)

#%% Compute MSE and Relative Error for d = 100
re = np.mean(losses_rel_x[100,100:] + losses_rel_y[100,100:])/2
mse = np.mean(losses_x[100,100:] + losses_y[100,100:])/2

#%%
# fx0 = forces_x[0] 
# fy0 = forces_y[0]

# fx23 = forces_x[23] 
# fy23 = forces_y[23]

# fx37 = forces_x[37] 
# fy37 = forces_y[37]

# fx75 = forces_x[75] 
# fy75 = forces_y[75]

# fx100 = forces_x[100] 
# fy100 = forces_y[100]

# fx125 = forces_x[125] 
# fy125 = forces_y[125]

# fx150 = forces_x[150] 
# fy150 = forces_y[150]

# fx175 = forces_x[175] 
# fy175 = forces_y[175]

# force_x_to_plot = [fx0, fx23, fx37, fx75, fx100, fx125, fx150, fx175]
# force_y_to_plot = [fy0, fy23, fy37, fy75, fy100, fy125, fy150, fy175]
#%%
errors = (np.mean(losses_x, axis=1) + np.mean(losses_y, axis=1))/2
d_it = np.arange(0, 150, 4)

plt.figure(figsize=(15, 15))
plt.semilogy(d_it,
         errors[d_it],
         marker="o",
         markersize=20,
#         linestyle="None",
#         color="blue",
#         label="Predicted",
         linewidth=8,
        )

plt.semilogy(100,
         errors[d_it][25],
         marker="o",
         markersize=20,
#         linestyle="None",
         color="red",
#         label="Predicted",
         linewidth=8,
        )


plt.grid(True)
# plt.legend(fontsize=25)
plt.xlabel("d", fontsize=30)
plt.ylabel("LR Loss", fontsize=30)

plt.tick_params(axis='x', labelsize=25)
plt.tick_params(axis='y', labelsize=25)

plt.savefig("LR_iterations_true.png")

#%% plot the error for d = 200
time_steps = np.arange(2001)
plt.plot(time_steps, losses_x[150,:], color='b')#, marker='*')
plt.ylim(0, 0.01)

#%%
# file_path_x = 'fx0.mat'
# scipy.io.savemat(file_path_x, {'fx0': fx1})
# file_path_y = 'fy0.mat'
# scipy.io.savemat(file_path_y, {'fy0': fy1})

# file_path_x = 'fx23.mat'
# scipy.io.savemat(file_path_x, {'fx23': fx23})
# file_path_y = 'fy23.mat'
# scipy.io.savemat(file_path_y, {'fy23': fy23})

# file_path_x = 'fx37.mat'
# scipy.io.savemat(file_path_x, {'fx37': fx37})
# file_path_y = 'fy37.mat'
# scipy.io.savemat(file_path_y, {'fy37': fy37})

# file_path_x = 'fx75.mat'
# scipy.io.savemat(file_path_x, {'fx75': fx75})
# file_path_y = 'fy75.mat'
# scipy.io.savemat(file_path_y, {'fy75': fy75})

# file_path_x = 'fx100.mat'
# scipy.io.savemat(file_path_x, {'fx100': fx100})
# file_path_y = 'fy100.mat'
# scipy.io.savemat(file_path_y, {'fy100': fy100})

# file_path_x = 'fx125.mat'
# scipy.io.savemat(file_path_x, {'fx125': fx125})
# file_path_y = 'fy125.mat'
# scipy.io.savemat(file_path_y, {'fy125': fy125})

# file_path_x = 'fx150.mat'
# scipy.io.savemat(file_path_x, {'fx150': fx150})
# file_path_y = 'fy150.mat'
# scipy.io.savemat(file_path_y, {'fy150': fy150})

# file_path_x = 'fx175.mat'
# scipy.io.savemat(file_path_x, {'fx175': fx175})
# file_path_y = 'fy175.mat'
# scipy.io.savemat(file_path_y, {'fy175': fy175})

#%%
# plt.plot(Fx[100,:], Fy[100,:], label="true")
# traj_to_plot = [1, 25, 50, 75, 100, 150]
# for i in range(len(force_x_to_plot)):
#     plt.plot(force_x_to_plot[i][100,:], force_y_to_plot[i][100,:])

