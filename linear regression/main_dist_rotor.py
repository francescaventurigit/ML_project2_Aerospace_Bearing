import numpy as np
from loading_datasets import *
from feature_engineering import *
from linear_regression import *
import tqdm
import matplotlib

#%% loading data
X, Y, Fx, Fy = load()
N = X.shape[0]
D = Y.shape[1]

#%% setting parameters
dd_min = 100
dd_max = 100
M = distance_from_rotor(X, Y)
k_fold = 5

#%% Least Squares REGRESSION
losses_x, losses_rel_x = LEAST_SQUARES_REGRESSION(dd_min, dd_max, Fx, M, k_fold)
losses_y, losses_rel_y = LEAST_SQUARES_REGRESSION(dd_min, dd_max, Fy, M, k_fold)

#%% Compute MSE and Relative Error for d = 100
mean_Fx_time = np.mean(Fx, axis=0)[100:]
mean_Fy_time = np.mean(Fy, axis=0)[100:]

re = np.mean(losses_rel_x[100,100:] + losses_rel_y[100,100:])/2
mse = np.mean(losses_x[100,100:] + losses_y[100,100:])/2

#%% plot the error for d = 50
plt.figure(figsize=(25, 15))
time_steps = np.arange(2001)
plt.plot(time_steps, losses_x[200,:], color='b')#, marker='*')
# plt.ylim(0,0.01)
plt.plot(time_steps,
         losses_x[50,:],
#         marker="X",
#         markersize=20,
#         linestyle="None",
         color="blue",
         label="Loss F_x",
         linewidth=10,
        )

plt.plot(time_steps,
         losses_y[50,:],
#         marker="X",
#         markersize=20,
#         linestyle="None",
         color="red",
         label="F_y(t=50)",
         linewidth=10,
        )

plt.grid(True)
plt.legend(fontsize=25)
plt.xlabel("Delay Parameter d", fontsize=30)
plt.ylabel("Relative Error", fontsize=30)

plt.tick_params(axis='x', labelsize=25)
plt.tick_params(axis='y', labelsize=25)

#%%
errors = (np.mean(losses_x, axis=1) + np.mean(losses_y, axis=1))/2
d_it = np.arange(0, 350, 4)

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

plt.savefig("LR_iterations.png")
