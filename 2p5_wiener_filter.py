import matplotlib.pyplot as plt
import numpy as np
import utils

root = r'C:/msys64/git_openai/git/machine_learning_for_signal_processing/data/'

# load the signal
signal_file = root + 'problem2_5_signal.mat'
signal_mat = utils.loadmatlabfile(signal_file)
signal_mat.keys()
s_n = signal_mat['signal']

# load the noise
noise_file = root + 'problem2_5_noise.mat'
noise_mat = utils.loadmatlabfile(noise_file)
noise_mat.keys()
w_n = noise_mat['noise']

# estimate the correlation functions
# proudly stolen from ex3.2.8.py
x_n = s_n + w_n

plt.plot(x_n) 

r_xx = utils.xcorr(x_n, x_n, 3)
r_dx = utils.xcorr(x_n, x_n, 3) 

R_xx = np.vstack((r_xx[2:5], r_xx[1:4], r_xx[0:3]))

# the line below does not run
#w_hat = np.linalg.solve(R_xx, r_dx[2:5])

