import matplotlib.pyplot as plt
import numpy as np
import utils

root = r'data/'

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
plt.show()
r_xx = utils.xcorr(x_n, x_n, 2)
r_sx = utils.xcorr(s_n, x_n, 2)

R_xx = np.hstack((r_xx[2:5], r_xx[1:4], r_xx[0:3]))

# the line below does not run
#w_hat = np.linalg.solve(R_xx, r_dx[2:5])

