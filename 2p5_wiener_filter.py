import matplotlib.pyplot as plt
import numpy as np
import utils
from numpy.lib.stride_tricks import sliding_window_view

root = r'data/'
output_root = 'outputs/'
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

def get_my_error(k):
    r_xx = utils.xcorr(x_n, x_n, k-1)
    r_sx = utils.xcorr(s_n, x_n, k-1)
    R_xx = sliding_window_view(r_xx[:, 0], k)
    w_hat = np.linalg.solve(R_xx, r_sx[k-1:])
    padded_x_n = np.pad(x_n[:, 0], (k-1,0), mode='constant', constant_values=(0,0))
    X = sliding_window_view(padded_x_n, k)
    s_hat = X@w_hat
    err = np.mean((s_hat - s_n)**2)
    return err

err_list = []
ks = range(1, 100)
for i in ks:
    err_list.append(get_my_error(i))

plt.plot(ks, err_list)
#plt.show()
plt.savefig(output_root + 'Problem_2_5_MSE.png')


