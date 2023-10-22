import matplotlib.pyplot as plt
import numpy as np
import utils
#from convmtx import convmtx
#from lms import lms
from numpy.lib.stride_tricks import sliding_window_view
import soundfile as sf # only needed if playing does not work
np.random.seed(42)

root = r'data/'

# load the speech signal
signal_file = root + 'problem2_7_speech.mat'
signal_mat = utils.loadmatlabfile(signal_file)
signal_mat.keys()
s_n = signal_mat['speech']
sf.write(root + 'problem2_7_speech.wav', s_n, 8000, subtype='PCM_24')

# load the filters
lpir_file = root + 'problem2_7_lpir.mat'
hpir_file = root + 'problem2_7_hpir.mat'
bpir_file = root + 'problem2_7_bpir.mat'

mat = utils.loadmatlabfile(lpir_file)
lpir = mat['lpir']

mat = utils.loadmatlabfile(hpir_file)
hpir = mat['hpir']

mat = utils.loadmatlabfile(bpir_file)
bpir = mat['bpir']

# make some noise
mean, variance = np.mean(s_n), np.var(s_n)
x_noise = np.random.normal(mean, variance, s_n.shape)
sf.write(root + 'problem2_7_noise.wav', x_noise, 8000, subtype='PCM_24')

# filter the noise signal through one of the filters
#k = lpir.shape[0]
#padded_x_noise = np.pad(x_noise[:, 0], (k,0), mode='constant', constant_values=(0,0))
#X = sliding_window_view(padded_x_noise, k)
#print(f"{X.shape = }")
#print(f"{lpir.shape = }")
#X @ lpir

#add filtered noise to speech signal
#signal_noise = s_n + X @ lpir

# use the LMS, NLMS and RLS algorithm to build an ANC system.
x = x_noise
#y = signal_noise
L = lpir.shape[0]
mu = 0.2 # step size
H = lpir
nsamp=x.shape[0]
delta = 1e-6

# LMS
w = np.zeros((L, nsamp+1))
e_lms = np.zeros(nsamp)

w_n = np.zeros((L))

for it in range(L, nsamp+1):
    x_n = np.flip(x[it - L: it]).T
    d = H.T @ x_n[0]
    e_n = d - w[:, it-1].T @ x_n[0]
    w_n = w_n + mu * e_n * x_n

    # store values for later plotting
    e_lms[it-1] = e_n
    w[:, it] = w_n

# NLMS
w = np.zeros((len(H), nsamp+1))
e_nlms = np.zeros(nsamp)

w_n = np.zeros((len(H)))

for it in range(len(H), nsamp+1):
    x_n = np.flip(x[it-len(H):it]).T
    d = H.T @ x_n[0]
    e_n = d - w[:,it-1].T @ x_n[0]
    w_n = w_n + (mu / (delta+x_n[0].T@x_n[0])) * x_n[0] * e_n

    # store values for later plotting
    e_nlms[it-1] = e_n
    w[:, it] = w_n



# plot the errors

plt.plot(range(len(e_lms)), e_lms, label='LMS')
plt.title('Errors')
plt.xlabel('n')
plt.ylabel('Error')
plt.legend()
plt.show()


plt.plot(range(len(e_lms)), e_nlms, label='NLMS')
plt.title('Errors')
plt.xlabel('n')
plt.ylabel('Error')
plt.legend()
plt.show()