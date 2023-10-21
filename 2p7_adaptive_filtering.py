import matplotlib.pyplot as plt
import numpy as np
import utils
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
mu, sigma = np.mean(s_n), np.var(s_n)
x_noise = np.random.normal(mu, sigma, s_n.shape)
sf.write(root + 'problem2_7_noise.wav', x_noise, 8000, subtype='PCM_24')


# filter the noise signal through one of the filters
k = lpir.shape[0]
padded_x_noise = np.pad(x_noise[:, 0], (k,0), mode='constant', constant_values=(0,0))
X = sliding_window_view(padded_x_noise, k)
#print(f"{X.shape = }")
#print(f"{lpir.shape = }")
X @ lpir

#add filtered noise to speech signal




