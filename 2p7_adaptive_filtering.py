import matplotlib.pyplot as plt
import numpy as np
import utils
from numpy.lib.stride_tricks import sliding_window_view

root = r'data/'

# load the speech signal
signal_file = root + 'problem2_7_speech.mat'
signal_mat = utils.loadmatlabfile(signal_file)
signal_mat.keys()
s_n = signal_mat['speech']

# load the filters
lpir_file = root + 'problem2_7_lpir.mat'
hpir_file = root + 'problem2_7_hpir.mat'
bpir_file = root + 'problem2_7_bpir.mat'

mat = utils.loadmatlabfile(lpir_file)
print(mat.keys())
lpir = mat['lpir']

mat = utils.loadmatlabfile(hpir_file)
print(mat.keys())
hpir = mat['hpir']

mat = utils.loadmatlabfile(bpir_file)
print(mat.keys())
bpir = mat['bpir']


