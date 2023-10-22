import matplotlib.pyplot as plt
import numpy as np
import utils
from numpy.lib.stride_tricks import sliding_window_view
import soundfile as sf # only needed if playing does not work
np.random.seed(42)

root = r'data/'
output_root = 'outputs/'

# load the speech signal
signal_file = root + 'problem2_7_speech.mat'
signal_mat = utils.loadmatlabfile(signal_file)
signal_mat.keys()
s_n = signal_mat['speech']
sf.write(root + 'problem2_7_speech.wav', s_n, 8000, subtype='PCM_24')

# load the filter
lpir_file = root + 'problem2_7_lpir.mat'

mat = utils.loadmatlabfile(lpir_file)
lpir = mat['lpir']

# make some noise
mean, variance = np.mean(s_n), np.var(s_n)
x_noise = np.random.normal(mean, variance*10, s_n.shape)
sf.write(output_root + 'problem2_7_noise.wav', x_noise, 8000, subtype='PCM_24')

# filter the noise signal through one of the filters
L = lpir.shape[0]
padded_x_noise = np.pad(x_noise[:, 0], (L-1,0), mode='constant', constant_values=(0,0))
X = sliding_window_view(padded_x_noise, L)


filtered_noise = X@lpir
filtered_noise.shape
sf.write(output_root + 'problem2_7_noise_lpir.wav', filtered_noise, 8000, subtype='PCM_24')

#add filtered noise to speech signal
signal_with_noise = s_n + filtered_noise
sf.write(output_root + 'problem2_7_signal_w_noise_lpir.wav', signal_with_noise, 8000, subtype='PCM_24')


# use the LMS, NLMS and RLS algorithm to build an ANC system.
x = x_noise
H = lpir
nsamp=x.shape[0]
delta = 1e-6

# LMS
w = np.zeros((L, nsamp))
e_lms = np.zeros(nsamp)
w_n = np.zeros(L)
d_hat = np.zeros(nsamp)
mu = 0.4  # step size

for it in range(nsamp):
    x_n = np.flip(X[it,:])
    d = signal_with_noise[it]
    d_hat[it] = w_n.T @ x_n
    e_n = d - d_hat[it]
    w_n = w_n + mu * e_n * x_n
    e_lms[it] = e_n
    w[:, it] = w_n

filtered_output_LMS = e_lms[:,None]
filtered_output_MSE_LMS = np.mean(filtered_output_LMS**2)

print(f"{filtered_output_MSE_LMS = }")
sf.write(output_root + 'problem2_7_filtered_output_LMS.wav', filtered_output_LMS, 8000, subtype='PCM_24')


# NLMS
w = np.zeros((L, nsamp))
e_nlms = np.zeros(nsamp)
d_hat = np.zeros(nsamp)
w_n = np.zeros(L)
mu = 0.4  # step size

for it in range(L, nsamp):
    x_n = np.flip(X[it,:])
    d = signal_with_noise[it]
    d_hat[it] = w_n.T @ x_n

    e_n = d - d_hat[it]
    w_n = w_n + (mu / (delta+x_n.T@x_n)) * x_n * e_n

    # store values for later plotting
    e_nlms[it] = e_n
    w[:, it] = w_n

filtered_output_NLMS = e_nlms[:,None]
filtered_output_MSE_NLMS = np.mean(filtered_output_NLMS**2)

print(f"{filtered_output_MSE_NLMS = }")
sf.write(output_root + 'problem2_7_filtered_output_NLMS.wav', filtered_output_NLMS, 8000, subtype='PCM_24')

# RLS
lambda_ = 1e-5
beta = 0.999

# start RLS
# initialize
w = np.zeros((L, nsamp))
e_rls = np.zeros(nsamp)
P = 1/lambda_*np.eye(L)
d_hat = np.zeros(nsamp)
w_n = np.zeros((L,1))
w_n.shape

# for each n do
for it in range(nsamp):
    # get x_n
    x_n = X[it,:]
    x_n = np.expand_dims(x_n, -1)
    d = signal_with_noise[it]
    # get filter output
    d_hat[it] = w_n.T @ x_n

    # update iteration
    e_rls_n = d - d_hat[it]
    denum = beta + x_n.T @ P @ x_n
    K_n = (P @ x_n) / denum
    w_n = w_n + K_n * e_rls_n
    P = (P - (K_n @ x_n.T) @ P)/beta
    e_rls[it] = e_rls_n
    w[:, it] = w_n[:,0]


filtered_output_RLS = e_rls[:,None]
filtered_output_MSE_RLS = np.mean(filtered_output_RLS**2)

print(f"{filtered_output_MSE_RLS = }")

# Output sound files
sf.write(f'{output_root}problem2_7_speech_filtered_RLS.wav', filtered_output_RLS, 8000, subtype='PCM_24')

# Find optimal values for beta and lambda
def get_my_RLS_error(beta, lambda_):
    # RLS
    # start RLS
    # initialize
    w = np.zeros((L, nsamp))
    e_rls = np.zeros(nsamp)
    P = 1 / lambda_ * np.eye(L)
    d_hat = np.zeros(nsamp)
    w_n = np.zeros((L, 1))
    w_n.shape

    # for each n do
    for it in range(nsamp):
        # get x_n
        x_n = X[it, :]
        x_n = np.expand_dims(x_n, -1)
        d = signal_with_noise[it]
        # get filter output
        d_hat[it] = w_n.T @ x_n

        # update iteration
        e_rls_n = d - d_hat[it]
        denum = beta + x_n.T @ P @ x_n
        K_n = (P @ x_n) / denum
        w_n = w_n + K_n * e_rls_n
        P = (P - (K_n @ x_n.T) @ P) / beta
        e_rls[it] = e_rls_n
        w[:, it] = w_n[:, 0]

    filtered_output_RLS = e_rls[:, None]
    filtered_output_MSE_RLS = np.mean(filtered_output_RLS ** 2)
    return filtered_output_MSE_RLS


lambda_ = 1e-2
beta = 0.99

myerrors=[]
for lambda_ in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3, 1e4]:
    for beta in [0.90, 0.95, 0.99, 0.999]:
        my_error = get_my_RLS_error(beta, lambda_)
        myerrors.append(my_error)
        print(f"Parameters {lambda_ =:.0e}, {beta = :2.5f}: error: {my_error:2.6f}")

myerrors_array = np.asarray(myerrors)


np.argmin(myerrors_array)

