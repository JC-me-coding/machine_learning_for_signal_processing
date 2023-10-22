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
k = lpir.shape[0]
padded_x_noise = np.pad(x_noise[:, 0], (k-1,0), mode='constant', constant_values=(0,0))
X = sliding_window_view(padded_x_noise, k)

filtered_noise = X@lpir
filtered_noise.shape
sf.write(output_root + 'problem2_7_noise_lpir.wav', filtered_noise, 8000, subtype='PCM_24')

#add filtered noise to speech signal
signal_with_noise = s_n + filtered_noise
sf.write(output_root + 'problem2_7_signal_w_noise_lpir.wav', signal_with_noise, 8000, subtype='PCM_24')


# use the LMS, NLMS and RLS algorithm to build an ANC system.
x = x_noise
y = signal_with_noise
L = k
mu = 0.2 # step size
H = lpir
nsamp=x.shape[0]
delta = 1e-6

# LMS
w = np.zeros((L, nsamp+1))
e_lms = np.zeros(nsamp)
w_n = np.zeros(L)
d_hat = np.zeros(nsamp)

for it in range(L, nsamp):
    x_n = X[it,:]
    #print(f"{x_n.shape = }")
    d = signal_with_noise[it]
    #print(f"{d.shape = }")
    #print(f"{w_n.shape = }")
    d_hat[it] = w_n.T @ x_n
    e_n = d - d_hat[it]
    #print(f"{e_n.shape = }")
    w_n = w_n + mu * e_n * x_n
    e_lms[it] = e_n
    w[:, it] = w_n

filtered_output_LMS = e_lms[:,None]
filtered_output_MSE_LMS = np.mean(filtered_output_LMS**2)

print(f"{filtered_output_MSE_LMS = }")
sf.write(output_root + 'problem2_7_filtered_output.wav', filtered_output_LMS, 8000, subtype='PCM_24')



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

filtered_output_NLMS = e_nlms[:,None] - ds
filtered_output_MSE_NLMS = np.mean(filtered_output_NLMS**2)

print(f"{filtered_output_MSE_NLMS = }")

# RLS
yhat = np.zeros(len(y))
e_rls = np.zeros(len(y))
lambda_ = 1e2
beta=0.95

padded_x_noise = np.pad(x_noise[:, 0], (L-1,0), mode='constant', constant_values=(0,0))
X = sliding_window_view(padded_x_noise, L)

# start RLS
# initialize
w = np.zeros(L)  # theta in the book
w = np.expand_dims(w, -1)
P = 1/lambda_*np.eye(L)

# for each n do
for n in range(len(y)):
    # get x_n
    x_n = X[n,:]
    x_n = np.expand_dims(x_n, -1)

    # get filter output
    yhat[n] = w.T@x_n

    # update iteration
    e_rls[n] = y[n] - yhat[n]
    denum = beta + x_n.T@P@x_n
    K_n = (P@x_n)/denum
    w = w + K_n*e_rls[n]
    P = (P - (K_n @ x_n.T) @ P)/beta


filtered_output_RLS = e_rls[:,None] - ds
filtered_output_MSE_RLS = np.mean(filtered_output_RLS**2)

print(f"{filtered_output_MSE_RLS = }")

# Output sound files
sf.write(f'{output_root}problem2_7_speech_noised_lpir.wav', ds, 8000, subtype='PCM_24')

sf.write(f'{output_root}problem2_7_speech_filtered_RLS.wav', filtered_output_RLS, 8000, subtype='PCM_24')


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

plt.plot(range(len(e_rls)), e_rls, label='RLS')
plt.title('Errors')
plt.xlabel('n')
plt.ylabel('Error')
plt.legend()
plt.show()
