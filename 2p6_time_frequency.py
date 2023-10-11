# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:11:55 2023

@author: JUCA
"""

import librosa
import librosa.display
import IPython
import matplotlib.pyplot as plt
import sounddevice as sd # only needed for playing
import soundfile as sf # only needed if playing does not work
from lms import lms
from nlms import nlms
from rls import rls
from mse import mse

# Load audio files
#x, Fs = librosa.load('data/problem2_6.wav')
y, Fs = librosa.load('data/problem2_6.wav') #, sr=16000

# one of lms, nlms, rls
adaptive_algo = 'lms'

# Filter Length
L = 30

# parameters for lms 
mu_lms = 0.5  # step size

# parameters for nlms
mu_nlms = 0.2  # normalized step-size
delta = 1e-2  # regularization parameter

# parameters for rls
beta = 0.997  # forget factor
lambda_ = 1e2  # regularization

# Switch between adaptive algorithms
if adaptive_algo == 'lms':
    yhat, _ = lms(y, L, mu_lms)
elif adaptive_algo == 'nlms':
    yhat, _ = nlms(y, L, mu_nlms, delta)
elif adaptive_algo == 'rls':
    yhat, _ = rls(y, L, beta, lambda_)

# Plot the spectrograms of the signals
# noisy speech

spec_y = librosa.stft(y, n_fft=512, hop_length=32, center=True)
y_db = librosa.amplitude_to_db(abs(spec_y))
plt.figure(figsize=(14, 3))
plt.title("Synthesizer")
librosa.display.specshow(y_db, sr=Fs)
'''
# noise
spec_x = librosa.stft(x, n_fft=512, hop_length=32, center=True)
x_db = librosa.amplitude_to_db(abs(spec_x))
plt.figure(figsize=(14, 3))
plt.title("HIGHWAY NOISE")
librosa.display.specshow(x_db, sr=Fs)
'''