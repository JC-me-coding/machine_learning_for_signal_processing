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
import numpy as np
import itertools
# Load audio files
#x, Fs = librosa.load('data/problem2_6.wav')
y, Fs = librosa.load('data/problem2_6.wav') #, sr=16000

# Plot the spectrograms of the signals
# noisy speech
output_folder = 'outputs/'

def plot_tf_analysis(y, nfft, hop_length, window_fct='hann', tofile=False, folder=None):
    spec_y = librosa.stft(y, n_fft=nfft, hop_length=hop_length, center=True, window=window_fct)
    y_db = librosa.amplitude_to_db(abs(spec_y))
    plt.figure(figsize=(14, 3))
    plt.title(f"Synthesizer, {hop_length = }, {nfft = }, {window_fct = }")
    librosa.display.specshow(y_db, sr=Fs)
    if not tofile:
        plt.show()
    else:
        filename = f"{folder}2p6_time_frequency_nfft{nfft}_hoplength{hop_length}_windowfct_{window_fct}.png"
        plt.savefig(filename)
    plt.close()

tofile = True
plot_tf_analysis(y, nfft=32, hop_length=128, window_fct='hann', tofile=tofile, folder=output_folder)
plot_tf_analysis(y, nfft=32, hop_length=32, window_fct='hann', tofile=tofile, folder=output_folder)
plot_tf_analysis(y, nfft=1014, hop_length=128, window_fct='hann', tofile=tofile, folder=output_folder)
plot_tf_analysis(y, nfft=512, hop_length=1, window_fct='hann', tofile=tofile, folder=output_folder)
plot_tf_analysis(y, nfft=512, hop_length=512, window_fct='hann', tofile=tofile, folder=output_folder)
plot_tf_analysis(y, nfft=128, hop_length=1, window_fct='hann', tofile=tofile, folder=output_folder)
plot_tf_analysis(y, nfft=256, hop_length=1, window_fct='hann', tofile=tofile, folder=output_folder) #BEST
plot_tf_analysis(y, nfft=256, hop_length=1, window_fct='triang', tofile=tofile, folder=output_folder)
plot_tf_analysis(y, nfft=256, hop_length=1, window_fct='boxcar', tofile=tofile, folder=output_folder)

spec_y = librosa.stft(y, n_fft=nfft, hop_length=hop_length, center=True, window=window_fct)
y_db = librosa.amplitude_to_db(abs(spec_y))
plt.figure(figsize=(14, 3))
plt.title(f"Synthesizer, {hop_length = }, {nfft = }, {window_fct = }")
librosa.display.specshow(y_db, sr=Fs)

plt.show()
#filename = f"{folder}2p6_time_frequency_nfft{nfft}_hoplength{hop_length}_windowfct_{window_fct}.png"
#plt.savefig(filename)
plt.close()




# Loop through all combinations:
windowfcts = ['hann', 'boxcar', 'triang']
nffts = 2**np.asarray(range(5,11))
hop_lengths = 2**np.asarray(range(5,11))

combinations = list(itertools.product(windowfcts, nffts, hop_lengths))
print("number of combinations: ", len(combinations))
for i, combination in enumerate(combinations):
    windowfct = combination[0]
    nfft = combination[1]
    hop_length = combination[2]
    print(f"creating plot number {i+1} with , {hop_length = }, {nfft = }, {windowfct = }")
    plot_tf_analysis(y, nfft=nfft, hop_length=hop_length, window_fct=windowfct, tofile=True, folder=output_folder)



'''
# noise
spec_x = librosa.stft(x, n_fft=512, hop_length=32, center=True)
x_db = librosa.amplitude_to_db(abs(spec_x))
plt.figure(figsize=(14, 3))
plt.title("HIGHWAY NOISE")
librosa.display.specshow(x_db, sr=Fs)
'''