import math
import numpy as np
from scipy.io import wavfile  # to read and write wav files
from scipy import signal, linalg
import sounddevice as sd  # for direct sound playing
import matplotlib.pyplot as plt  # for plotting
from matplotlib import style



#read wav file
fs, x = wavfile.read ("audio_samples/guit3.wav")     #load input audio

duration = len(x)/fs

t = np.linspace(0, duration, len(x))

#panning modulation of 1Hz.

p = np.sin(2 * np.pi * 1 * t)

#Energy-preserving panning
gL = (1 + p) / np.sqrt(2 * (1 + p ** 2))
gR = (1 - p) / np.sqrt(2 * (1 + p ** 2))

# Apply gains
left = gL * x
right = gR * x

# Stereo signal
y = np.vstack([left, right]).T


y = (y / np.max(np.abs(y)) * 32767).astype(np.int16)
wavfile.write("outputs/output_panning.wav", fs, y)

plt.plot (y)
plt.show()


