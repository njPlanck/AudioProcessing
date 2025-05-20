import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import math
from scipy.io import wavfile 

# Parameters
duration = 5  # seconds
freq = 5100  # Hz (frequency of the sine tone)
fs = 5102  # original low sampling rate

#fs = 44100  # high sample rate

'''
At a higher frequency of 44,100 Hz, the effect was that of a siren beep.

'''


# Time vector for the original sine wave
t_original = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Generated input sine wave at 5500 Hz
x = np.sin(2 * np.pi * freq * t_original)

y = np.zeros (x.shape)  #output signal of same size


delay = 3e-3  # 3ms max delay
lfo_freq = 5  # 5 Hz LFO frequency
t_delay = np.arange(len(x))/fs  # Time vector

# Generate the LFO (sinusoidal)
lfo = (np.sin(2 * np.pi * lfo_freq * t_delay) + 1) / 2  # Normalize between 0 and 1
m = (lfo * delay * fs).astype(int)  # Convert ms to samples
f = 0.5  #fractional delay


for i in range(len(x)):
    if i - m[i] >= 0:
        y[i] = (1-f) * x[i - m[i]] + (f * x[i - m[i]])    #output from linear interpolation
        
    else:
        y[i] = x[0]  # Avoid out-of-bounds access

# Save the output

y = (y / np.max(np.abs(y)) * 32767).astype(np.int16)
wavfile.write("output_linear_interpolation.wav", fs, y)

plt.plot(y)
plt.show()


