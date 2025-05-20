import math
import numpy as np
from scipy.io import wavfile
import sounddevice as sd
import matplotlib.pyplot as plt

def bandpass_filter(x, fs, fc, fd, block_size=100):
    """
    Implements a bandpass filter with center frequency fc and bandwidth fd.
    :param x: Input signal
    :param fs: Sampling frequency
    :param fc: Center frequency (Hz)
    :param fd: Bandwidth (Hz)
    :param block_size: Number of samples per control update
    :return: Filtered signal
    """
    y = np.zeros_like(x, dtype=float)
    a0, a1 = 0, 0  # Allpass buffer variables
    
    for to in range(1, len(x) - block_size, block_size):
        f1 = (fc - fd / 2) / fs  # Low cutoff frequency
        f2 = (fc + fd / 2) / fs  # High cutoff frequency
        
        c1 = (math.tan(math.pi * f1) - 1) / (math.tan(math.pi * f1) + 1)  # High-pass filter coeff
        c2 = (math.tan(math.pi * f2) - 1) / (math.tan(math.pi * f2) + 1)  # Low-pass filter coeff
        
        for ti in range(block_size):
            t = to + ti
            a1 = a0
            a0 = c1 * x[t] + x[t - 1] - c1 * a1  # High-pass filter
            y[t] = (a0 + x[t]) / 2  # First stage output
            
            a1 = a0
            a0 = c2 * y[t] + y[t - 1] - c2 * a1  # Low-pass filter
            y[t] = (a0 + y[t]) / 2  # Second stage output
    
    return y

# Read input WAV file
fs, x = wavfile.read("audio_samples/guit3.wav")
y = bandpass_filter(x, fs, fc=1000, fd=500)

# Normalize output for 16-bit WAV format
y = (y / np.max(np.abs(y)) * 32767).astype(np.int16)

# Write output WAV file
wavfile.write("bandpass.wav", fs, y)

# Play filtered signal
sd.play(y, fs)
sd.wait()

# Plot filtered waveform
plt.plot(y)
plt.title("Bandpass Filtered Signal")
plt.show()