import numpy as np
from scipy.io import wavfile  # to read and write wav files
import matplotlib.pyplot as plt

# === Parameters ===
fs = 44100
duration = 3.0
f0 = 440.0              # Base frequency
f_lfo = 2.0             # LFO frequency
depth = 0.4             # LFO depth
a_target = 0.5          # Desired average amplitude
alpha = 0.001           # Averager smoothing factor (equal attack/release)

# === Time vector ===
t = np.arange(0, duration, 1/fs)
samples = len(t)

# === LFO-modulated frequency ===
f_mod = f0 * (1 + depth * np.sin(2 * np.pi * f_lfo * t))
w = 2 * np.pi * f_mod / fs
r_mod = 2 * np.cos(w)

# === Output buffer ===
x = np.zeros(samples)
x[0] = 1.0  # Initial impulse
x[1] = r_mod[0] * x[0]

# === Amplitude tracking ===
a = x[0]**2  # Initial detected amplitude

# === Oscillator with amplitude correction ===
for n in range(2, samples):
    # Oscillator
    x_n = r_mod[n] * x[n - 1] - x[n - 2]

    # Amplitude detection (squared + leaky integrator)
    a = (1 - alpha) * a + alpha * x_n**2

    # Correction factor
    scale = (a - a_target) / 10 + 1

    # Normalize output and memory
    x_n /= scale
    x[n - 1] /= scale  # Back-correct the previous sample

    # Store output
    x[n] = x_n

# === Normalize and write to file ===
y = x

#Save the output file
#Normalize output for 16-bit WAV format
y = (x / np.max(np.abs(y)) * 32767).astype(np.int16)
wavfile.write("outputs/output_oscillator.wav", fs, y)

plt.plot (y)
plt.show()
