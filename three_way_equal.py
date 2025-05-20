import numpy as np
import math
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Read the input WAV file
fs, x = wavfile.read("audio_samples/guit3.wav")
x = x.astype(float)  # Convert to float for processing

# Define cutoff frequencies (normalized)
fc_low = 500 / fs  # Example: 500 Hz cutoff
fc_high = 2000 / fs  # Example: 2000 Hz cutoff

# Compute filter parameters
K_low = math.tan(math.pi * fc_low)
K_high = math.tan(math.pi * fc_high)

# Initialize filter buffers
y_low = np.zeros_like(x)
y_high = np.zeros_like(x)
y_band = np.zeros_like(x)

# Processing loops
x_prev1 = 0
x_prev2 = 0
y_low_prev1 = 0
y_low_prev2 = 0
y_high_prev1 = 0
y_high_prev2 = 0

# Loop through the audio sample by sample
for t in range(2, len(x)):
    # Lowpass filter calculation
    y_low[t] = (1 / (1 + math.sqrt(2) * K_low + K_low ** 2)) * (
        K_low ** 2 * x[t] + 2 * K_low ** 2 * x[t - 1] + K_low ** 2 * x[t - 2]
        - 2 * (K_low ** 2 - 1) * y_low_prev1
        - (1 - math.sqrt(2) * K_low + K_low ** 2) * y_low_prev2
    )

    # Highpass filter calculation
    y_high[t] = (1 / (1 + math.sqrt(2) * K_high + K_high ** 2)) * (
        x[t] - 2 * x[t - 1] + x[t - 2]
        - 2 * (K_high ** 2 - 1) * y_high_prev1
        - (1 - math.sqrt(2) * K_high + K_high ** 2) * y_high_prev2
    )

    # Bandpass signal = Highpass on Lowpass signal
    y_band[t] = x[t] - y_low[t] - y_high[t]

    # Update previous values
    y_low_prev2 = y_low_prev1
    y_low_prev1 = y_low[t]
    
    y_high_prev2 = y_high_prev1
    y_high_prev1 = y_high[t]

# Normalize and convert to int16 for WAV output
y_low = (y_low / np.max(np.abs(y_low)) * 32767).astype(np.int16)
y_high = (y_high / np.max(np.abs(y_high)) * 32767).astype(np.int16)
y_band = (y_band / np.max(np.abs(y_band)) * 32767).astype(np.int16)

# Save the output WAV files
wavfile.write("output_lowpass.wav", fs, y_low)
wavfile.write("output_highpass.wav", fs, y_high)
wavfile.write("output_bandpass.wav", fs, y_band)

# Plot the waveforms
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(y_low, label="Lowpass Output")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(y_band, label="Bandpass Output", color="orange")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(y_high, label="Highpass Output", color="red")
plt.legend()

plt.show()
