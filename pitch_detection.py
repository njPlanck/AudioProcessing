import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, get_window

# --- Signal parameters ---
fs = 16000
duration = 2.0
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# --- Create chirp signal with harmonics ---
a = 100  # chirp rate
fundamental = np.sin(np.pi * a * t**2)

# Add harmonics with random phase and amplitude
signal = fundamental.copy()
for h in range(2, 8):
    amp = np.random.uniform(0.2, 0.6)
    phase = np.random.uniform(0, 2*np.pi)
    signal += amp * np.sin(h * np.pi * a * t**2 + phase)

# Add noise
signal += 0.05 * np.random.randn(len(signal))

# --- Ground truth frequency: f(t) = a * t ---
true_freq = a * t

# --- Analysis parameters ---
block_size = 1024
hop_size = 512
num_blocks = (len(signal) - block_size) // hop_size
detected_freqs = []
times = []

# --- Block-by-block analysis ---
for i in range(num_blocks):
    start = i * hop_size
    end = start + block_size
    block = signal[start:end] * get_window('hann', block_size)

    # Autocorrelation
    ac = correlate(block, block, mode='full')
    mid = len(ac) // 2
    ac = ac[mid:]  # keep only second half

    # Find first positive zero crossing
    zero_crossings = np.where((ac[:-1] < 0) & (ac[1:] >= 0))[0]
    if len(zero_crossings) == 0:
        detected_freqs.append(0)
        times.append((start + end) / 2 / fs)
        continue
    zc = zero_crossings[0]

    # From zc onward, find peak > 80% of max
    peak_search_region = ac[zc:]
    max_val = np.max(peak_search_region)
    threshold = 0.8 * max_val

    peaks = np.where((peak_search_region[1:-1] > peak_search_region[:-2]) &
                     (peak_search_region[1:-1] > peak_search_region[2:]) &
                     (peak_search_region[1:-1] > threshold))[0]

    if len(peaks) == 0:
        detected_freqs.append(0)
    else:
        peak_lag = zc + peaks[0] + 1
        freq = fs / peak_lag
        detected_freqs.append(freq)

    times.append((start + end) / 2 / fs)

# --- Ground truth at analysis times ---
true_freq_at_blocks = a * np.array(times)

# --- Plot ---
plt.figure(figsize=(10, 4))
plt.plot(times, true_freq_at_blocks, label='True Frequency', linewidth=2)
plt.plot(times, detected_freqs, label='Detected Frequency (Autocorrelation)', linewidth=2, linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Pitch Detection via Autocorrelation")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
