import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt

# --- Read audio using scipy.io.wavfile ---
fs, x = wavfile.read("audio_samples/fox.wav")  # sr: sample rate, audio: ndarray


# --- Time-stretch parameters ---
frameSize = 512
hopSize_analysis = frameSize/6
hopSize_synthesis = frameSize/4

stretch_factor = hopSize_analysis/hopSize_synthesis

# --- STFT ---
_, _, X = signal.stft (x, fs, window='hann', nperseg=frameSize, noverlap=frameSize-hopSize_analysis)
magnitude = np.abs(X)
phase = np.angle(X)

num_bins, num_frames = X.shape
new_num_frames = int(num_frames / stretch_factor)
output_stft = np.zeros((num_bins, new_num_frames), dtype=complex)

# --- Initialize phase tracking ---
phase_acc = phase[:, 0]
output_stft[:, 0] = magnitude[:, 0] * np.exp(1j * phase_acc)
omega = 2 * np.pi * np.arange(num_bins) / frameSize

# --- Phase vocoder time-stretch loop ---
for t in range(1, new_num_frames):
    t_orig = t * stretch_factor
    t_int = int(np.floor(t_orig))
    t_frac = t_orig - t_int

    if t_int + 1 >= num_frames:
        break

    mag = (1 - t_frac) * magnitude[:, t_int] + t_frac * magnitude[:, t_int + 1]
    delta_phi = phase[:, t_int + 1] - phase[:, t_int]
    delta_phi = np.angle(np.exp(1j * delta_phi))  # unwrap
    true_freq = omega + delta_phi / hopSize_analysis
    phase_acc += true_freq * hopSize_synthesis

    output_stft[:, t] = mag * np.exp(1j * phase_acc)

# --- Inverse STFT ---
_, y = signal.istft(output_stft, fs, window='hann', nperseg=frameSize, noverlap=frameSize-hopSize_synthesis)



# Save the output file
#Normalize output for 16-bit WAV format
y = (y / np.max(np.abs(y)) * 32767).astype(np.int16)
wavfile.write("outputs/output_time_stretching.wav", fs, y)

plt.plot (y)
plt.show()
