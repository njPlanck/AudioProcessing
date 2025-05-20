import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal
import matplotlib.pyplot as plt


input_file = "audio_samples/fox.wav"  #input file

fs, x = wavfile.read(input_file)  #load the input file

x = x.astype(np.float32) / np.max(np.abs(x))  # Normalize

frameSize = 1024
hopSize = frameSize/4
pitch_factor = 1.27  # >1 raises pitch, <1 lowers pitch

_, _, X = signal.stft (x, fs, window='hann', nperseg=frameSize, noverlap=frameSize-hopSize)

n_bins, n_frames = X.shape
w = np.arange(X.shape[0])  

X_shifted = np.zeros_like(X, dtype=np.complex64) #initialize the shifted output
phase = np.angle(X[:, 0])
new_phase = phase
X_shifted[:, 0] = X[:, 0]  # Copy first frame

#Direct pitch shift through bin shifting
for i in range(1, n_frames):
    mag = np.abs(X[:, i])
    phase = np.angle(X[:, i])
    prev_mag = np.abs(X[:, i - 1])
    prev_phase_frame = np.angle(X[:, i - 1])


    # Phase difference and wrap
    phase_proj = prev_phase_frame + (2 * np.pi * hopSize * w / frameSize)
    phase_unwrap = phase + np.round((phase_proj - phase)/(2 * np.pi)) * 2 * np.pi
    delta_phase = phase_unwrap - prev_phase_frame
    new_phase = prev_phase_frame + pitch_factor * delta_phase

    for bin_index in range(n_bins):
        kw = pitch_factor * bin_index
        if kw >= n_bins - 1:
            continue
        low = int(np.floor(kw))
        high = int(np.ceil(kw))
        frac = kw - low

        # Interpolated complex value
        val = mag[bin_index] * np.exp(1j * new_phase[bin_index])
        X_shifted[low, i] += (1 - frac) * val
        X_shifted[high, i] += frac * val

    prev_phase = new_phase


_, y = signal.istft(X_shifted, fs, nperseg=frameSize, noverlap=frameSize - hopSize)



# Save the output file
#Normalize output for 16-bit WAV format
y = (y / np.max(np.abs(y)) * 32767).astype(np.int16)
wavfile.write("outputs/output_pitch_shifting.wav", fs, y)

plt.plot (y)
plt.show()
