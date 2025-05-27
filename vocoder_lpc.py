import math
import numpy as np
from scipy.io import wavfile  # to read and write wav files
from scipy import signal, linalg
import sounddevice as sd  # for direct sound playing
import matplotlib.pyplot as plt  # for plotting
from matplotlib import style



#read wav file
fs_mod, mod = wavfile.read ("audio_samples/guit3.wav")  #load modulator
fs_car, car = wavfile.read ("audio_samples/fox.wav")    #load carrier

min_len = min(len(mod), len(car))
mod = mod[:min_len]
car = car[:min_len]

# === Parameters ===
blocksize = 1024
m = 16  # LPC order
hop = blocksize//2 # no overlap

y = np.zeros_like(car, dtype=np.float64) #initialize the output

for t in range(m, len(car) - blocksize, hop): #looping between the LPC order and len of the carrier + blocksize at hop intervals
    pre_emphasis = 0.97
    car_block = signal.lfilter([1, -pre_emphasis], 1, car[t:t + blocksize])
    mod_block = signal.lfilter([1, -pre_emphasis], 1, mod[t:t + blocksize])
    window = np.hamming(blocksize)
    car_block *= window
    mod_block *= window

    car_corr = signal.correlate(car_block, car_block, mode='full')
    mod_corr = signal.correlate(mod_block, mod_block, mode='full')

    mid = len(car_corr) // 2                          #to find the lag
    car_corr = car_corr[mid:mid + m + 1]
    mod_corr = mod_corr[mid:mid + m + 1]

    try:
        # LPC coefficients
        R_x = car_corr[:-1]
        R_y = mod_corr[:-1]
        rhs_x = -car_corr[1:]
        rhs_y = -mod_corr[1:]
        p_x = linalg.solve_toeplitz((R_x, R_x), rhs_x)
        p_y = linalg.solve_toeplitz((R_y, R_y), rhs_y)
    except np.linalg.LinAlgError:
        continue

    if np.any(np.isnan(p_x)) or np.max(np.abs(p_x)) > 10:
        print(f"Skipping frame {t} due to unstable p_car")
        continue
    if np.any(np.isnan(p_y)) or np.max(np.abs(p_y)) > 10:
        print(f"Skipping frame {t} due to unstable p_mod")
        continue

    
    residual = np.zeros(blocksize)
    for n in range(m, blocksize):
        residual[n] = car_block[n] + np.inner(p_x, car_block[n - m:n][::-1])

    # === Synthesize using Modulator's LPC
    synth = np.zeros(blocksize)
    for n in range(m, blocksize):
        synth[n] = residual[n] - np.inner(p_y, synth[n - m:n][::-1])

    y[t:t + blocksize] += synth

# === Normalize and Save ===
y = (y / np.max(np.abs(y)) * 32767).astype(np.int16)
wavfile.write("outputs/output_vocoder_mutation.wav", fs_mod, y)

plt.plot(y)
plt.title("LPC Vocoder Output")
plt.show()