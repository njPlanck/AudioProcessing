import math
import numpy as np
from scipy.io import wavfile  # to read and write wav files
from scipy import signal
#import simpleaudio as sa  # alternative to sounddevice
import matplotlib.pyplot as plt  # for plotting
from matplotlib import style




#read wav file
fs_guit, guit = wavfile.read ("audio_samples/guit3.wav")  #load guit

norm_guit = guit/np.max(np.abs(guit))  #normalising the guit input between 0 and 1.

y_distortion = np.zeros_like(guit, dtype=np.float32)
y_hard_clip = np.zeros_like(guit, dtype=np.float32)
y_soft_clip = np.zeros_like(guit, dtype=np.float32)

#parameters
a = 8 #distortion
# a = 90 gives us a fuzz, and a less than 4 gives us an overdrive

#a0, a1, a2 are all co-efficients of the approximated polynomial of taylor's series for hard clipping
a0 = 0
a1 = 1
a2 = 0.5 


for t in range(len(guit)):
    #polynomial approximation of a taylor expansion for a hard clip
    y_hard_clip[t] = (a0) + (a1 * guit[t]) + (a2 * guit[t]**2)
    
    #impletmenting the distortion
    exp_component = 1 - np.exp(-a * abs(norm_guit[t]))
    y_distortion[t] = np.sign(norm_guit[t])*exp_component

    #setting the boudary conditions for soft clip
    if abs(norm_guit[t]) >= 0 and abs(norm_guit[t]) <= 1/3:
        y_soft_clip[t] = np.sign(norm_guit[t]) * 2 * abs(norm_guit[t])
    elif abs(norm_guit[t]) >= 1/3 and abs(norm_guit[t]) <= 2/3:
        y_soft_clip[t] = np.sign(norm_guit[t]) * (3 - (2 - (3 * abs(norm_guit[t])))**2)/2
    elif abs(norm_guit[t]) >= 2/3 and abs(norm_guit[t]) <= 1:
        y_soft_clip[t] = np.sign(norm_guit[t])
    
    




y = y_hard_clip
# Save the output file
#Normalize output for 16-bit WAV format
y_dist = (y_distortion / np.max(np.abs(y_distortion)) * 32767).astype(np.int16)
y_hard = (y_hard_clip / np.max(np.abs(y_hard_clip)) * 32767).astype(np.int16)
y_soft = (y_soft_clip / np.max(np.abs(y_soft_clip)) * 32767).astype(np.int16)

wavfile.write("output_distortion.wav", fs_guit, y_dist)
wavfile.write("output_hardclipped.wav", fs_guit, y_hard)
wavfile.write("output_softclipped.wav", fs_guit, y_soft)


fig, axs = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
fig.suptitle("Guitar Signal Processing Comparison", y=1.02)

axs[0].plot(guit)
axs[0].set_title("Input Guit")
axs[0].set_ylabel("Amplitude")

axs[1].plot(y_hard)
axs[1].set_title("Hard Clipped Output")
axs[1].set_ylabel("Amplitude")

axs[2].plot(y_soft)
axs[2].set_title("Soft Clipped Output")
axs[2].set_ylabel("Amplitude")

axs[3].plot(y_dist)
axs[3].set_title("Distorted Output")
#axs[3].set_xlabel("Samples")
#axs[3].set_ylabel("Amplitude")

plt.tight_layout()
plt.show()

