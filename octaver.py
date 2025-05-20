import math
import numpy as np
from scipy.io import wavfile  # to read and write wav files
import sounddevice as sd  # for direct sound playing
#import simpleaudio as sa  # alternative to sounddevice
import matplotlib.pyplot as plt  # for plotting

# Read wav file
fs, x = wavfile.read("audio_samples/fox.wav")

x_n = 0 #amplitude follower
r = False #negative peak reached
s = False #sound on

y = np.zeros(x.shape)

for t in range(len(x)):
    if x[t] < x_n:
        x_n = x[t]
        r = True
    else:
        x_n = x[t] * 0.999
        if r and x[t] > 0:
            r = False
            s = not s  # toggle sound on/off
    
    if x[t] > 0 and s:
        y[t] = x[t]
    else:
        y[t] = 0.0

    
#test edit of code

# Save the output file
#Normalize output for 16-bit WAV format
y = (y / np.max(np.abs(y)) * 32767).astype(np.int16)
wavfile.write("outputs/output_octaver.wav", fs, y)

plt.plot (y)
plt.show()
