import math
import numpy as np
from scipy.io import wavfile  # to read and write wav files
from scipy import signal, linalg
import sounddevice as sd  # for direct sound playing
import matplotlib.pyplot as plt  # for plotting
from matplotlib import style
from scipy.signal import convolve

#read wav file
fs, x = wavfile.read ("audio_samples/guit3.wav")     #load input audio


#Normalize input
x = x / np.max(np.abs(x))

# 2. Create two white-noise impulse responses (~4000 samples)
ir_len = 4000
ir_left = np.random.randn(ir_len)
ir_right = np.random.randn(ir_len)
ir_same = np.random.randn(ir_len)  # Same for both for non-decorrelated

# 3. Convolve input with left and right IRs
left_decorrelated = convolve(x, ir_left, mode='full')
right_decorrelated = convolve(x, ir_right, mode='full')
stereo_decorrelated = np.stack((left_decorrelated, right_decorrelated), axis=-1)

# 4. Non-decorrelated version
left_same = convolve(x, ir_same, mode='full')
right_same = convolve(x, ir_same, mode='full')
stereo_same = np.stack((left_same, right_same), axis=-1)

# Normalize both for listening
stereo_decorrelated /= np.max(np.abs(stereo_decorrelated))
stereo_same /= np.max(np.abs(stereo_same))

# 5. Play both back-to-back
y = np.concatenate((stereo_decorrelated, stereo_same), axis=0)

y = (y / np.max(np.abs(y)) * 32767).astype(np.int16)
wavfile.write("outputs/output_decorrelated_stereo.wav", fs, y)

plt.plot (y)
plt.show()