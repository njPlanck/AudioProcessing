import math
import numpy as np
from scipy.io import wavfile  # to read and write wav files
import sounddevice as sd  # for direct sound playing
#import simpleaudio as sa  # alternative to sounddevice
import matplotlib.pyplot as plt  # for plotting

# Read wav file
fs, x = wavfile.read("audio_samples/guit3.wav")
# fs = sampling rate (44100)
# x = numpy-array (1D for single channel, 2D for more channels) of type int16, range -32768..+32767

# Parameters
b = 2 * np.pi * 5 / fs  # Rotation speed (2 Hz LFO)
a = 150  # Depth of pitch modulation in samples
l, r = 0.4, 0.7  # Left and right speaker amplitudes

y = np.zeros(x.shape)  #initialising the output  stereo rotary speaker from the left and right channels
yl = np.zeros(x.shape) #initialising output from the left channel
yr = np.zeros(x.shape) #initialsing output from the right channel



for t in range(len(x)):

    #parameters dependent on t

    sinBt = np.sin(b * t) 
    delay1 = int(t - (a * (1 - sinBt)))  #first simplified delay factor from the rotary effect formular
    delay2 = int(t - (a * (1 + sinBt)))  #second simplified delay factor from the rotary effect formular

    yl[t] = (l * (1 + sinBt) * x[delay1]) +  (r * (1 - sinBt) * x[delay2])  #output from the left channel
    
    '''
    to maintain symmetry, we can swap the values for l and r for both the right and left channel

    '''
    yr[t] = (r * (1 + sinBt) * x[delay1]) +  (l * (1 - sinBt) * x[delay2])  #output from the right channel



y = np.vstack((yl, yr)).T  #output  stereo rotary speaker from the left and right channels


# Save the output file
#Normalize output for 16-bit WAV format
y = (y / np.max(np.abs(y)) * 32767).astype(np.int16)
wavfile.write("output_stereo_audio.wav", fs, y)

plt.plot (y)
plt.show()
