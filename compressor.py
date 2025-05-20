import math
import numpy as np
from scipy.io import wavfile  # to read and write wav files
from scipy import signal
import sounddevice as sd  # for direct sound playing
#import simpleaudio as sa  # alternative to sounddevice
import matplotlib.pyplot as plt  # for plotting
from matplotlib import style




#read wav file
fs_guit, guit = wavfile.read ("audio_samples/guit3.wav")  #load guit
fs_fox, fox = wavfile.read ("audio_samples/fox.wav")    #load fox


guit = guit.astype(np.float32) / 32768.0
fox = fox.astype(np.float32) / 32768.0

guit = guit/10


# ensuring both signals havve the same length

min_len = min(len(guit), len(fox))
guit = guit[:min_len]
fox = fox[:min_len]

mixed_fox_guit = guit + fox


attack_t = 5
release_t = 130

attack_coeff = np.exp(-1.0 / (fs_guit * attack_t / 1000))
release_coeff = np.exp(-1.0 / (fs_guit * release_t / 1000))

y = np.zeros_like(mixed_fox_guit) #inititilising the output signal

# two variables as signal buffer: allpass[t] = a0,  allpass[t - 1] = a1
a0 = 0
a1 = 0

#threshold dB value is set at between -30 dB and 0 dB
thresholdDB = -30

for t in range(len(mixed_fox_guit)):
    a1 = a0
    squarer = mixed_fox_guit[t]**2
    squarerDB = 10 * math.log10(max(abs(squarer), 1e-10))

    #attack and release 
    # Gain reduction calculation

    '''
    for dB values below -30dB, we set r to 0dB,
    and for dB values that between -30dB and 0dB we use a linear ramp
    '''
    if squarerDB < thresholdDB:
        reduction_db = 0
    else:
        reduction_db = - (squarerDB + 30)  # Linear ramp from -30 to 0 dB

    # Smooth gain changes
    if a1 < mixed_fox_guit[t]:
        #attacks
        a0 = (1 - attack_coeff) * mixed_fox_guit[t] + (attack_coeff * a1)
        y[t] = mixed_fox_guit[t] + (attack_coeff * a0)
    else:
        #releases
        a0 = (1 - release_coeff) * mixed_fox_guit[t] + (release_coeff * a1)
        y[t] = mixed_fox_guit[t] + (release_coeff * a0)
    


# Save the output file
#Normalize output for 16-bit WAV format
y = (y / np.max(np.abs(y)) * 32767).astype(np.int16)
wavfile.write("output_compressor.wav", fs_guit, y)

plt.plot (y)
plt.show()

    
