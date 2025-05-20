import math
import numpy as np
from scipy.io import wavfile  # to read and write wav files
from scipy import signal
import sounddevice as sd  # for direct sound playing
#import simpleaudio as sa  # alternative to sounddevice
import matplotlib.pyplot as plt  # for plotting
from matplotlib import style


#hilbert filter function
def hilbert_filter(t, N=30):
    # Adjust t to center around 0
    t_centered = t - N

    if 0 <= t <= 2 * N:
        if t_centered == 0:
            return 0.0
        elif t_centered % 2 != 0:  # corrected the even/odd check
            return 2.0 / (math.pi * t_centered)
        else:
            return 0.0
    else:
        return 0.0
    
#generating the an impulse respnse from the hilbert filter
#parameters
N = 30                     #truncation interval +/- 30
filter_length = 2 * N + 1  # 61 for N=30,  which the length of the index from -30 to 30

hilbert_impulse_response = [hilbert_filter(n, N) for n in range(filter_length)]   #hilbert filter impulse response

#read wav file
fs_mod, modulator = wavfile.read ("audio_samples/guit3.wav")  #load modulator
fs_car, carrier = wavfile.read ("audio_samples/fox.wav")    #load carrier

#transform both signals by convolving them with the hilbert filter impulse response
modulator_transformed = signal.convolve(modulator,hilbert_impulse_response,mode='same')  #transformed modulator input signal
carrier_transformed = signal.convolve(carrier,hilbert_impulse_response,mode='same')      #transformed carrier input signal


#clipping the output, so the index does not run out of bounds.
if len(modulator) > len(carrier):
    y = np.zeros_like(carrier_transformed)
else:
    y = np.zeros_like(modulator_transformed)


#amplitude substitution for two transformed signals
for t in range(len(y)):
    y[t] = modulator_transformed[t] * (abs(carrier_transformed[t])/abs(modulator_transformed[t]))  #the absolute values of these transformed vales are the amplitudes of the signals


# Save the output file
#Normalize output for 16-bit WAV format
y = (y / np.max(np.abs(y)) * 32767).astype(np.int16)
wavfile.write("output_vocoder.wav", fs_mod, y)

plt.plot (y)
plt.show()

    
