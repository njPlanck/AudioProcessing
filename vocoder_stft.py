import math
import numpy as np
from scipy.io import wavfile  # to read and write wav files
from scipy import signal
import sounddevice as sd  # for direct sound playing
import matplotlib.pyplot as plt  # for plotting
from matplotlib import style



#read wav file
fs_mod, modulator = wavfile.read ("audio_samples/guit3.wav")  #load modulator
fs_car, carrier = wavfile.read ("audio_samples/fox.wav")    #load carrier

min_len = min(len(modulator), len(carrier))
modulator = modulator[:min_len]
carrier = carrier[:min_len]

y_stft = np.zeros(modulator.shape)
#frameSize = 4096 #this provides a better transiton of the modulator
frameSize = 128  #this mixes
hopSize = frameSize / 4

_, _, transformed_mod = signal.stft (modulator, fs_mod, window='hann', nperseg=frameSize, noverlap=frameSize-hopSize)
_, _, transformed_carrier = signal.stft (carrier, fs_mod, window='hann', nperseg=frameSize, noverlap=frameSize-hopSize)


y_stft = np.abs(transformed_mod) * np.exp(1j * np.angle(transformed_carrier))


_, y = signal.istft (y_stft, fs_mod, window='hann', nperseg=frameSize, noverlap=frameSize-hopSize)




# Save the output file
#Normalize output for 16-bit WAV format
y = (y / np.max(np.abs(y)) * 32767).astype(np.int16)
wavfile.write("outputs/output_vocoder_stft.wav", fs_mod, y)

plt.plot (y)
plt.show()
