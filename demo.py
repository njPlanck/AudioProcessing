import math
import numpy as np
from scipy.io import wavfile  # to read and write wav files
import sounddevice as sd  # for direct sound playing
#import simpleaudio as sa  # alternative to sounddevice
import matplotlib.pyplot as plt  # for plotting

#read wav file
fs, x = wavfile.read ("audio_samples/guit3.wav")
# fs = sampling rate (44100)
# x = numpy-array (1D for single channel, 2D for more channels) of type int16, range -32768..+32767

y = np.zeros (x.shape)  # output signal of same size

# two variables as signal buffer: allpass[t] = a0,  allpass[t - 1] = a1
a0=0; a1=0

# control flow loop over x with step size 100
for to in range (1, len (x) - 99, 100):
   # cut-off frequency oscillating with 1Hz between 100 and 1900Hz (LFO, low frequency oscillation)
   fc = (1000 + math.sin (to * 1 * 2 * math.pi / fs) * 900) / fs
   # filter parameter depending on cut-off frequency
   c = (math.tan (math.pi * fc) - 1) / (math.tan (math.pi * fc) + 1)
   

   # signal flow loop over block
   for ti in range (0, 100):
      t = to + ti  # actual time
      a1 = a0  # when time progresses, the buffer shifts
      a0 = c * x[t] + x[t - 1] - c * a1  # output of allpass filter
      #y[t] = (a0 + x[t]) / 2  # lowpass filter via allpass filter
      y[t] = (x[t] - a0) / 2  # lowpass filter via allpass filter

      
# output result into wav file
# the array has to be converted to 2-byte integers
# wavfile.write ("lowpass.wav", fs, np.array (y, np.int16))
# another possibility:
# wavfile.write ("lowpass.wav", fs, y.astype (np.int16))
# with normalization so that the maximum amplitude fills the 16-bit-range:
# wavfile.write ("lowpass.wav", fs, (y / max (y, key=abs) * ((2 ** 15) - 1)).astype (np.int16))
# if we write a floating point array, then an (unusual) floating point wav file is created,
# in which case the numbers must be between -1.0 and +1.0

# play result directly
# arguments: (data, samplingrate=None, mapping=None, blocking=False, loop=False)#@sd.play (np.array (y, np.int16), fs)
# arguments: (data, #channels, #bytes per sample, sampling rate)

#po = sa.play_buffer (np.array (y, np.int16), 1, 2, fs)
y = (y / np.max(np.abs(y)) * 32767).astype(np.int16)
#wavfile.write("output_low_filtered.wav", fs, y)
wavfile.write("output_high_filtered.wav", fs, y)

plt.plot (y)
plt.show()

# wait for sound to end
sd.wait()
#po.wait_done()

"""
# visualizing the frequency spectrum before and after applying the filter
def plot_spectrum (data, sampling_rate, title):
   Y = np.fft.fft(data)
   freq = np.fft.fftfreq(data.size, 1 / sampling_rate)

   plt.figure(figsize=(10, 6))
   plt.plot(freq, np.abs(Y))
   plt.title(title)
   plt.xlabel('Frequency [Hz]')
   plt.ylabel('|Y(f)|')
   plt.xlim([0, 4000])
   plt.grid()

plot_spectrum(x, fs, "Original Signal spectrum")
plot_spectrum(y, fs, "Filtered Signal spectrum. Bandpass at 500Hz")
plt.show()
"""