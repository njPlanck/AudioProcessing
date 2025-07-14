import numpy as np
from scipy.io import wavfile
from scipy import signal
import sounddevice
import matplotlib.pyplot as plt


fs = 44100

mls = signal . max_len_seq (17) [0]  # maximum length sequence of 0s and 1s
mls = 2.0 * mls - 1  # turn mls into {+1,-1} signal
# concatenate last half of mls + mls + firsthalf of mls; so that cross-correlation will be roughly cyclical
mlssig = np . array (np . concatenate ((mls [len (mls) // 2:], mls, mls [:len (mls) // 2])))
#wavfile.write ("mls.wav", fs, mlssig * (2**15-1), np.int16))

mlsr = sounddevice . playrec (mlssig, fs, 1) [:,0]  # record room response to mls
sounddevice . wait()

ir = signal . correlate (mlsr, mls, 'valid')  # calculate the cross-correlation
# find the starting point of impulse response in cross-correlation
mxir = max (ir)
t = np . argmax (ir > 0.5 * mxir)
ir = ir [t - 20 : t + int (0.4 * fs)]  # cut out 400ms or impulse response
plt . plot (ir)
plt . show()

wavfile.write ("ir.wav", fs, np.array (ir / max (ir, key=abs) * (2**15 - 1), np.int16))