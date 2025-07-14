import numpy as np
from scipy.io import wavfile
from scipy import signal
import sounddevice

fs, ir = wavfile . read ("audio_samples/ir.wav")
ir = ir * 1.0

fs, fox = wavfile . read ("audio_samples/fox.wav")

cfox = signal . convolve (fox, ir)  # use impulse response as filter

wavfile.write ("outputs/output_applyir.wav", fs, np . array (cfox / max (cfox, key=abs) * (2 ** 15 - 1), np.int16))
