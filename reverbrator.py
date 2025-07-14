import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal
import matplotlib.pyplot as plt

def delay_signal(x, delay_samples, gain=1.0):
    y = np.zeros(len(x) + delay_samples)
    y[delay_samples:delay_samples+len(x)] += gain * x
    return y

def comb_filter(x, delay_samples, g):
    y = np.zeros(len(x) + delay_samples)
    for n in range(delay_samples, len(y)):
        y[n] = x[n - delay_samples] + g * y[n - delay_samples]
    return y

def compute_delays(fs, c=343):
    # Room geometry
    room_width = 10.0   # meters
    room_depth = 15.0   # meters
    source = np.array([2.0, 3.0])    # (x, y)
    listener = np.array([7.0, 6.0])  # (x, y)

    def distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    # Direct path
    d_direct = distance(source, listener)
    t_direct = d_direct / c * 1000  # ms

    # Early reflections (image sources)
    image_left  = np.array([-source[0], source[1]])
    image_right = np.array([2 * room_width - source[0], source[1]])
    image_back  = np.array([source[0], -source[1]])

    d_left  = distance(image_left, listener)
    d_right = distance(image_right, listener)
    d_back  = distance(image_back, listener)

    t_left  = d_left  / c * 1000  # ms
    t_right = d_right / c * 1000  # ms
    t_back  = d_back  / c * 1000  # ms

    # Comb filter delays based on room modes (n = (1,0), (0,1), (1,1))
    def mode_freq(n_x, n_y):
        return (c / 2) * np.sqrt((n_x / room_width) ** 2 + (n_y / room_depth) ** 2)

    f1 = mode_freq(1, 0)
    f2 = mode_freq(0, 1)
    f3 = mode_freq(1, 1)

    t_comb1 = 1000 / f1  # period in ms
    t_comb2 = 1000 / f2
    t_comb3 = 1000 / f3

    delays_ms = {
        'direct': round(t_direct, 4),
        'left':   round(t_left, 4),
        'right':  round(t_right, 4),
        'back':   round(t_back, 4),
        'comb1':  round(t_comb1, 4),
        'comb2':  round(t_comb2, 4),
        'comb3':  round(t_comb3, 4),
    }

    delays_samples = {k: int(fs * (v / 1000)) for k, v in delays_ms.items()}
    return delays_ms, delays_samples

def moorer_reverb(x, fs, g=0.5, b=0.5):
    # === Convert delay times to samples (fs = 44100 Hz default) ===
    delays_ms, delays_samples = compute_delays(fs)
    delays_samples = {k: int(fs * (v / 1000)) for k, v in delays_ms.items()}

    # === Direct sound ===
    y_direct = delay_signal(x, delays_samples['direct'], gain=1.0)

    # Early reflections
    y_left = delay_signal(x, delays_samples['left'], gain=0.7)
    y_right = delay_signal(x, delays_samples['right'], gain=0.7)
    y_back = delay_signal(x, delays_samples['back'], gain=0.7)

    # Pad all to the same length
    max_early_len = max(len(y_left), len(y_right), len(y_back))
    y_left = np.pad(y_left, (0, max_early_len - len(y_left)))
    y_right = np.pad(y_right, (0, max_early_len - len(y_right)))
    y_back = np.pad(y_back, (0, max_early_len - len(y_back)))

    y_early = y_left + y_right + y_back

    # === Comb filters ===
    y_comb1 = comb_filter(x, delays_samples['comb1'], g)
    y_comb2 = comb_filter(x, delays_samples['comb2'], g)
    y_comb3 = comb_filter(x, delays_samples['comb3'], g)

    # Pad all comb outputs to the same length
    max_comb_len = max(len(y_comb1), len(y_comb2), len(y_comb3))
    y_comb1 = np.pad(y_comb1, (0, max_comb_len - len(y_comb1)))
    y_comb2 = np.pad(y_comb2, (0, max_comb_len - len(y_comb2)))
    y_comb3 = np.pad(y_comb3, (0, max_comb_len - len(y_comb3)))

    y_comb_total = y_comb1 + y_comb2 + y_comb3
    y_comb_total *= b

    # === Align all signals to the same length ===
    max_len = max(len(y_direct), len(y_early), len(y_comb_total))
    y_direct = np.pad(y_direct, (0, max_len - len(y_direct)))
    y_early = np.pad(y_early, (0, max_len - len(y_early)))
    y_comb_total = np.pad(y_comb_total, (0, max_len - len(y_comb_total)))

    # === Final output ===
    y = y_direct + y_early + y_comb_total
    return y

# === Load audio file ===
fs, x = wavfile.read('audio_samples/fox.wav')  # 16-bit PCM or float32 mono file

# Convert to float32 if needed
if x.dtype != np.float32:
    x = x.astype(np.float32) / np.iinfo(x.dtype).max

# === Apply reverb ===
y = moorer_reverb(x, fs, g=0.5, b=0.5)


# Save the output file
#Normalize output for 16-bit WAV format
y = (y / np.max(np.abs(y)) * 32767).astype(np.int16)
wavfile.write("outputs/output_reverbrator.wav", fs, y)

plt.plot (y)
plt.show()
