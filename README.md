# 🎧 Audio Processing

A collection of lecture-based experiments and implementations exploring **audio signal processing** concepts in Python. The focus ranges from **time-domain and frequency-domain filtering** to creative audio effects and transformations, based on hands-on DSP coursework.

---

##  Topics Covered

### 🎛 Filters & Equalizers

- **Bandpass Filter** with configurable center (`fc`) and bandwidth (`fd`)
- **Three-Way Equalizer** using cascaded low/high-pass filters and gain blending
- **Proof-of-concept**: Check if gain factors of 1 reconstruct the original signal

### 🌀 Audio Effects

- **Phaser Effect** using:
  - Single allpass filter with modulated `fc`
  - Four allpasses with independent low-frequency oscillators
  - Feedback loop implementation
- **Wah-Wah Effect** (4-fold), modulating `fc` with a constant-Q relationship
- **Rotary Speaker Effect** in stereo
- **Primitive Vocoder** based on Hilbert transform
- **Distortion Effects**:
  - Hard Clipping
  - Soft Clipping
  - Classic Distortion
- **Octaver** using zero-crossings and negative peak tracking
- **Compressor / Limiter** with squarer detector and dynamic response (-30dB threshold)

### 🔁 STFT-Based Processing

- **Vocoder Effect** using Short-Time Fourier Transform (STFT)
- **Time-Stretching** using STFT
- **Pitch-Shifting** by manipulating phase and magnitude in the STFT domain

### Sampling & Resampling

- **Resampling** a 2200Hz sine wave from 5100Hz to 5102Hz using:
  - Linear interpolation
  - Lanczos interpolation
  - Allpass interpolation

### Signal Modeling

- **Oscillator with LFO** control (digital resonator)
  - Frequency modulation with amplitude correction
  - Dynamic amplitude normalization using a squarer + averager

### Prediction & Estimation

- **Levinson-Durbin Algorithm (manual)** for:
  - Optimal Linear Prediction Coefficients
  - Autocorrelation matrix (Toeplitz)
  - Time-domain prediction for symmetric signals

---

## Getting Started

1. Clone this repo:
   ```bash
   git clone https://github.com/njPlanck/AudioProcessing.git
   cd audio-processing
