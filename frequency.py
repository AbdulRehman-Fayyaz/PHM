import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

# --- same signals as Step 2 ---
fs = 20000
t = np.arange(0, 1.0, 1/fs)
rng = np.random.default_rng(42)
healthy = 0.5*np.sin(2*np.pi*30*t) + 0.2*rng.standard_normal(t.size)
faulty = healthy.copy()
for start in np.arange(0, 1.0, 1/100):
    idx = (t >= start) & (t < start + 0.005)
    tau = t[idx] - start
    faulty[idx] += 2.0*np.exp(-800*tau)*np.sin(2*np.pi*3000*tau)

# --- helper functions ---
def spectrum(x):
    X = np.abs(np.fft.rfft(x)) / len(x) * 2
    f = np.fft.rfftfreq(len(x), 1/fs)
    return f, X

def envelope_spectrum(x):
    b, a = butter(4, [2000, 4000], btype="bandpass", fs=fs)  # keep 2-4 kHz ringing
    env = np.abs(hilbert(filtfilt(b, a, x)))                  # outline of the knocks
    env -= env.mean()
    return spectrum(env)

# --- find the strongest peak between 10 and 500 Hz ---
for name, x in [("Healthy", healthy), ("Faulty", faulty)]:
    f, E = envelope_spectrum(x)
    m = (f > 10) & (f < 500)
    print(f"{name}: envelope peak at {f[m][np.argmax(E[m])]:.0f} Hz, "
          f"height {E[m].max():.3f}")

# --- plots ---
fig, ax = plt.subplots(2, 2, figsize=(12, 7))
for col, (name, x, c) in enumerate([("Healthy", healthy, "#6764f6"),
                                     ("Faulty", faulty, "#FF000F")]):
    f, X = spectrum(x)
    ax[0, col].plot(f, X, color=c); ax[0, col].set_xlim(0, 5000)
    ax[0, col].set_title(f"{name}: normal FFT")
    f, E = envelope_spectrum(x)
    ax[1, col].plot(f, E, color=c); ax[1, col].set_xlim(0, 500)
    ax[1, col].set_ylim(0, 0.4)
    ax[1, col].set_title(f"{name}: envelope spectrum")
    ax[1, col].set_xlabel("Frequency [Hz]")
plt.tight_layout()
plt.show()