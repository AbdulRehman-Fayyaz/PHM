import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

fs = 20000                      # samples per second
t = np.arange(0, 1.0, 1/fs)     # 1 second of time
rng = np.random.default_rng(42)

# Healthy: shaft turning at 30 Hz + small random noise
healthy = 0.5*np.sin(2*np.pi*30*t) + 0.2*rng.standard_normal(t.size)

# Faulty: same signal + a short "knock" 100 times per second
faulty = healthy.copy()
for start in np.arange(0, 1.0, 1/100):
    idx = (t >= start) & (t < start + 0.005)
    tau = t[idx] - start
    faulty[idx] += 2.0*np.exp(-800*tau)*np.sin(2*np.pi*3000*tau)

# Two simple health numbers
for name, x in [("Healthy", healthy), ("Faulty", faulty)]:
    rms = np.sqrt(np.mean(x**2))
    kurt = kurtosis(x, fisher=False)
    print(f"{name}: RMS = {rms:.3f}, Kurtosis = {kurt:.2f}")

# Plot the first 0.1 seconds
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(t, healthy, color="#6764f6"); ax[0].set_title("Healthy bearing")
ax[1].plot(t, faulty, color="#FF000F");  ax[1].set_title("Faulty bearing")
ax[1].set_xlim(0, 0.1); ax[1].set_xlabel("Time [s]")
plt.tight_layout()
plt.show()
