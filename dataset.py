import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.signal import butter, filtfilt, hilbert

fs = 20000
t = np.arange(0, 0.5, 1/fs)          # 0.5 s per signal
rng = np.random.default_rng(0)
b, a = butter(4, [2000, 4000], btype="bandpass", fs=fs)

def make_signal(kind):
    noise = rng.uniform(0.15, 0.35)
    x = 0.5*np.sin(2*np.pi*30*t) + noise*rng.standard_normal(t.size)
    if kind == "healthy":
        return x
    rate = 100 if kind == "outer" else 160
    strength = rng.uniform(0.5, 2.0)
    for start in np.arange(0, t[-1], 1/rate):
        idx = (t >= start) & (t < start + 0.005)
        tau = t[idx] - start
        amp = strength
        if kind == "inner":                       # rotates with shaft
            amp *= 0.6 + 0.4*np.cos(2*np.pi*30*start)
        x[idx] += amp*np.exp(-800*tau)*np.sin(2*np.pi*3000*tau)
    return x

def env_peak(x, target):
    env = np.abs(hilbert(filtfilt(b, a, x)))
    env -= env.mean()
    E = np.abs(np.fft.rfft(env)) / len(env) * 2
    f = np.fft.rfftfreq(len(env), 1/fs)
    m = (f > target-5) & (f < target+5)
    return E[m].max()

def features(x):
    rms = np.sqrt(np.mean(x**2))
    peak = np.max(np.abs(x))
    return {"rms": rms, "peak": peak, "crest": peak/rms,
            "kurtosis": kurtosis(x, fisher=False),
            "env_100Hz": env_peak(x, 100),
            "env_160Hz": env_peak(x, 160)}

rows = []
for kind in ["healthy", "outer", "inner"]:
    for i in range(100):
        f = features(make_signal(kind))
        f["label"] = kind
        rows.append(f)

df = pd.DataFrame(rows)
df.to_csv("bearing_features.csv", index=False)
print(df.shape)
print(df.groupby("label").mean().round(3))
