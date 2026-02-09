import pandas as pd
import numpy as np
import h5py
from temporaldata import Data
from pathlib import Path
import matplotlib.pyplot as plt


raw_parquet = Path("/root/capsule/data/parquet_iEEG/Patient1/iEEG/Patient1_ieeg_20251126_042728.parquet")
h5_path = Path(f"/root/capsule/results/iEEG/20251126_042728__seg000__Patient1_ieeg_20251126_042728.h5")

df = pd.read_parquet(raw_parquet)

with h5py.File(h5_path, "r") as f:
    d = Data.from_hdf5(f)

    # Compare counts
    print("raw rows:", len(df), "processed n_time:", d.ieeg.data.shape[0])
    print("processed n_chan:", d.ieeg.data.shape[1])
    
    # Compare one channel if you know a name
    ids = [
        v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
        for v in d.channels.id
    ]
    missing = [c for c in ids if c not in df.columns]
    print("missing:", len(missing), "e.g.", missing[:5])

    if ids[0] in df.columns:
        raw = df[ids[0]].to_numpy()
        proc = d.ieeg.data[:, 0]
        print("example channel:", ids[0])
        print("max abs diff:", float(np.nanmax(np.abs(raw - proc))))

    fs = float(d.ieeg.sampling_rate)
    x = d.ieeg.data[: int(fs*10), 0]   # first 10 seconds, channel 0
    start = d.ieeg.domain.start.item()
    t = np.arange(len(x)) / fs + start

    plt.figure()
    plt.plot(t, x)
    plt.xlabel("time (s)")
    plt.ylabel("signal")
    plt.title(f"{d.segment} ch0")
    out_png = "/root/capsule/results/debug_seg000_ch0.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print("saved plot to:", out_png)
