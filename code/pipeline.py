from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import datetime as dt
import os

# Disable warnings from docker
# os.environ.setdefault("RAY_DISABLE_METRICS", "1")
os.environ.setdefault("RAY_TMPDIR", "/root/capsule/results/ray_tmp")
os.makedirs(os.environ["RAY_TMPDIR"], exist_ok=True)

import numpy as np
import pandas as pd
import h5py
import pyarrow.parquet as pq

from brainsets.pipeline import BrainsetPipeline
from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    SubjectDescription,
    SessionDescription,
    DeviceDescription,
)
from brainsets.taxonomy import Species, Sex, RecordingTech

from temporaldata import Data, RegularTimeSeries, IrregularTimeSeries, ArrayDict, Interval



parser = ArgumentParser()
parser.add_argument("--reprocess", action="store_true", help="Overwrite existing processed .h5 outputs")
parser.add_argument(
    "--out_dir",
    type=str,
    default="/root/capsule/results",
    help="Where to write outputs (default: /root/capsule/results)",
)
parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"], help="Signal dtype")
parser.add_argument(
    "--non_signal_cols",
    type=str,
    default="DC1,sample_index,time,absolute_timestamp_utc,absolute_timestamp",
    help="Comma-separated columns to exclude from signals",
)


class Pipeline(BrainsetPipeline):
    brainset_id = "iEEG"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir: Path, args) -> pd.DataFrame:
        raw_dir = Path(raw_dir)

        meta_paths = sorted(raw_dir.glob("*_metadata_*.json"))
        rows = []

        for mp in meta_paths:
            with mp.open("r", encoding="utf-8") as f:
                meta = json.load(f)

            # base session id (same logic as you had)
            session_id = meta.get("start_timestamp") or mp.stem.replace("_metadata_", "_")

            parquet_entries = meta.get("parquet_files", [])
            if not parquet_entries:
                continue

            for seg_idx, ent in enumerate(sorted(parquet_entries, key=lambda e: int(e.get("start_sample", 0)))):
                fname = ent["filename"]
                start_sample = int(ent.get("start_sample", 0))

                parquet_path = raw_dir / fname
                csv_path = parquet_path.with_suffix(".csv")
                csv_path_str = str(csv_path) if csv_path.exists() else ""

                seg_id = f"{session_id}__seg{seg_idx:03d}__{parquet_path.stem}"

                rows.append(
                    {
                        "segment_id": seg_id,
                        "session_id": session_id,
                        "segment_index": seg_idx,
                        "metadata_path": str(mp),
                        "parquet_path": str(parquet_path),
                        "csv_path": csv_path_str,
                        "start_sample": start_sample,
                    }
                )

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.set_index("segment_id", drop=False)


    def download(self, manifest_item):
        parquet_path = Path(manifest_item.parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing parquet: {parquet_path}")

        csv_path = Path(manifest_item.csv_path) if getattr(manifest_item, "csv_path", "") else None
        if csv_path is not None and not csv_path.exists():
            csv_path = None

        metadata_path = Path(manifest_item.metadata_path)
        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        return {
            "segment_id": manifest_item.segment_id,
            "session_id": manifest_item.session_id,
            "segment_index": int(manifest_item.segment_index),
            "start_sample": int(manifest_item.start_sample),
            "meta": meta,
            "parquet_path": parquet_path,
            "csv_path": csv_path,
        }

    def process(self, download_output):
        seg_id = download_output["segment_id"]
        meta = download_output["meta"]
        pq_path = download_output["parquet_path"]
        start_sample = download_output["start_sample"]

        out_path = self.processed_dir / f"{seg_id}.h5"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not self.args.reprocess:
            return

        channels_meta = meta.get("channels", [])
        sr = next((c.get("sample_rate") for c in channels_meta if c.get("sample_rate") is not None), None)
        if sr is None:
            raise RuntimeError("No sample_rate found in metadata channels[]")
        fs = float(sr)

        df = pd.read_parquet(pq_path)

        drop_cols = {c.strip() for c in self.args.non_signal_cols.split(",") if c.strip()}
        candidate = [c.get("label") for c in channels_meta if c.get("label")]
        sig_cols = [c for c in candidate if c in df.columns and c not in drop_cols]
        if not sig_cols:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            sig_cols = [c for c in numeric_cols if c not in drop_cols]
        if not sig_cols:
            raise RuntimeError("No signal columns found")

        dtype = np.float32 if self.args.dtype == "float32" else np.float64
        X = df[sig_cols].to_numpy(dtype=dtype, copy=False)   # (n_time, n_chan)

        # session-relative start in seconds (KEEP it if you want alignment)
        domain_start = start_sample / fs

        ieeg = RegularTimeSeries(
            data=X,
            sampling_rate=fs,
            domain_start=float(domain_start),
            domain="auto",
        )

        n_ch = len(sig_cols)

        channels = ArrayDict(
            id=np.array(sig_cols, dtype=object),
            timekeys=np.array([""] * n_ch, dtype=object),  # one per channel
        )

        data = Data(
            brainset="iEEG",
            session=download_output["session_id"],
            segment=seg_id,
            ieeg=ieeg,
            channels=channels,
            domain=ieeg.domain,
        )

        with h5py.File(out_path, "w") as f:
            data.to_hdf5(f, serialize_fn_map=serialize_fn_map)

        return str(out_path)
