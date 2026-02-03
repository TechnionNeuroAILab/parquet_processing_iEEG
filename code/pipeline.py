from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import datetime as dt
import os

# Disable warnings from docker
os.environ.setdefault("RAY_DISABLE_METRICS", "1")
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

        # all metadata JSONs are directly in raw_dir
        meta_paths = sorted(raw_dir.glob("*_metadata_*.json"))

        rows = []
        for mp in meta_paths:
            with mp.open("r", encoding="utf-8") as f:
                meta = json.load(f)

            session_id = meta.get("start_timestamp") or mp.stem.replace("_metadata_", "_")
            rows.append({"session_id": session_id, "metadata_path": str(mp)})

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.set_index("session_id", drop=False)

    def download(self, manifest_item):
        metadata_path = Path(manifest_item.metadata_path)
        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        parquet_entries = meta.get("parquet_files", [])
        if not parquet_entries:
            raise RuntimeError(f"No parquet_files found in {metadata_path}")

        # Assume everything is in the SAME directory as raw_dir
        raw_dir = Path(self.raw_dir)

        segments = []
        for ent in parquet_entries:
            fname = ent["filename"]
            parquet_path = raw_dir / fname
            if not parquet_path.exists():
                raise FileNotFoundError(f"Missing parquet: {parquet_path}")

            csv_path = parquet_path.with_suffix(".csv")
            if not csv_path.exists():
                csv_path = None  # allow missing annotations

            segments.append(
                {
                    "parquet_path": parquet_path,
                    "csv_path": csv_path,
                    "start_sample": int(ent.get("start_sample", 0)),
                }
            )

        segments.sort(key=lambda x: x["start_sample"])

        return {
            "session_id": manifest_item.session_id,
            "metadata_path": metadata_path,
            "meta": meta,
            "segments": segments,
        }

    def process(self, download_output):
        session_id = download_output["session_id"]
        meta = download_output["meta"]
        segments = download_output["segments"]

        out_dir = Path(self.args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{session_id}.h5"

        if out_path.exists() and not self.args.reprocess:
            return

        # sampling rate from channel metadata (your metadata has this) 
        channels_meta = meta.get("channels", [])
        sr_candidates = [c.get("sample_rate") for c in channels_meta if c.get("sample_rate") is not None]
        if not sr_candidates:
            raise RuntimeError("No sample_rate found in metadata channels[]")
        fs = float(sr_candidates[0])

        non_signal_cols = set([c.strip() for c in self.args.non_signal_cols.split(",") if c.strip()])

        kept_channel_labels = [
            c["label"]
            for c in channels_meta
            if (not c.get("is_skipped_by_mask", False)) and (c.get("label") not in non_signal_cols)
        ]

        dtype = np.float32 if self.args.dtype == "float32" else np.float64

        signal_chunks = []
        ts_chunks = []

        ann_timestamps = []
        ann_description = []
        ann_duration = []

        self.update_status("Loading parquets + annotations")

        for seg in segments:
            paq = seg["parquet_path"]
            start_sample = seg["start_sample"]
            time_offset = start_sample / fs

            table = pq.ParquetFile(str(paq))
            schema_names = set(table.schema.names)
            time_candidates = [c for c in ["sample_index", "time"] if c in schema_names]

            if kept_channel_labels:
                sig_cols = [c for c in kept_channel_labels if c in schema_names]
            else:
                # if you must infer: read a small subset or just choose all non-time columns
                # (better: enforce kept_channel_labels via metadata)
                sig_cols = [c for c in schema_names if c not in non_signal_cols and c not in time_candidates]

            cols = sig_cols + time_candidates
            self.update_status(f"started loading {paq}")
            df = pd.read_parquet(paq, columns=cols, engine="pyarrow")
            self.update_status(f"finished loading {paq}")


            if "sample_index" in df.columns:
                sample_idx = df["sample_index"].to_numpy(dtype=np.int64, copy=False)
                t = (sample_idx / fs).astype(np.float64) + time_offset
            elif "time" in df.columns:
                t = df["time"].to_numpy(dtype=np.float64, copy=False) + time_offset
            else:
                t = (np.arange(len(df), dtype=np.float64) / fs) + time_offset

            X = df[sig_cols].to_numpy(dtype=dtype, copy=False)
            if X.ndim != 2:
                raise RuntimeError(f"Signals array not 2D for {paq}: got shape {X.shape}")

            signal_chunks.append(X)
            ts_chunks.append(t)

            csv_path = seg["csv_path"]
            if csv_path is not None:
                adf = pd.read_csv(csv_path)

                if "sample_index" in adf.columns:
                    at = (adf["sample_index"].to_numpy(dtype=np.float64) / fs) + time_offset
                elif "time_seconds" in adf.columns:
                    at = adf["time_seconds"].to_numpy(dtype=np.float64) + time_offset
                else:
                    at = None

                if at is not None and "description" in adf.columns:
                    ann_timestamps.append(at)
                    desc = adf["description"].astype(str).to_list()
                    ann_description.append(np.array([s.encode("utf-8", errors="replace") for s in desc], dtype="S"))
                    if "duration_seconds" in adf.columns:
                        ann_duration.append(adf["duration_seconds"].to_numpy(dtype=np.float64))
                    else:
                        ann_duration.append(np.full(len(at), -1.0, dtype=np.float64))

        X_all = np.concatenate(signal_chunks, axis=0) if signal_chunks else np.empty((0, 0), dtype=dtype)
        t_all = np.concatenate(ts_chunks, axis=0) if ts_chunks else np.empty((0,), dtype=np.float64)

        if t_all.size > 0:
            t0 = float(t_all[0])
            t_all = t_all - t0
        else:
            t0 = 0.0

        duration = float(t_all[-1]) if t_all.size > 0 else 0.0

        self.update_status("Building temporaldata objects")

        ieeg = RegularTimeSeries(
            sampling_rate=fs,
            timestamps=t_all,
            signal=X_all,
            domain=Interval(start=0.0, end=duration),
        )

        channels = ArrayDict(
            label=np.array(
                [c.encode("utf-8") for c in (kept_channel_labels or list(map(str, range(X_all.shape[1]))))],
                dtype="S",
            ),
        )

        annotations = None
        if ann_timestamps:
            at_all = np.concatenate(ann_timestamps, axis=0) - t0
            ad_all = np.concatenate(ann_description, axis=0)
            dur_all = np.concatenate(ann_duration, axis=0)

            annotations = IrregularTimeSeries(
                timestamps=at_all.astype(np.float64),
                description=ad_all,
                duration_seconds=dur_all.astype(np.float64),
                domain="auto",
            )

        sex_raw = (meta.get("patient_info", {}) or {}).get("sex", None)
        sex = Sex.UNKNOWN
        if isinstance(sex_raw, str):
            if sex_raw.upper().startswith("M"):
                sex = Sex.MALE
            elif sex_raw.upper().startswith("F"):
                sex = Sex.FEMALE

        start_dt_str = meta.get("start_datetime", None)
        if start_dt_str:
            try:
                recording_date = dt.datetime.fromisoformat(start_dt_str)
            except Exception:
                recording_date = dt.datetime.utcfromtimestamp(int(meta.get("start_timestamp_utc", 0)))
        else:
            recording_date = dt.datetime.utcfromtimestamp(int(meta.get("start_timestamp_utc", 0)))

        brainset_desc = BrainsetDescription(
            id=self.brainset_id,
            origin_version="0.0.0",
            derived_version="0.0.1",
            source="local_folder",
            description="Local iEEG parquet+csv session assembled from metadata-defined segments.",
        )

        subject = SubjectDescription(
            id=(meta.get("patient_info", {}) or {}).get("name", "unknown_subject"),
            species=Species.HOMO_SAPIENS,
            sex=sex,
        )

        session = SessionDescription(
            id=session_id,
            recording_date=recording_date,
        )

        device = DeviceDescription(
            id="ieeg_device",
            recording_tech=RecordingTech.IEEG,
        )

        data = Data(
            brainset=brainset_desc,
            subject=subject,
            session=session,
            device=device,
            ieeg=ieeg,
            channels=channels,
            annotations=annotations,
            domain="auto",
        )

        data.set_train_domain(data.domain)

        self.update_status(f"Writing HDF5 -> {out_path}")
        with h5py.File(out_path, "w") as f:
            data.to_hdf5(f, serialize_fn_map=serialize_fn_map)
