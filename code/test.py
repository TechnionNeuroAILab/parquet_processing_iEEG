from pathlib import Path
import glob
import ray.dashboard
path = r"/root/capsule/data/parquet_iEEG/Patient1/iEEG"
path = Path(path)
print(path.glob("*_metadata_*.json"))
# brainsets config
# /root/capsule/data/parquet_iEEG/Patient1
# /root/capsule/results

# export RAY_DISABLE_METRICS=1
# export RAY_DISABLE_DASHBOARD=1
# export RAY_TMPDIR=/root/capsule/results/ray_tmp

# brainsets prepare /root/capsule/code --use-active-env --local -c 1
