from pathlib import Path
import glob
path = r"/root/capsule/data/parquet_iEEG/Patient1/iEEG"
path = Path(path)
print(path.glob("*_metadata_*.json"))
# brainsets config
# /root/capsule/data/parquet_iEEG/Patient1
# /root/capsule/results
# brainsets prepare /root/capsule/code --use-active-env --local -c 1
# export RAY_TMPDIR=/root/capsule/results/ray_tmp
