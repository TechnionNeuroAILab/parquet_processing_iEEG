import pandas as pd
import re
from typing import Tuple, List
import pyarrow.parquet as pq

# =============================================================================
# CONFIGURATION
# =============================================================================
# Set to True to load the entire file, False to just peek at the first 100 rows.
FULL_LOAD: bool = False 

# Ensure pandas prints ALL columns when calling head()
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Paths (Update these for your local environment)
PARQUET_PATH = r"/data/parquet_iEEG/Patient1/iEEG/Patient1_ieeg_20251126_042728.parquet"
ANNOT_PATH = r"/data/parquet_iEEG/Patient1/iEEG/Patient1_ieeg_20251126_042728.csv"

# =============================================================================
# EXPECTED DATA STRUCTURES
# =============================================================================
# Data Parquet (Wide Format):
#   - sample_index: int (Unique sample identifier)
#   - absolute_timestamp: datetime/utc (Wall-clock time)
#   - P<elec>-<chan>: float (e.g., "P1-1", "P1-2" -> actual iEEG signal values)
#
# Annotations CSV:
#   - sample_index: int (Matches the sample_index in the Parquet)
#   - description: str (The event label, e.g., "Note On", "Button Press")
#   - time_seconds: float (Offset from recording start)
# =============================================================================

def separate_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Separates columns into iEEG signals (P-series) and Metadata.
    """
    signal_regex = r"^[pP]\d+-\d+$"
    
    signals = [c for c in df.columns if re.match(signal_regex, str(c))]
    metadata = [c for c in df.columns if c not in signals]
    
    return signals, metadata

def parse_channel_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies 'P<elec>-<chan>' columns and adds 'electrode' and 'channel'
    mapping info if needed, or simply helps you identify them.
    """
    regex = r"^[pP](?P<electrode>\d+)-(?P<channel>\d+)$"
    channel_cols = [c for c in df.columns if re.match(regex, str(c))]

    print(f"Found {len(channel_cols)} signal channels.")
    return channel_cols


def load_session_data(parquet_path: str, annotations_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads both files into pandas DataFrames.
    """
    # 1. Load Data
    print(f"Loading data from: {parquet_path}")
    df_data = pd.read_parquet(parquet_path)

    # 2. Load Annotations
    print(f"Loading annotations from: {annotations_path}")
    # Using simple read_csv; ensures sample_index is integer for merging
    df_ann = pd.read_csv(annotations_path)
    df_ann['sample_index'] = df_ann['sample_index'].astype(int)

    return df_data, df_ann


def get_tidy_channels(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Optional: Breaks the wide 'P1-1' format into a long 'tidy' format.
    Useful for seaborn plotting or grouped analysis.
    """
    id_vars = ['sample_index', 'absolute_timestamp']
    # Filter for P-style columns
    channel_cols = [c for c in df_wide.columns if re.match(r"^[pP]\d+-\d+$", str(c))]

    df_long = df_wide.melt(
        id_vars=id_vars,
        value_vars=channel_cols,
        var_name="channel_label",
        value_name="voltage"
    )

    # Extract electrode and channel integers
    extracted = df_long["channel_label"].str.extract(r"^[pP](?P<elec>\d+)-(?P<chan>\d+)")
    df_long["electrode"] = extracted["elec"].astype(int)
    df_long["channel"] = extracted["chan"].astype(int)

    return df_long

def load_and_inspect():
    """
    Main loader function with column categorization and inspection.
    """
    print(f"--- Loading {'FULL' if FULL_LOAD else 'PREVIEW'} data ---")
    
    # 1. Load Data
    if FULL_LOAD:
        df_data, df_ann = load_session_data(PARQUET_PATH, ANNOT_PATH)
    else:
        # read_parquet doesn't have nrows, so we use a temporary pyarrow table head
        table = pq.read_table(PARQUET_PATH).slice(0, 100)
        df_data = table.to_pandas()
        df_ann = pd.read_csv(ANNOT_PATH)

    # 2. Categorize Columns
    signals, metadata = separate_columns(df_data)

    # 3. Print Summary
    print(f"\n[FILE]: {PARQUET_PATH}")
    print(f"[SHAPE]: {df_data.shape[0]} rows x {df_data.shape[1]} columns")
    
    print("\n--- NON-SIGNAL COLUMNS (Metadata) ---")
    print(", ".join(metadata))
    
    print(f"\n--- SIGNAL COLUMNS ({len(signals)} detected) ---")
    if len(signals) > 10:
        print(f"{', '.join(signals[:5])} ... {', '.join(signals[-5:])}")
    else:
        print(", ".join(signals))

    print("\n--- DATA HEAD (All columns visible) ---")
    print(df_data.head())
    
    print("\n--- ANNOTATIONS HEAD ---")
    print(df_ann.head())

    return df_data, df_ann

def get_electrode_mapping(signals: List[str]):
    """
    Breakdown of which electrodes have which contacts.
    """
    mapping = {}
    for s in signals:
        match = re.match(r"^[pP](?P<elec>\d+)-(?P<chan>\d+)", s)
        if match:
            e = int(match.group('elec'))
            c = int(match.group('chan'))
            mapping.setdefault(e, []).append(c)
    
    print("\n--- ELECTRODE INVENTORY ---")
    for e in sorted(mapping.keys()):
        contacts = sorted(mapping[e])
        print(f"Electrode {e}: {len(contacts)} contacts ( {contacts[0]}-{contacts[-1]} )")

if __name__ == "__main__":
    data, annots = load_and_inspect()
    
    # Just to see the electrode breakdown
    sig_cols, _ = separate_columns(data)
    get_electrode_mapping(sig_cols)
