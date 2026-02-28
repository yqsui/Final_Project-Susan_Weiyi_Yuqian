import pandas as pd
from pathlib import Path

### Path
script_dir = Path(__file__).parent

raw_dir = script_dir / "../data/raw-data"
out_dir = script_dir / "../data/derived-data"
out_dir.mkdir(parents=True, exist_ok=True)

### Load Data
rent_burden_path = raw_dir / "rent_burden_B25070.csv"
median_income_path = raw_dir / "median_household_income_B19013.csv"

### Standard Functions
# Split lebel with data
def split_label_and_data(df: pd.DataFrame):
    labels = df.iloc[0].copy()
    data = df.iloc[1:].copy().reset_index(drop=True)
    return labels, data

# Clean GEOID
def add_clean_geoid(df: pd.DataFrame, geo_col: str = "GEO_ID") -> pd.DataFrame:
    df["GEOID"] = (
        df[geo_col]
        .astype(str)
        .str.replace("1400000US", "", regex=False)
    )
    return df

# Standardize a series to z-scores
def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)

### Economic Pressure1: Rent Burden
rent_burden = pd.read_csv(rent_burden_path)
labels, rent_burden = split_label_and_data(rent_burden)
print(rent_burden.columns)
print("B25070_001E label =", labels["B25070_001E"])