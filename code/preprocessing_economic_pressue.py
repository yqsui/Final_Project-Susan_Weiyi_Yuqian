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
    df[geo_col] = (
        df[geo_col]
        .astype(str)
        .str.strip()
        .str[-11:]
    )
    df = df.rename(columns={geo_col: "GEOID"})
    
    return df

# Change to numerics
def convert_numeric_except_geoid(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c != "GEOID"]
    df[numeric_cols] = df[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    return df

# Standardize a series to z-scores
def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)

### Economic Pressure 1: Rent Burden
rent_burden = pd.read_csv(rent_burden_path)
labels, rent_burden = split_label_and_data(rent_burden)
rent_burden = add_clean_geoid(rent_burden)

print(rent_burden.columns)
print("B25070_001E label =", labels["B25070_001E"])

rename_rent_burden = {
    col: str(labels[col]).replace("Estimate", "").replace("Total:", "").strip()
    for col in labels.index
    if col in rent_burden.columns
}

rent_burden = rent_burden.rename(columns=rename_rent_burden)
rent_burden = rent_burden.loc[
    :, ~rent_burden.columns.str.startswith("M")
]
rent_burden = rent_burden.dropna(axis=1, how="all")
rent_burden.columns = rent_burden.columns.str.replace(r"^!+", "", regex=True)
rent_burden = rent_burden.rename(
    columns={rent_burden.columns[2]: "total"}
)
rent_burden = rent_burden.drop(rent_burden.columns[1], axis=1)
rent_burden["GEOID"] = rent_burden["GEOID"].str[-11:]
rent_burden

# Standarize
rent_burden = convert_numeric_except_geoid(rent_burden)

rent_burden["rb_30plus_share"] = (
    rent_burden["30.0 to 34.9 percent"]
    + rent_burden["35.0 to 39.9 percent"]
    + rent_burden["40.0 to 49.9 percent"]
    + rent_burden["50.0 percent or more"]
)

rent_burden["rb_30plus_z"] = zscore(
    rent_burden["rb_30plus_share"]
)

print(rent_burden["rb_30plus_z"].describe()) #check if correctly standarized

### Economic Pressure 2: Median Income
median_income = pd.read_csv(median_income_path)
labels, median_income = split_label_and_data(median_income)
median_income = add_clean_geoid(median_income)

rename_median_income = {
    col: str(labels[col]).replace("Estimate", "").replace("Total:", "").strip()
    for col in labels.index
    if col in median_income.columns
}

median_income = median_income.rename(columns=rename_median_income)
median_income = median_income.loc[
    :, ~median_income.columns.str.startswith("M")
]
median_income = median_income.dropna(axis=1, how="all")
median_income.columns = median_income.columns.str.replace(r"^!+", "", regex=True)
median_income = median_income.rename(
    columns={median_income.columns[2]: "median_income_usd"}
)
median_income = median_income.drop(median_income.columns[1], axis=1)
median_income["GEOID"] = median_income["GEOID"].str[-11:]

# Standarize
median_income = convert_numeric_except_geoid(median_income)

median_income["income_z"] = zscore(median_income["median_income_usd"])

median_income["income_z_rev"] = -median_income["income_z"]

print(median_income[["median_income_usd", "income_z", "income_z_rev"]].describe())
print("NaN income rows:", median_income["median_income_usd"].isna().sum())

### Merge into a "Economic Pressure" DataFrame
economic_df = rent_burden.merge(
    median_income[["GEOID", "median_income_usd", "income_z_rev"]],
    on="GEOID",
    how="left",
    indicator=True
)

print(economic_df["_merge"].value_counts())

# check no income and NaN
no_income = economic_df[
    economic_df["median_income_usd"].isna()
]
print("Number of tracts with missing income:", len(no_income))
print(no_income[["GEOID", "median_income_usd"]].head(20))
nan_key = economic_df[
    economic_df[["rb_30plus_z", "income_z_rev"]].isna().any(axis=1)
]

print("Rows with NaN in key vars:", len(nan_key))
print(nan_key[["GEOID", "rb_30plus_z", "income_z_rev"]].head(20))

# Clean economic_df
economic_df_clean = economic_df.dropna(
    subset=["rb_30plus_z", "income_z_rev"]
)

print("Before drop:", len(economic_df_clean))
print("After drop:", len(economic_df_clean))

# Final economic pressue index
economic_df_clean["economic_pressure_index"] = (
    economic_df_clean["rb_30plus_z"]
    + economic_df_clean["income_z_rev"]
)

print(economic_df_clean["economic_pressure_index"].describe())

economic_df_final = economic_df_clean[
    [
        "GEOID",
        "rb_30plus_z",
        "income_z_rev",
        "economic_pressure_index"
    ]
].copy()

print(economic_df_final.head())
print("Final rows:", len(economic_df_final))
print("Final columns:", economic_df_final.columns.tolist())

# save as cvs
output_path = out_dir / "economic_pressure.csv"
economic_df_final.to_csv(output_path, index=False)
print("Saved to:", output_path)