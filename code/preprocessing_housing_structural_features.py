import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from shapely import wkt

### Path
script_dir = Path(__file__).parent

raw_dir = script_dir / "../data/raw-data"
out_dir = script_dir / "../data/derived-data"
out_dir.mkdir(parents=True, exist_ok=True)

### Load Data
poverty_status_path = raw_dir / ("poverty_status_B17001.csv")
tenure_owner_renter_path = raw_dir / ("tenure_owner_renter_B25003.csv")
units_in_structure_path = raw_dir / ("units_in_structure_B25024.csv")

### Standard Functions
# Split label with data
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


### Housing Structural Features 1: Poverty Rate
poverty_status = pd.read_csv(poverty_status_path)
labels, poverty_status = split_label_and_data(poverty_status)
poverty_status = add_clean_geoid(poverty_status)

poverty_status.rename(
    columns={
        "B17001_001E": "total_population",
        "B17001_002E": "poverty_population"
    },
    inplace=True
)

rename_poverty_status = {
    col: str(labels[col]).replace("Estimate", "").replace("Total:", "").strip()
    for col in labels.index
    if col in poverty_status.columns
}
poverty_status = poverty_status.rename(columns=rename_poverty_status)

poverty_status = poverty_status.loc[
    :, ~poverty_status.columns.str.startswith("M")
]
poverty_status = poverty_status.dropna(axis=1, how="all")
poverty_status.columns = poverty_status.columns.str.replace(r"^!+", "", regex=True)

core_cols = ["GEOID", "total_population", "poverty_population"]
poverty_status = poverty_status[core_cols].copy()

poverty_status = convert_numeric_except_geoid(poverty_status)

poverty_status = poverty_status[
    (poverty_status["total_population"] > 0) &
    (poverty_status["total_population"].notna()) &
    (poverty_status["poverty_population"].notna())
].reset_index(drop=True)

poverty_status["poverty_rate"] = round(
    (poverty_status["poverty_population"] / poverty_status["total_population"]) * 100, 2
)
poverty_status["poverty_rate_z"] = zscore(poverty_status["poverty_rate"])
poverty_df = poverty_status[["GEOID", "poverty_rate", "poverty_rate_z"]].copy()

### Housing Structural Features 2: Tenure Owner Renter
tenure_owner_renter = pd.read_csv(tenure_owner_renter_path)
labels, tenure_owner_renter = split_label_and_data(tenure_owner_renter)
tenure_owner_renter = add_clean_geoid(tenure_owner_renter)

tenure_owner_renter.rename(
    columns={
        "B25003_001E": "total_housing",
        "B25003_002E": "owner_occupied",
        "B25003_003E": "renter_occupied"
    },
    inplace=True
)

rename_tenure_owner_renter = {
    col: str(labels[col]).replace("Estimate", "").replace("Total:", "").strip()
    for col in labels.index
    if col in tenure_owner_renter.columns
}
tenure_owner_renter = tenure_owner_renter.rename(columns=rename_tenure_owner_renter)

tenure_owner_renter = tenure_owner_renter.loc[
    :, ~tenure_owner_renter.columns.str.startswith("M")
]
tenure_owner_renter = tenure_owner_renter.dropna(axis=1, how="all")
tenure_owner_renter.columns = tenure_owner_renter.columns.str.replace(r"^!+", "", regex=True)

core_cols = ["GEOID", "total_housing", "owner_occupied", "renter_occupied"]
tenure_owner_renter = tenure_owner_renter[core_cols].copy()

tenure_owner_renter = convert_numeric_except_geoid(tenure_owner_renter)

tenure_owner_renter = tenure_owner_renter[
    (tenure_owner_renter["total_housing"] > 0) &
    (tenure_owner_renter["total_housing"].notna()) &
    (tenure_owner_renter["owner_occupied"].notna()) &
    (tenure_owner_renter["renter_occupied"].notna())
].reset_index(drop=True)

tenure_owner_renter["renter_rate"] = round(
    (tenure_owner_renter["renter_occupied"] / tenure_owner_renter["total_housing"]) * 100, 2
)
tenure_owner_renter["renter_rate_z"] = zscore(tenure_owner_renter["renter_rate"])
tenure_df = tenure_owner_renter[["GEOID", "renter_rate", "renter_rate_z"]].copy()

### Housing Structural Features 3: Units in Structure
units_in_structure = pd.read_csv(units_in_structure_path)
labels, units_in_structure = split_label_and_data(units_in_structure)
units_in_structure = add_clean_geoid(units_in_structure)

units_in_structure.rename(
    columns={
        "B25034_001E": "total_buildings",
        "B25034_002E": "built_2020_later",
        "B25034_003E": "built_2010_2019",
        "B25034_004E": "built_2000_2009",
        "B25034_005E": "built_1990_1999",
        "B25034_006E": "built_1980_1989",
        "B25034_007E": "built_1970_1979",
        "B25034_008E": "built_1960_1969",
        "B25034_009E": "built_1950_1959",
        "B25034_010E": "built_1940_1949",
        "B25034_011E": "built_1939_earlier"
    },
    inplace=True
)

rename_units_in_structure = {
    col: str(labels[col]).replace("Estimate", "").replace("Total:", "").strip()
    for col in labels.index
    if col in units_in_structure.columns
}
units_in_structure = units_in_structure.rename(columns=rename_units_in_structure)

units_in_structure = units_in_structure.loc[
    :, ~units_in_structure.columns.str.startswith("M")
]
units_in_structure = units_in_structure.dropna(axis=1, how="all")
units_in_structure.columns = units_in_structure.columns.str.replace(r"^!+", "", regex=True)

core_cols = [
    "GEOID", "total_buildings", 
    "built_2020_later", "built_2010_2019", "built_2000_2009", "built_1990_1999",
    "built_1980_1989", "built_1970_1979", "built_1960_1969", "built_1950_1959",
    "built_1940_1949", "built_1939_earlier"
]
units_in_structure = units_in_structure[core_cols].copy()

units_in_structure = convert_numeric_except_geoid(units_in_structure)

units_in_structure = units_in_structure[
    (units_in_structure["total_buildings"] > 0) &
    (units_in_structure["total_buildings"].notna())
].reset_index(drop=True)

units_in_structure["old_building_rate"] = round(
    (units_in_structure["built_1939_earlier"] / units_in_structure["total_buildings"]) * 100, 2
)
units_in_structure["old_building_rate_z"] = zscore(units_in_structure["old_building_rate"])
building_df = units_in_structure[["GEOID", "old_building_rate", "old_building_rate_z"]].copy()


### Merge into a "Housing Structural Features" DataFrame
housing_df = poverty_df.merge(
    tenure_df[["GEOID", "renter_rate", "renter_rate_z"]],
    on="GEOID",
    how="left",
    indicator="merge_tenure"
)

housing_df = housing_df.merge(
    building_df[["GEOID", "old_building_rate", "old_building_rate_z"]],
    on="GEOID",
    how="left",
    indicator="merge_building"
)

print("Merge tenure result:")
print(housing_df["merge_tenure"].value_counts())
print("\nMerge building result:")
print(housing_df["merge_building"].value_counts())

# Check no tenure, no building and NaN
no_tenure = housing_df[housing_df["renter_rate"].isna()]
print(f"\nTracts with missing tenure data: {len(no_tenure)}")
print(no_tenure[["GEOID", "renter_rate"]].head(10))

no_building = housing_df[housing_df["old_building_rate"].isna()]
print(f"\nTracts with missing building data: {len(no_building)}")
print(no_building[["GEOID", "old_building_rate"]].head(10))

nan_key = housing_df[
    housing_df[["poverty_rate_z", "renter_rate_z", "old_building_rate_z"]].isna().any(axis=1)
]
print(f"\nRows with NaN in key vars: {len(nan_key)}")
print(nan_key[["GEOID", "poverty_rate_z", "renter_rate_z", "old_building_rate_z"]].head(10))

# Clean housing_df
housing_df_clean = housing_df.dropna(
    subset=["poverty_rate_z", "renter_rate_z", "old_building_rate_z"]
)

print(f"\nBefore drop NaN: {len(housing_df)}")
print(f"After drop NaN: {len(housing_df_clean)}")

# Final housing structural index
housing_df_clean["housing_structural_index"] = (
    housing_df_clean["poverty_rate_z"] + 
    housing_df_clean["renter_rate_z"] + 
    housing_df_clean["old_building_rate_z"]
)

print("\nHousing Structural Index stats:")
print(housing_df_clean["housing_structural_index"].describe())

housing_df_final = housing_df_clean[
    [
        "GEOID",
        "poverty_rate",
        "poverty_rate_z",
        "renter_rate",
        "renter_rate_z",
        "old_building_rate",
        "old_building_rate_z",
        "housing_structural_index"
    ]
].copy()

print("\nFinal Housing Structural Features DataFrame:")
print(housing_df_final.head(10))
print(f"Final rows: {len(housing_df_final)}")
print(f"Final columns: {housing_df_final.columns.tolist()}")

# Save as csv
output_path = out_dir / "housing_structural_features.csv"
housing_df_final.to_csv(output_path, index=False, encoding="utf-8")
print(f"\nSaved to: {output_path}")