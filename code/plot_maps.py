import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import geopandas as gpd
import numpy as np

### Path
script_dir = Path(__file__).parent

raw_dir = script_dir / "../data/raw-data"
out_dir = script_dir / "../data/derived-data"
out_dir.mkdir(parents=True, exist_ok=True)

### Load data
violation_path = out_dir / "building_violation_by_code_rate.csv"
economic_path = out_dir / "economic_pressure.csv"
housing_path = out_dir / "housing_structural_features.csv"

violation = pd.read_csv(violation_path)
economic = pd.read_csv(economic_path)
housing = pd.read_csv(housing_path)

# Sum up by GEOID+YEAR_MONTH (the total violation rate for the same area and the same month)
violation_month_sum = violation.groupby(["GEOID", "YEAR_MONTH"])["VIOL_PER_1000_BUILDINGS"].sum().reset_index()
violation_month_sum.rename(columns={"VIOL_PER_1000_BUILDINGS": "monthly_total_violation_rate"}, inplace=True)
violation_month_sum.head()

# Calculate the average by GEOID (the average monthly total violation rate for all months in the same area)
violation_agg = violation_month_sum.groupby("GEOID")["monthly_total_violation_rate"].mean().reset_index()
violation_agg.rename(columns={"monthly_total_violation_rate": "avg_monthly_total_violation_rate"}, inplace=True)
violation_agg.head()

# Merge
economic_needed = economic[["GEOID", "economic_pressure_index"]]
housing_needed = housing[["GEOID", "housing_structural_index"]]

merged = violation_agg.merge(economic_needed, on="GEOID", how="inner")
merged = merged.merge(housing_needed, on="GEOID", how="inner")

print(f"Number of valid data rows after merging: {len(merged)}")
print(f"NA check:\n{merged.isna().sum()}")

### Static 1: Geographic Overlap

# Step1: Independent Geographic spread for violation, economic pressure, and structural vulnerability

# Merge with shapefile into gpd
tracts = gpd.read_file(raw_dir/"tl_2024_17_tract.shp")

tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)
merged["GEOID"] = merged["GEOID"].astype(str).str.zfill(11)

gdf = tracts.merge(merged, on="GEOID", how="inner")

gdf = gdf.to_crs(epsg=3857)

print(gdf.shape)
gdf.head()

# Conduct log for violation because it is is too far to the right tail
gdf["log_violation"] = np.log1p(gdf["avg_monthly_total_violation_rate"])

vmin_index = -3
vmax_index = 3

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Econ
gdf.plot(
    column="economic_pressure_index",
    cmap="OrRd",
    linewidth=0.1,
    edgecolor="white",
    legend=True,
    ax=axes[0]
)
axes[0].set_title("Economic Pressure Index")
axes[0].axis("off")

# Structural
gdf.plot(
    column="housing_structural_index",
    cmap="OrRd",
    linewidth=0.1,
    edgecolor="white",
    legend=True,
    ax=axes[1]
)
axes[1].set_title("Structural Feature Index")
axes[1].axis("off")

# Violation (log)
gdf.plot(
    column="log_violation",
    cmap="Reds",
    linewidth=0.1,
    edgecolor="white",
    legend=True,
    ax=axes[2]
)
axes[2].set_title("Log Violation Rate")
axes[2].axis("off")

chart_path = out_dir / "part1_distribution.png"
fig.savefig(chart_path, dpi=300, bbox_inches="tight")

print(f"Distribution plot saved to：{chart_path}")

plt.tight_layout()

plt.show()

## Step2: Top 25% Overlap

# Calculate top 25% thresholds
q_econ = gdf["economic_pressure_index"].quantile(0.75)
q_struct = gdf["housing_structural_index"].quantile(0.75)
q_vio = gdf["avg_monthly_total_violation_rate"].quantile(0.75)

# Create indicator columns
gdf["top_econ"] = gdf["economic_pressure_index"] >= q_econ
gdf["top_struct"] = gdf["housing_structural_index"] >= q_struct
gdf["top_vio"] = gdf["avg_monthly_total_violation_rate"] >= q_vio

# Count overlap (0-3)
gdf["overlap_count"] = (
    gdf["top_econ"].astype(int) +
    gdf["top_struct"].astype(int) +
    gdf["top_vio"].astype(int)
)

# Identify triple burden tracts
gdf["triple_hotspot"] = gdf["overlap_count"] == 3

# Overlap Map
fig, ax = plt.subplots(1, 1, figsize=(9, 9))

gdf["overlap_score_cat"] = gdf["overlap_count"].astype("category")

gdf.plot(
    column="overlap_score_cat",
    cmap="Reds",
    linewidth=0.05,
    edgecolor="white",
    legend=True,
    ax=ax
)

ax.set_title(
    "Cumulative Overlap of Top 25% Risk Factors\n(Economic + Structural + Violations)"
)
ax.axis("off")
overlap_path = out_dir / "part2_overlap.png"
fig.savefig(overlap_path, dpi=300, bbox_inches="tight")
print(f"Overlap map saved to: {overlap_path}")
plt.show()

# Triple Hotspo Map
fig, ax = plt.subplots(1, 1, figsize=(9, 9))

gdf.plot(color="#eeeeee", linewidth=0.05, edgecolor="white", ax=ax)

gdf[gdf["triple_hotspot"]].plot(color="red", linewidth=0.2, edgecolor="white", ax=ax)

ax.set_title("Triple-Burden Hotspots")
ax.axis("off")
triple_path = out_dir / "part2_highlight.png"
fig.savefig(triple_path, dpi=300, bbox_inches="tight")
print(f"Highlight map saved to: {triple_path}")

plt.show()
plt.show()

# Calculate Relative Risk
group_means = gdf.groupby("overlap_count")["avg_monthly_total_violation_rate"].mean()

print(group_means)

mean_0 = group_means.loc[0]
mean_3 = group_means.loc[3]

risk_ratio = mean_3 / mean_0

print("Relative Risk (Triple vs Low):", round(risk_ratio, 2))