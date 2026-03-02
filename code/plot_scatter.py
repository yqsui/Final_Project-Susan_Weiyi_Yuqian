import pandas as pd
import altair as alt
from pathlib import Path
import vl_convert as vlc

# Path
script_dir = Path(__file__).parent

out_dir = script_dir / "../data/derived-data"
out_dir.mkdir(parents=True, exist_ok=True)

# Load data
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

# Plot
alt.data_transformers.disable_max_rows()

# Plot A: economic_pressure_index vs. avg_monthly_total_violation_rate
econ_points = alt.Chart(merged).mark_circle(size=25, opacity=0.6, color="#4F8CC2").encode(
    alt.X("economic_pressure_index:Q", title="Economic Pressure Index"),
    alt.Y("avg_monthly_total_violation_rate:Q", title="Average Monthly Total Violation Rate")
).properties(width=400, height=400)

econ_reg = alt.Chart(merged).transform_regression(
    "economic_pressure_index", "avg_monthly_total_violation_rate"
).mark_line(color="#9B2F2F", strokeWidth=2).encode(
    alt.X("economic_pressure_index:Q"),
    alt.Y("avg_monthly_total_violation_rate:Q")
)

chart_econ = alt.layer(econ_points, econ_reg).properties(
    title={"text": "Economic Pressure vs Building Violation Rate", "fontSize":14, "fontWeight":"bold"}
)

# Plot B: housing_structural_index vs. avg_monthly_total_violation_rate
housing_points = alt.Chart(merged).mark_circle(size=25, opacity=0.6, color="#E39920").encode(
    alt.X("housing_structural_index:Q", title="Housing Structural Index"),
    alt.Y("avg_monthly_total_violation_rate:Q", title="Average Monthly Total Violation Rate")
).properties(width=400, height=400)

housing_reg = alt.Chart(merged).transform_regression(
    "housing_structural_index", "avg_monthly_total_violation_rate"
).mark_line(color="#BE2121", strokeWidth=2).encode(
    alt.X("housing_structural_index:Q"),
    alt.Y("avg_monthly_total_violation_rate:Q")
)

chart_housing = alt.layer(housing_points, housing_reg).properties(
    title={"text": "Housing Structural Features vs Building Violation Rate", "fontSize":14, "fontWeight":"bold"}
)

# Integration
final_chart = (chart_econ | chart_housing).configure_axis(
    labelFontSize=12, titleFontSize=12, gridOpacity=0.3
)
final_chart.show()

# Save as png
chart_path = out_dir / "scatter_plot.png"
try:
    png_data = vlc.vegalite_to_png(final_chart.to_dict(), scale=2)
    with open(chart_path, "wb") as f:
        f.write(png_data)
    print(f"PNG has saved to: {chart_path}")
except ImportError:
    print("Need to install vl-convert.")
except Exception as e:
    print(f"Failed to save as PNG: {e}")