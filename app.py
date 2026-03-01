import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk
import numpy as np
from pathlib import Path
st.set_page_config(page_title="Chicago Building Violations Dashboard", layout="wide")


# set paths & load tract shapefile + violation rate dataset
script_dir=Path(__file__).parent
raw_dir=script_dir / "../data/raw-data"
out_dir=script_dir / "../data/derived-data"

TRACT_SHP=raw_dir / "tl_2024_17_tract.shp"
RATE_PATH=out_dir / "building_violation_by_code_rate.csv"
ECON_PATH=out_dir / "economic_pressure.csv"
STRUCT_PATH=out_dir / "housing_structural_features.csv"

econ=pd.read_csv(ECON_PATH)
econ["GEOID"]=econ["GEOID"].astype(str).str.zfill(11)
econ=econ[["GEOID","economic_pressure_index"]].copy()

struct=pd.read_csv(STRUCT_PATH)
struct["GEOID"]=struct["GEOID"].astype(str).str.zfill(11)
struct=struct[["GEOID","housing_structural_index"]].copy()

tracts=gpd.read_file(TRACT_SHP)[["GEOID", "geometry"]].copy()
tracts["GEOID"]=tracts["GEOID"].astype(str).str.zfill(11)

df=pd.read_csv(RATE_PATH)
df["GEOID"]=df["GEOID"].astype(str).str.zfill(11)
df["YEAR_MONTH_DT"]=pd.to_datetime(df["YEAR_MONTH_DT"], errors="coerce")

need_cols=[
    "GEOID", "YEAR_MONTH_DT", "YEAR_MONTH_STR",
    "VIOLATION CODE", "VIOLATION_DESCRIPTION",
    "VIOLATION_COUNT", "total_buildings", "VIOL_PER_1000_BUILDINGS"
]
df=df[[c for c in need_cols if c in df.columns]].copy()


# restrict tracts using GEOIDs shared across Chicago tract-level tables
tracts=tracts.to_crs("EPSG:4326")

df_geoids=set(df["GEOID"].dropna())
econ_geoids=set(econ["GEOID"].dropna())
struct_geoids=set(struct["GEOID"].dropna())
keep_geoids=df_geoids&econ_geoids&struct_geoids

tracts=tracts[tracts["GEOID"].isin(keep_geoids)].copy()


# sidebar controls
st.sidebar.header("Filters")

valid_months=(df["YEAR_MONTH_DT"]
                .dropna().drop_duplicates().sort_values().to_list())
if len(valid_months)==0:
    st.error("No valid YEAR_MONTH_DT found in the dataset.")
    st.stop()

month_dt=st.sidebar.select_slider(
    "Select month",
    options=valid_months,
    value=valid_months[0],
    format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m")
)

code_list=sorted(df["VIOLATION CODE"].dropna().unique().tolist())
code_choice=st.sidebar.selectbox("Select violation code", ["(All codes)"] + code_list)

# ── toggle between raw rate and relative-to-city-average ──
relative_mode=st.sidebar.toggle("Relative to city average", value=False,
                                  help="ON → rate / city_mean_rate  |  OFF → raw rate per 1,000 units")

show_top_codes=st.sidebar.checkbox("Show top codes table", value=True)


# filter data for selected month 
month_df=df[df["YEAR_MONTH_DT"] == pd.Timestamp(month_dt)].copy()
if code_choice!="(All codes)":
    month_df=month_df[month_df["VIOLATION CODE"] == code_choice].copy()


# aggregate to tract-level
tract_agg=(month_df.groupby("GEOID", as_index=False)
             .agg(VIOLATION_COUNT=("VIOLATION_COUNT", "sum"),
                  RATE=("VIOL_PER_1000_BUILDINGS", "sum")))

# ── compute display value ──
city_mean=tract_agg["RATE"].mean() if len(tract_agg) > 0 else 1.0
if city_mean==0:
    city_mean=1.0

if relative_mode:
    tract_agg["DISPLAY_VAL"]=(tract_agg["RATE"] / city_mean).round(3)
    map_label="Rate / City Mean"
else:
    tract_agg["DISPLAY_VAL"]=tract_agg["RATE"].round(2)
    map_label="Rate per 1,000 units"


# merge onto tract polygons
map_gdf=tracts.merge(tract_agg,on="GEOID",how="left")
map_gdf["VIOLATION_COUNT"]=map_gdf["VIOLATION_COUNT"].fillna(0).astype(int)
map_gdf["RATE"]=map_gdf["RATE"].fillna(0.0)
map_gdf["DISPLAY_VAL"]=map_gdf["DISPLAY_VAL"].fillna(0.0)

map_gdf=map_gdf.merge(econ,on="GEOID",how="left")
map_gdf=map_gdf.merge(struct,on="GEOID",how="left")

map_gdf["economic_pressure_index"]=map_gdf["economic_pressure_index"].fillna("N/A")
map_gdf["housing_structural_index"]=map_gdf["housing_structural_index"].fillna("N/A")

for col in ["economic_pressure_index","housing_structural_index"]:
    map_gdf[col]=map_gdf[col].apply(
        lambda v: round(float(v),3) if v!="N/A" else "N/A"
    )


# headline metrics
col1, col2, col3=st.columns(3)
col1.metric("Month",pd.Timestamp(month_dt).strftime("%Y-%m"))
col2.metric("Selected code",code_choice)
col3.metric("Total violations (count)", int(map_gdf["VIOLATION_COUNT"].sum()))

mode_txt = "relative to city average (ratio)" if relative_mode else "raw rate per 1,000 housing units"
st.caption(f"Map shows **{mode_txt}** (darker=higher).")


# choropleth map using f-string expression 
map_gdf_ll=map_gdf.to_crs("EPSG:4326").copy()
map_gdf_ll["DISPLAY_VAL"]=pd.to_numeric(map_gdf_ll["DISPLAY_VAL"], errors="coerce").fillna(0.0)

if relative_mode:
    vmax=float(map_gdf_ll["DISPLAY_VAL"].quantile(0.95))
    if vmax<=1:
        vmax=1.5

    map_gdf_ll["fill_alpha"]=(
        (map_gdf_ll["DISPLAY_VAL"]-1)
        .clip(lower=0, upper=vmax-1)
        .div(vmax-1)
        .mul(235)
        .add(20)
        .clip(20, 255)
        .astype(int)
    )
else:
    vmax=float(map_gdf_ll["DISPLAY_VAL"].quantile(0.95))
    if vmax<= 0:
        vmax=1.0

    map_gdf_ll["fill_alpha"]=(
        map_gdf_ll["DISPLAY_VAL"]
        .clip(upper=vmax)
        .div(vmax)
        .mul(235)
        .add(20)
        .clip(20, 255)
        .astype(int)
    )

layer=pdk.Layer(
    "GeoJsonLayer",
    data=map_gdf_ll.__geo_interface__,
    pickable=True,
    stroked=True,
    filled=True,
    extruded=False,
    get_line_color=[255, 255, 255, 60],
    line_width_min_pixels=0.5,
    get_fill_color="[255, 80, 0, properties.fill_alpha]",
)

view_state = pdk.ViewState(
    latitude=41.88, longitude=-87.63,
    zoom=10, min_zoom=8, max_zoom=14,
)

tooltip={
    "html": (
        "<b>Tract GEOID:</b> {GEOID}<br/>"
        f"<b>{map_label}:</b> {{DISPLAY_VAL}}<br/>"
        "<b>Violation count:</b> {VIOLATION_COUNT}<br/>"
        "<b>Econ pressure index:</b> {economic_pressure_index}<br/>"
        "<b>Housing structural index:</b> {housing_structural_index}"
    ),
    "style": {"backgroundColor": "white", "color": "black"}
}

st.subheader("Map")
st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="light",  
        tooltip=tooltip,
    ),
    use_container_width=True
)

# final tables
st.subheader("Top Tracts")
top_tracts=(map_gdf[["GEOID", "DISPLAY_VAL", "VIOLATION_COUNT",
                         "economic_pressure_index", "housing_structural_index"]]
              .sort_values("DISPLAY_VAL", ascending=False)
              .head(15))
st.dataframe(top_tracts, use_container_width=True)

if show_top_codes:
    st.subheader("Most Categories This Month")
    top_codes=(month_df.groupby(["VIOLATION CODE", "VIOLATION_DESCRIPTION"], as_index=False)
                 ["VIOLATION_COUNT"].sum()
                 .sort_values("VIOLATION_COUNT", ascending=False)
                 .head(15))
    st.dataframe(top_codes, use_container_width=True)