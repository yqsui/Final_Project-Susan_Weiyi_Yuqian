import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk
import numpy as np
from pathlib import Path
import re
import matplotlib.cm as cm

st.set_page_config(page_title="Chicago Building Violations Dashboard",layout="wide")


# helpers: shorten labels / map intensity / match Reds palette
def simplify_desc(s,max_len=26):
    if pd.isna(s):
        return ""
    s=str(s).strip().lower()

    s=re.sub(r"[^a-z0-9 ]"," ",s)
    s=re.sub(r"\s+"," ",s).strip()

    # normalize common variants
    s=s.replace("comercial","commercial")
    s=s.replace("residtl","residential")
    s=s.replace("bldg","building")
    s=s.replace("bldgs","building")
    s=s.replace("sec","secure")
    s=s.replace("secur","secure")
    s=s.replace("encl","enclosure")
    s=s.replace("flr","floor")
    s=s.replace("bsmt","basement")
    s=s.replace("bx","box")
    s=s.replace("eleva","elevator")
    s=s.replace("elect","electric")
    s=s.replace("hydro","hydraulic")

    # drop specs/noise
    drop=set([
        "hour","hours","hr","pre57","story","stories","gauge","above","over","under",
        "ft","feet","in","inch","inches","du","lot","req","required","the","and","or","to","of","for","with","on","at","by","from"
    ])
    toks=[]
    for t in s.split():
        if t in drop:
            continue
        if re.fullmatch(r"\d+(\.\d+)?",t):
            continue
        if re.fullmatch(r"\d+\+?",t):
            continue
        if re.fullmatch(r"<\d+|\d+>",t):
            continue
        toks.append(t)

    text=" ".join(toks)

    # collapse to readable categories
    patterns=[
        (r".*\bboard\b.*\bsecure\b.*\bbuilding\b.*","board/secure building"),
        (r".*\bboard\b.*\bbuilding\b.*","board building"),
        (r".*\bsecure\b.*\bbuilding\b.*","secure building"),
        (r".*\bheating\b.*\bplant\b.*\benclosure\b.*","heating plant enclosure"),
        (r".*\bstair\b.*\benclosure\b.*\bhotel\b.*","stair enclosure (hotel)"),
        (r".*\bstair\b.*\benclosure\b.*","stair enclosure"),
        (r".*\bporch\b.*","porch"),
        (r".*\bhood\b.*\bstack\b.*\broof\b.*","hood stack (roof)"),
        (r".*\bbond\b.*\bfitting\b.*","electrical bonding"),
        (r".*\bbond\b.*","electrical bonding"),
        (r".*\brung\b.*|.*\bladder\b.*","ladder safety"),
        (r".*\bshaft\b.*\bduct\b.*","shaft duct"),
        (r".*\belevator\b.*","elevator"),
        (r".*\belectric\b.*","electric"),
        (r".*\bpermit\b.*|.*\bpermits\b.*","permit issue"),
        (r".*\binspection\b.*","inspection"),
        (r".*\brepair\b.*\bwall\b.*","wall repair"),
        (r".*\brepair\b.*\bporch\b.*","porch repair"),
        (r".*\brepair\b.*","repair"),
        (r".*\bbox\b.*","box"),
    ]
    for pat,label in patterns:
        if re.match(pat,text):
            return label if len(label)<=max_len else label[:max_len-1]+"…"

    # fallback: keep first few words
    toks=toks[:4]
    out=" ".join(toks).strip()
    if out=="":
        out="other"
    if len(out)>max_len:
        out=out[:max_len-1]+"…"
    return out


def make_alpha(series,relative_mode):
    series=pd.to_numeric(series,errors="coerce").fillna(0.0)

    if relative_mode:
        vmax=float(series.quantile(0.95))
        if vmax<=1:
            vmax=1.5
        fill_alpha=(
            (series-1)
            .clip(lower=0,upper=vmax-1)
            .div(vmax-1)
            .mul(235)
            .add(20)
            .clip(20,255)
            .astype(int)
        )
        return fill_alpha
    else:
        vmax=float(series.quantile(0.95))
        if vmax<=0:
            vmax=1.0
        fill_alpha=(
            series
            .clip(upper=vmax)
            .div(vmax)
            .mul(235)
            .add(20)
            .clip(20,255)
            .astype(int)
        )
        return fill_alpha


def reds_rgba(n):
    n=int(n)
    n=max(0,min(3,n))
    r,g,b,a=cm.get_cmap("Reds")(n/3)
    return [int(r*255),int(g*255),int(b*255),int(a*255)]


# set paths & load data
script_dir=Path(__file__).parent
raw_dir=script_dir/"../data/raw-data"
out_dir=script_dir/"../data/derived-data"

TRACT_SHP=raw_dir/"tl_2024_17_tract.shp"
RATE_PATH=out_dir/"building_violation_by_code_rate.csv"
ECON_PATH=out_dir/"economic_pressure.csv"
STRUCT_PATH=out_dir/"housing_structural_features.csv"

econ=pd.read_csv(ECON_PATH)
econ["GEOID"]=econ["GEOID"].astype(str).str.zfill(11)
econ=econ[["GEOID","economic_pressure_index"]].copy()

struct=pd.read_csv(STRUCT_PATH)
struct["GEOID"]=struct["GEOID"].astype(str).str.zfill(11)
struct=struct[["GEOID","housing_structural_index"]].copy()

tracts=gpd.read_file(TRACT_SHP)[["GEOID","geometry"]].copy()
tracts["GEOID"]=tracts["GEOID"].astype(str).str.zfill(11)
tracts=tracts.to_crs("EPSG:4326")

df=pd.read_csv(RATE_PATH)
df["GEOID"]=df["GEOID"].astype(str).str.zfill(11)
df["YEAR_MONTH_DT"]=pd.to_datetime(df["YEAR_MONTH_DT"],errors="coerce")

need_cols=[
    "GEOID","YEAR_MONTH_DT","YEAR_MONTH_STR",
    "VIOLATION CODE","VIOLATION_DESCRIPTION",
    "VIOLATION_COUNT","total_buildings","VIOL_PER_1000_BUILDINGS"
]
df=df[[c for c in need_cols if c in df.columns]].copy()

# precompute readable categories + frequency
code_desc=(df[["VIOLATION CODE","VIOLATION_DESCRIPTION","VIOLATION_COUNT"]]
            .dropna(subset=["VIOLATION CODE"])
            .copy())
code_desc["DESC_READABLE"]=code_desc["VIOLATION_DESCRIPTION"].apply(simplify_desc)

desc_freq=(code_desc.groupby("DESC_READABLE",as_index=False)["VIOLATION_COUNT"]
            .sum()
            .rename(columns={"VIOLATION_COUNT":"FREQ"}))
desc_options=(desc_freq.sort_values(["FREQ","DESC_READABLE"],ascending=[False,True])["DESC_READABLE"].tolist())

# align geoids across sources
keep_geoids=set(df["GEOID"].dropna())&set(econ["GEOID"].dropna())&set(struct["GEOID"].dropna())
tracts=tracts[tracts["GEOID"].isin(keep_geoids)].copy()


# sidebar
st.sidebar.header("Filters")
page=st.sidebar.radio("View",["Monthly explorer","Hotspot overlap"],index=0)


# view 1: Monthly explorer
if page=="Monthly explorer":
    valid_months=(df["YEAR_MONTH_DT"].dropna().drop_duplicates().sort_values().to_list())
    if len(valid_months)==0:
        st.error("No valid YEAR_MONTH_DT found.")
        st.stop()

    month_dt=st.sidebar.select_slider(
        "Select month",
        options=valid_months,
        value=valid_months[0],
        format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m")
    )

    desc_choice=st.sidebar.selectbox(
        "Select category (description)",
        ["(All categories)"]+desc_options
    )
    st.sidebar.caption("Type to search. Options sorted by frequency.")

    relative_mode=st.sidebar.toggle(
        "Relative to city average",
        value=False,
        help="ON → rate / city_mean_rate  |  OFF → raw rate per 1,000 units"
    )

    show_top_codes=st.sidebar.checkbox("Show top codes table",value=True)

    month_df=df[df["YEAR_MONTH_DT"]==pd.Timestamp(month_dt)].copy()

    if desc_choice!="(All categories)":
        sel_codes=(code_desc.loc[code_desc["DESC_READABLE"]==desc_choice,"VIOLATION CODE"]
                   .dropna().unique().tolist())
        month_df=month_df[month_df["VIOLATION CODE"].isin(sel_codes)].copy()

    # aggregate by tract
    tract_agg=(month_df.groupby("GEOID",as_index=False)
                 .agg(VIOLATION_COUNT=("VIOLATION_COUNT","sum"),
                      RATE=("VIOL_PER_1000_BUILDINGS","sum")))

    city_mean=tract_agg["RATE"].mean() if len(tract_agg)>0 else 1.0
    if city_mean==0:
        city_mean=1.0

    if relative_mode:
        tract_agg["DISPLAY_VAL"]=(tract_agg["RATE"]/city_mean).round(3)
        map_label="Rate / City Mean"
    else:
        tract_agg["DISPLAY_VAL"]=tract_agg["RATE"].round(2)
        map_label="Rate per 1,000 units"

    map_gdf=tracts.merge(tract_agg,on="GEOID",how="left")
    map_gdf["VIOLATION_COUNT"]=map_gdf["VIOLATION_COUNT"].fillna(0).astype(int)
    map_gdf["RATE"]=map_gdf["RATE"].fillna(0.0)
    map_gdf["DISPLAY_VAL"]=map_gdf["DISPLAY_VAL"].fillna(0.0)

    map_gdf=map_gdf.merge(econ,on="GEOID",how="left")
    map_gdf=map_gdf.merge(struct,on="GEOID",how="left")

    col1,col2,col3=st.columns(3)
    col1.metric("Month",pd.Timestamp(month_dt).strftime("%Y-%m"))
    col2.metric("Selected category",desc_choice)
    col3.metric("Total violations (count)",int(map_gdf["VIOLATION_COUNT"].sum()))

    mode_txt="relative to city average (ratio)" if relative_mode else "raw rate per 1,000 housing units"
    st.caption(f"Map shows **{mode_txt}** (darker=higher).")

    # map layer
    map_ll=map_gdf.to_crs("EPSG:4326").copy()
    map_ll["DISPLAY_VAL"]=pd.to_numeric(map_ll["DISPLAY_VAL"],errors="coerce").fillna(0.0)
    map_ll["fill_alpha"]=make_alpha(map_ll["DISPLAY_VAL"],relative_mode)

    layer=pdk.Layer(
        "GeoJsonLayer",
        data=map_ll.__geo_interface__,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        get_line_color=[255,255,255,60],
        line_width_min_pixels=0.5,
        get_fill_color="[255, 80, 0, properties.fill_alpha]",
    )

    view_state=pdk.ViewState(latitude=41.88,longitude=-87.63,zoom=10,min_zoom=8,max_zoom=14)

    tooltip={
        "html":(
            "<b>Tract GEOID:</b> {GEOID}<br/>"
            f"<b>{map_label}:</b> {{DISPLAY_VAL}}<br/>"
            "<b>Violation count:</b> {VIOLATION_COUNT}<br/>"
            "<b>Econ pressure index:</b> {economic_pressure_index}<br/>"
            "<b>Housing structural index:</b> {housing_structural_index}"
        ),
        "style":{"backgroundColor":"white","color":"black"}
    }

    st.subheader("Map")
    st.pydeck_chart(
        pdk.Deck(layers=[layer],initial_view_state=view_state,map_style="light",tooltip=tooltip),
        use_container_width=True
    )

    st.subheader("Top Tracts")
    top_tracts=(map_gdf[["GEOID","DISPLAY_VAL","VIOLATION_COUNT","economic_pressure_index","housing_structural_index"]]
                  .sort_values("DISPLAY_VAL",ascending=False)
                  .head(15))
    st.dataframe(top_tracts,use_container_width=True)

    if show_top_codes:
        st.subheader("Most Categories This Month")
        top_codes=(month_df.groupby(["VIOLATION CODE","VIOLATION_DESCRIPTION"],as_index=False)["VIOLATION_COUNT"]
                     .sum()
                     .sort_values("VIOLATION_COUNT",ascending=False)
                     .head(15))
        top_codes["VIOLATION_DESCRIPTION"]=top_codes["VIOLATION_DESCRIPTION"].apply(simplify_desc)
        st.dataframe(top_codes,use_container_width=True)


# view 2: Hotspot overlap
else:
    st.sidebar.caption("Hotspot overlap is computed using tract-level averages across the selected time window.")

    window_mode=st.sidebar.radio("Time window",["Overall (2020–2024)","Single year"],index=0)
    if window_mode=="Single year":
        years=sorted(df["YEAR_MONTH_DT"].dropna().dt.year.unique().tolist())
        if len(years)==0:
            st.error("No valid year found.")
            st.stop()
        year_choice=st.sidebar.selectbox("Select year",years,index=0)
        df_win=df[df["YEAR_MONTH_DT"].dt.year==int(year_choice)].copy()
    else:
        df_win=df.copy()

    layers=st.sidebar.multiselect(
        "Overlap layers",
        ["Economic Pressure Index","Structural Feature Index","Log Violation Rate"],
        default=["Economic Pressure Index","Structural Feature Index","Log Violation Rate"]
    )
    st.sidebar.caption("Darker=more selected layers overlap (0–3).")

    q=st.sidebar.slider("Hotspot threshold (top quantile)",min_value=0.5,max_value=0.95,value=0.75,step=0.05)
    st.sidebar.caption(f"Hotspot=top {int(round((1-q)*100))}% tracts (cutoff at q={q:.2f}).")

    if len(layers)<3:
        show_only_triple=False
        st.sidebar.caption("To filter triple-burden tracts, please select all 3 layers.")
    else:
        show_only_triple=st.sidebar.checkbox("Show only triple-burden tracts",value=False)

    # tract-month sum → tract mean (violations)
    vio_month=(df_win.groupby(["GEOID","YEAR_MONTH_DT"],as_index=False)
                 .agg(monthly_total_violation_rate=("VIOL_PER_1000_BUILDINGS","sum")))
    vio_agg=(vio_month.groupby("GEOID",as_index=False)
              .agg(avg_monthly_total_violation_rate=("monthly_total_violation_rate","mean")))

    overlap_df=(vio_agg.merge(econ,on="GEOID",how="inner")
                    .merge(struct,on="GEOID",how="inner"))
    map2_gdf=tracts.merge(overlap_df,on="GEOID",how="inner").copy()
    if len(map2_gdf)==0:
        st.error("No tracts available after merging for overlap analysis.")
        st.stop()

    q_econ=map2_gdf["economic_pressure_index"].quantile(q)
    q_struct=map2_gdf["housing_structural_index"].quantile(q)
    q_vio=map2_gdf["avg_monthly_total_violation_rate"].quantile(q)

    map2_gdf["top_econ"]=map2_gdf["economic_pressure_index"]>=q_econ
    map2_gdf["top_struct"]=map2_gdf["housing_structural_index"]>=q_struct
    map2_gdf["top_vio"]=map2_gdf["avg_monthly_total_violation_rate"]>=q_vio

    map2_gdf["overlap_count"]=0
    if "Economic Pressure Index" in layers:
        map2_gdf["overlap_count"]=map2_gdf["overlap_count"]+map2_gdf["top_econ"].astype(int)
    if "Structural Feature Index" in layers:
        map2_gdf["overlap_count"]=map2_gdf["overlap_count"]+map2_gdf["top_struct"].astype(int)
    if "Log Violation Rate" in layers:
        map2_gdf["overlap_count"]=map2_gdf["overlap_count"]+map2_gdf["top_vio"].astype(int)

    map2_gdf["triple_hotspot"]=map2_gdf["overlap_count"]==3

    if show_only_triple:
        map2_gdf=map2_gdf[map2_gdf["triple_hotspot"]].copy()

    col1,col2,col3,col4=st.columns(4)
    col1.metric("Window",str(year_choice) if window_mode=="Single year" else "Overall")
    col2.metric("Quantile cutoff",f"q={q:.2f} (top {int(round((1-q)*100))}%)")
    col3.metric("Tracts shown",int(len(map2_gdf)))
    col4.metric("Triple-burden tracts",int(map2_gdf["triple_hotspot"].sum()))

    st.caption("This view highlights spatial overlap of selected risk layers (econ + structural + violations).")

    map2_ll=map2_gdf.to_crs("EPSG:4326").copy()
    map2_ll["fill_color"]=map2_ll["overlap_count"].apply(reds_rgba)

    layer2=pdk.Layer(
        "GeoJsonLayer",
        data=map2_ll.__geo_interface__,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        get_line_color=[255,255,255,60],
        line_width_min_pixels=0.5,
        get_fill_color="properties.fill_color",
    )

    view_state2=pdk.ViewState(latitude=41.88,longitude=-87.63,zoom=10,min_zoom=8,max_zoom=14)

    tooltip2={
        "html":(
            "<b>Tract GEOID:</b> {GEOID}<br/>"
            "<b>Overlap count (0–3):</b> {overlap_count}<br/>"
            "<b>Triple-burden:</b> {triple_hotspot}<br/>"
            "<b>Economic Pressure Index:</b> {economic_pressure_index}<br/>"
            "<b>Structural Feature Index:</b> {housing_structural_index}<br/>"
            "<b>Avg monthly violation rate:</b> {avg_monthly_total_violation_rate}"
        ),
        "style":{"backgroundColor":"white","color":"black"}
    }

    st.subheader("Overlap Map (0–3)")
    st.pydeck_chart(
        pdk.Deck(layers=[layer2],initial_view_state=view_state2,map_style="light",tooltip=tooltip2),
        use_container_width=True
    )

    st.subheader("Top Tracts by overlap")
    out_cols=[
        "GEOID","overlap_count","triple_hotspot",
        "avg_monthly_total_violation_rate",
        "economic_pressure_index","housing_structural_index"
    ]
    top_overlap=(map2_gdf[out_cols]
                  .sort_values(["overlap_count","avg_monthly_total_violation_rate"],ascending=[False,False])
                  .head(25))
    st.dataframe(top_overlap,use_container_width=True)

    st.download_button(
        "Download overlap table (CSV)",
        data=top_overlap.to_csv(index=False).encode("utf-8"),
        file_name="hotspot_overlap_top_tracts.csv",
        mime="text/csv"
    )