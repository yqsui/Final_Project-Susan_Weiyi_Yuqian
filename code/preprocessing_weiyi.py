import pandas as pd
from pathlib import Path
script_dir=Path(__file__).parent

# Paths
raw_dir=script_dir / "../data/raw-data"
out_dir=script_dir / "../data/derived-data"
out_dir.mkdir(parents=True, exist_ok=True)


# read building_violations_raw
violations_path=raw_dir/'building_violations_raw.csv'
violations=pd.read_csv(violations_path)
violations['VIOLATION DATE']=pd.to_datetime(violations['VIOLATION DATE'],errors='coerce')
violations['YEAR_MONTH']=violations['VIOLATION DATE'].dt.to_period('M')
violations.head()

violations['LATITUDE']=pd.to_numeric(violations['LATITUDE'],errors='coerce')
violations['LONGITUDE']=pd.to_numeric(violations['LONGITUDE'],errors='coerce')
violations=violations.dropna(subset=['LATITUDE','LONGITUDE']).copy()


# CRS
import geopandas as gpd
viol_gdf=gpd.GeoDataFrame(
    violations,
    geometry=gpd.points_from_xy(violations['LONGITUDE'],violations['LATITUDE']),
    crs='EPSG:4326'
)

print("CRS:",viol_gdf.crs)
print("Geometry type:",viol_gdf.geometry.geom_type.value_counts().to_dict())
print("Rows:",viol_gdf.shape[0])

viol_gdf[['ID','LATITUDE','LONGITUDE','geometry']].head()


# read tracts
tract_path=raw_dir/'tl_2024_17_tract.shp'
tracts=gpd.read_file(tract_path)
print("Tracts CRS:",tracts.crs)
print("Tracts columns:",list(tracts.columns))
tracts=tracts[['GEOID','geometry']].copy()


# spatial join
print("viol_gdf CRS:",viol_gdf.crs)
print("tracts CRS:",tracts.crs)

if viol_gdf.crs!=tracts.crs:
    viol_gdf=viol_gdf.to_crs(tracts.crs)
print("viol_gdf CRS (after):", viol_gdf.crs)

viol_with_tract=gpd.sjoin(
    viol_gdf,
    tracts,
    how='left',
    predicate='intersects'
)


# check whether sjoin created one-to-many matches
print("violations before:",viol_gdf.shape[0])
print("rows after join:",viol_with_tract.shape[0])
print("unmatched points:",viol_with_tract["GEOID"].isna().sum())
print("unique GEOID matched:",viol_with_tract["GEOID"].nunique())


# create tract×month aggregation (for Streamlit dynamic map)
tract_month=viol_with_tract.groupby(['GEOID','YEAR_MONTH'],as_index=False).size()
tract_month=tract_month.rename(columns={'size':'VIOLATION_COUNT'})

tract_month['YEAR_MONTH_STR']=tract_month['YEAR_MONTH'].astype(str)
tract_month['YEAR_MONTH_DT']=tract_month['YEAR_MONTH'].dt.to_timestamp()
tract_month.head()


# Complete the panel of tract*month (give 0)
all_geoids=tract_month['GEOID'].unique()
all_months=pd.period_range('2020-01','2024-12',freq='M')

full_index=pd.MultiIndex.from_product(
    [all_geoids,all_months],
    names=['GEOID','YEAR_MONTH']
)

tract_month_full=(tract_month
                  .set_index(['GEOID','YEAR_MONTH'])
                  .reindex(full_index,fill_value=0)
                  .reset_index())

tract_month_full['YEAR_MONTH_STR']=tract_month_full['YEAR_MONTH'].astype(str)
tract_month_full['YEAR_MONTH_DT']=tract_month_full['YEAR_MONTH'].dt.to_timestamp()

tract_month_full.head()


# save derived data
out_path=out_dir/'building_violation.csv'
tract_month_full.to_csv(out_path,index=False)
