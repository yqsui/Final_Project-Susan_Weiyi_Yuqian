# Final_Project-Susan_Weiyi_Yuqian
This project focuses on the spatial distribution and changes of housing risk in Chicago, and examines whether it overlaps with housing quality issues and economic pressure.

## Setup

```bash
conda env create -f requirements.yml

conda activate housing_risk_analysis    # for Windows
source activate housing_risk_analysis   # for MacOS / Linux
```

## Project Structure

```
data/
  raw-data/                             # Raw data files
    building_violations_raw             # Building violation record
    median_household_income_B19013.csv  # Median household income in the past 12 months
    poverty_status_B17001.csv           # Poverty status (for calculating poverty rate)
    rent_burden_B25070.csv              # Gross rent as a percentage of household income in the past 12 months
    tenure_owner_renter_B25003.csv      # Renter and owner counting
    units_in_structure_B25024.csv       # Building age classified by decades
  derived-data/                         # Filtered data and output plots
    economic_pressure.csv               # Merged median_household_income and rent_burden with adding the economic pressure index
    housing_structural_features.csv     # Merged poverty_status, tenure_owner_renter and units_in_structure with adding housing structural index
    building_violation_by_code_rate     # Monthly tract-level violation counts and rates per 1,000 housing units, disaggregated by violation code
    part1_distribution.png              # Visualization of the spatial distribution of the three index
    part2_highlight.png                 # Display of the degree of overlap of the top 25% of three distribution
    part2_overlap.png                   # Hightlight of only tracts with overlap_count = 3
    scatter_plot.png                    # Scatter plot of relationships between economic index, housing index & building violation rate

code/
  preprocessing.py                      # Filters all raw data
  plot_maps.py                          # Geographic distribution and overlap of economic and structural pressure, and violation intensity
  plot_scatter.py                       # Draw scatter plot
  app.py                                # Create Streamlit app
```

## Usage

1. Run preprocessing to filter data:
   ```bash
   code/preprocessing.py
   ```

2. Generate the plots:
   ```bash
   code/plot_maps.py
   code/plot_scatter.py
   ```

3. Generate the Streamlit app:
   ```bash
   code/app.py
   ```

## Large Raw Dataset
`building_violations_raw.csv` is >100MB and is not stored in this GitHub repository.

Download (Google Drive):
- https://drive.google.com/file/d/1pgeuZq6TKBKSqyTAro4o173iqnzVodFW/view?usp=sharing
/
Link access: anyone with the link can view/download.

After downloading, save it to this exact path (do not rename):
- `data/raw-data/building_violations_raw.csv`

## Streamlit Link
-https://finalproject-susanweiyiyuqian-h5doxuwqc6xyifknukjwu2.streamlit.app/
/
Here is the link to Streamlit Community Cloud Dashboard.