[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YlfKWlZ5)

# Final_Project-Susan_Weiyi_Yuqian
This project focuses on the spatial distribution and changes of housing risk in Chicago, and examines whether it overlaps with housing quality issues and economic pressure.

## Setup

```bash
conda env create -f environment.yml
conda activate fire_analysis
```

## Project Structure

```
data/
  raw-data/                             # Raw data files
    median_household_income_B19013.csv  # Median household income in the past 12 months
    poverty_status_B17001.csv           # Poverty status (for calculating poverty rate)
    rent_burden_B25070.csv              # Gross rent as a percentage of household income in the past 12 months
    tenure_owner_renter_B25003.csv      # Renter and owner counting
    units_in_structure_B25024.csv       # Building age classified by decades
  derived-data/                         # Filtered data and output plots
    economic_pressure.csv               # Merge median_household_income and rent_burden with adding the economic pressure index
    housing_structural_features.csv     # Merge poverty_status, tenure_owner_renter and units_in_structure with adding the housing structural index
code/
  preprocessing.py                      # Filters all raw data
  xxxplot.py
```

## Usage

1. Run preprocessing to filter data:
   ```bash
   python code/preprocessing.py
   ```

2. Generate the xxx plot:
   ```bash
   python code/xxx.py
   ```

## Large Raw Dataset
`building_violations_raw.csv` is >100MB and is not stored in this GitHub repository.

Download (Google Drive):
- https://drive.google.com/file/d/1pgeuZq6TKBKSqyTAro4o173iqnzVodFW/view?usp=sharing
'Link access: anyone with the link can view/download.'

After downloading, save it to this exact path (do not rename):
- `data/raw-data/building_violations_raw.csv`
