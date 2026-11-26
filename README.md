# QM640 Capstone: Explaining Variation in Medicare DME Spending

Shashi Shekhar | Walsh College | QM640 Data Analytics Capstone | Mentor: Smitraj Raut

## Project summary
This repository supports a capstone study of variation in Original Medicare
Durable Medical Equipment (DME) spending and utilization across U.S. counties,
supplier billing anomalies, and supplier market concentration. It uses the
**complete** public CMS and Census data files (not a truncated sample) and a
mix of statistical and machine learning methods: multiple regression, Random
Forest, one-way ANOVA, K-Means, IQR screening, Isolation Forest, and the
Herfindahl-Hirschman Index (HHI).

## Status (Final Report)
All results in the final report are computed from the complete CMS/ACS files
in `data/raw/`. Those files were split into multiple parts (not truncated —
every row is intact) to fit GitHub's 25MB-per-file limit. Known remaining
data gaps are documented in the report's Limitations section: no county-level
HCC risk score (a supplier-level risk score is used for RQ3 instead), and
RQ4's referring-provider HHI view measures ordering concentration rather than
supplier billing concentration directly.

## Data sources
- Medicare Geographic Variation by National, State & County Public Use File (CMS) — complete file, all years
- Medicare DME, Devices & Supplies by Supplier (CMS) — complete file
- Medicare DME, Devices & Supplies by Referring Provider and Service (CMS) — complete file
- American Community Survey 5-Year Estimates, county level (U.S. Census Bureau), 2023

All CMS files are from https://data.cms.gov. The ACS file is from https://data.census.gov.

## Repository structure
```
qm640-capstone/
├── data/
│   ├── raw/                                   # complete source files, split into <25MB parts
│   │   ├── geographic_variation_full_part_*.csv
│   │   ├── Medicare_DME_Devices_Supplies_by_Supplier_2024_part_*.csv
│   │   ├── Medicare_DME_Devices_Supplies_by_Referring_Provider_and_Service_2024_part_*.csv
│   │   ├── acs5_2023_county.csv
│   │   └── MUP_DME_RY24_Data-Dictionary_20240620_rfrr-DY17-22.pdf   # official CMS data dictionary
│   ├── clean_county_geographic.csv             # cleaned county file (3,193 rows) — committed
│   ├── clean_supplier.csv                      # cleaned supplier file (56,562 rows) — NOT committed, regenerate locally (32MB > GitHub limit)
│   ├── county_with_cluster.csv                 # county file with RQ2 cluster labels — committed
│   └── qm640_capstone.db                       # SQLite database — NOT committed, regenerate locally (35MB > GitHub limit)
├── notebooks/
│   ├── 00_master_pipeline_full.py              # data cleaning, ACS merge, database setup, EDA
│   ├── 01_modeling_pipeline_full.py            # RQ1-RQ4 modeling, evaluation, all figures
│   └── 02_build_workflow_diagram.py            # builds final/workflow_diagram.png (Figure 6)
├── figures/                                     # 14 charts (Figures 1-10 in the report map to fig1-fig14 here)
├── final/
│   ├── workflow_diagram.png                    # Figure 6 in the report
│   └── dataset_preview_screenshot.png
├── results/
│   └── results.json                            # every statistic cited in the report
├── requirements.txt
└── README.md
```

**Note:** the `companion/` folder from an earlier stage of this project
(`figs.py`, `run03.py`, `figs2.py`, `run04.py`) is superseded by the three
scripts in `notebooks/` above, which now include that same functionality
(Mann-Whitney U, chi-square, cluster profiling, workflow diagram) natively.
Delete the `companion/` folder to avoid confusion about which pipeline is
authoritative.

## How to reproduce the results
Run every command from the repository root.

1. Clone the repository:
   ```
   git clone https://github.com/shashi2211/qm640-capstone.git
   cd qm640-capstone
   ```
2. Create and activate a Python 3.11 environment (conda or venv).
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the pipeline, in order, from the repository root (no path edits needed —
   both scripts use paths relative to the repo root):
   ```
   python notebooks/00_master_pipeline_full.py
   python notebooks/01_modeling_pipeline_full.py
   python notebooks/02_build_workflow_diagram.py
   ```
5. Outputs:
   - `data/clean_county_geographic.csv`, `data/clean_supplier.csv`,
     `data/county_with_cluster.csv`, and `data/qm640_capstone.db`
   - All 14 charts in `figures/`
   - `final/workflow_diagram.png`
   - `results/results.json`, with every statistic cited in the report
6. Compare the console output and `results.json` against the tables in the
   final report to confirm an exact reproduction.

## Research questions
- **RQ1:** What beneficiary, socioeconomic, economic, and supply-side factors
  explain variation in standardized Original Medicare DME spending per
  beneficiary across U.S. counties, and which factors contribute most?
- **RQ2:** Are there statistically significant geographic disparities in DME
  utilization and supplier availability, and do high-utilization or low-access
  areas share identifiable characteristics?
- **RQ3:** Can DME supplier billing outliers be identified using statistical
  and unsupervised machine learning methods, and what characteristics
  distinguish flagged suppliers from other suppliers?
- **RQ4:** How concentrated is Original Medicare DME spending among suppliers
  across states and product categories, and is greater concentration
  associated with higher spending per beneficiary or lower supplier
  availability?

## Known limitations
See the final report's Limitations section for full detail: no county-level
HCC risk score (supplier-level risk score used for RQ3 instead), 44.3%
suppression in race-composition fields (a real CMS privacy rule, not a data
gap), RQ4's referring-provider HHI view measures ordering concentration
rather than supplier billing concentration directly, no labeled fraud outcome
for RQ3, and counties nested within states (non-independent observations).
