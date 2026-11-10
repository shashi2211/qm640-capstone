# QM640 Capstone: Explaining Variation in Medicare DME Spending

Shashi Shekhar | Walsh College | QM640 Data Analytics Capstone | Mentor: Smitraj Raut

## Project summary
This repository supports a capstone study of variation in Original Medicare
Durable Medical Equipment (DME) spending and utilization across U.S. counties,
supplier billing anomalies, and supplier market concentration. It uses public
CMS data and a mix of statistical and machine learning methods: multiple
regression, Random Forest, one-way ANOVA, K-Means, IQR screening, Isolation
Forest, and the Herfindahl-Hirschman Index (HHI).

## Status (Final Report draft)
All results in the final report draft are computed from the CMS files in
`data/`. The files were truncated to under 25 MB each to fit GitHub, so results
are preliminary. Known data gaps (documented in the report's Limitations
section): untruncated national files, an HCC risk score, ACS county-level
covariates, and a supplier-by-HCPCS-category payment file for RQ4.

## Data sources
- Medicare Geographic Variation by National, State & County Public Use File (CMS)
- Medicare DME, Devices & Supplies by Supplier (CMS)
- Medicare DME, Devices & Supplies by Referring Provider and Service (CMS)
- U.S. Census Bureau survey data (currently a Current Population Survey
  extract, not used in the models; the ACS 5-Year county summary file is to be
  substituted)

All CMS files are from https://data.cms.gov.

## Repository structure
```
qm640-capstone/
├── data/
│   ├── 2014-2024_Original_Medicare_Geographic_Variation_data.csv   # raw (truncated)
│   ├── DME_Device_Supplier.csv                                     # raw (truncated)
│   ├── DME_Referring_Provider_Supplier.csv                         # raw (truncated)
│   ├── Census_Data.zip                                             # CPS extract (not used)
│   ├── MUP_DME_RY24_Data-Dictionary_20240620_rfrr-DY17-22.pdf      # CMS data dictionary
│   ├── clean_county_geographic.csv                                 # cleaned county file
│   └── clean_supplier.csv                                          # cleaned supplier file
├── notebooks/
│   └── 02_analysis_pipeline_real.py   # cleaning, EDA, modeling pipeline (RQ1-RQ4)
├── companion/                         # additional analyses added for the final report
│   ├── figs.py      # regenerates figures at print resolution; K-Means, Isolation Forest, OLS diagnostics
│   ├── run03.py     # descriptive stats, assumption tests, HC3 SEs, permutation importance, power checks
│   ├── figs2.py     # workflow diagram and combined regional chart
│   └── run04.py     # RQ3 Mann-Whitney / chi-square tests, train RMSE, mean-only baseline
├── figures/                           # charts written by the main pipeline
├── results/
│   └── results.json                   # statistics written by the main pipeline
├── requirements.txt
└── README.md
```

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
4. Create the output folders used by the companion scripts:
   ```
   mkdir -p figs final
   ```
5. Run the main pipeline (cleaning, EDA, RQ1-RQ4 baseline models):
   ```
   python notebooks/02_analysis_pipeline_real.py
   ```
   This writes the cleaned CSVs, `data/rq4_hhi_preliminary.csv`, other
   intermediate tables, the charts in `figures/`, and `results/results.json`.
6. Run the companion scripts **in this order** (later scripts use files
   created by earlier ones):
   ```
   python companion/figs.py     # figures in figs/ and data/county_with_cluster.csv
   python companion/run03.py    # tables t_*.csv and results_extra.json in data/
   python companion/figs2.py    # workflow diagram and regional chart; collects figures in final/
   python companion/run04.py    # RQ3 tests and train/baseline RMSE (printed to console)
   ```
7. Compare the console output and the generated files with the tables and
   figures in the final report.

### Path note for the main pipeline
`notebooks/02_analysis_pipeline_real.py` was written with absolute paths
(`/home/claude/interim_v2/...`). Before running it, change the three path
constants near the top and every output path to repository-relative paths:
```
DATA = "data/"
FIG  = "figures/"
# and replace "/home/claude/interim_v2/data/" with "data/" in every output path
```
A one-line fix on Linux/macOS:
```
sed -i 's#/home/claude/interim_v2/data/#data/#g; s#/home/claude/interim_v2/figures/#figures/#g' notebooks/02_analysis_pipeline_real.py
```
`results.json` is written to `data/` by that script; move it to `results/` or
change its path as well. The companion scripts already use relative paths.

## Research questions
- **RQ1:** What beneficiary, socioeconomic, geographic, and supply-side factors
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
See the final report for details: truncated CMS files (State-level rows cover
22 of 50 states), no HCC risk score, 42.8% suppression in race-composition
fields, no category-level supplier payments for RQ4, no labeled fraud data for
RQ3, and counties nested within states (non-independent observations).
