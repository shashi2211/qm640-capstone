# QM640 Capstone: Explaining Variation in Medicare DME Spending

Shashi Shekhar | Walsh College | QM640 Data Analytics Capstone

## Project summary
This repository supports a capstone study of variation in Original Medicare
Durable Medical Equipment (DME) spending and utilization across U.S. counties,
supplier billing anomalies, and supplier market concentration, using real CMS
public-use data and a mix of statistical and machine learning methods.

## Status (Week 5 Interim Report)
All results in the Week 5 interim report are computed from the real CMS files
in `/data` (truncated to under 25MB each for GitHub). See the report's
"Interim Limitations and Risks" section for the specific, scheduled items
(untruncated files, HCC risk score, ACS covariates, category-level supplier
data) that will be completed before the final report.

## Data sources
- Medicare Geographic Variation Public Use File (CMS)
- Medicare DME, Devices & Supplies by Supplier (CMS)
- Medicare DME by Referring Provider and Service (CMS)
- U.S. Census Bureau survey data (currently a CPS extract; ACS 5-Year county
  summary file to be substituted — see Interim Limitations)

## Repository structure
```
qm640-capstone/
├── data/                    # raw CMS/Census downloads (truncated <25MB each)
│   ├── 2014-2024_Original_Medicare_Geographic_Variation_data.csv
│   ├── DME_Device_Supplier.csv
│   ├── DME_Referring_Provider_Supplier.csv
│   ├── Census_Data.zip
│   ├── MUP_DME_RY24_Data-Dictionary_20240620_rfrr-DY17-22.pdf
│   └── processed/           # cleaned outputs
│       ├── clean_county_geographic.csv
│       └── clean_supplier.csv
├── notebooks/
│   └── 02_analysis_pipeline_real.py   # full cleaning/EDA/modeling pipeline
├── figures/                 # all 11 generated charts
├── results/
│   └── results.json         # every statistic cited in the report
├── requirements.txt
└── README.md
```

## How to reproduce every table and figure in the report
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
4. Run the pipeline from the repository root:
   ```
   python notebooks/02_analysis_pipeline_real.py
   ```
5. Outputs:
   - Cleaned data written to `data/processed/`
   - All 11 charts written to `figures/`
   - Complete numeric results written to `results/results.json`
6. Compare the console output and `results.json` against the tables in the
   interim report to confirm an exact reproduction.

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
