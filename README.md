# QM640 Capstone: Medicare DME Spending Variation

## Status (Week 5 / Interim Report)
Results in the interim report are computed on a **pilot sample** (see `notebooks/01_generate_pilot_data.py`)
built to match the real CMS PUF schemas.

## Folder structure
- `data/raw/` — pilot input CSVs (schema-matched to real CMS files)
- `data/processed/` — cleaned CSVs after the pipeline's cleaning step
- `notebooks/` — acquisition and analysis scripts
- `figures/` — all generated charts (Figures 1-11 in the interim report)
- `results/` — results.json with every computed statistic referenced in the report

## Data sources (real, to be used in the final report)
- Medicare DME, Devices & Supplies by Supplier (CMS)
- Medicare DME by Referring Provider and Service (CMS)
- Medicare Geographic Variation PUF (CMS)
- American Community Survey 5-Year Estimates (U.S. Census Bureau)
