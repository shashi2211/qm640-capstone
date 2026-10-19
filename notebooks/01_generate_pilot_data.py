"""
Generates an ILLUSTRATIVE PILOT dataset matching the schema of the real CMS
Medicare DME public use files (Supplier PUF, Geographic Variation PUF).

IMPORTANT: This is NOT real extracted CMS data. This sandboxed environment
cannot reach data.cms.gov. Values are randomly generated but calibrated to
real, cited, published national parameters (e.g., ~34M Original Medicare
FFS beneficiaries in 2023-2024 per CMS Fast Facts; ~$10B annual national
DME spending per ProMarket/Diwan et al. 2026), so that the pilot sample's
distributions are plausible stand-ins while the full national PUFs are
being ingested (see 02_real_data_acquisition.py for the real download script).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ---------------- COUNTY-LEVEL (geographic) PILOT DATA : RQ1, RQ2 ----------------
N_COUNTY = 420  # exceeds RQ1 (114) and RQ2 (180) minimum sample size requirements

states = ["AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","ID","IL","IN","IA",
          "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH",
          "NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX",
          "UT","VT","VA","WA","WV","WI","WY"]
region_map = {}
ne = {"CT","ME","MA","NH","NJ","NY","PA","RI","VT"}
mw = {"IL","IN","IA","KS","MI","MN","MO","NE","ND","OH","SD","WI"}
south = {"AL","AR","DE","FL","GA","KY","LA","MD","MS","NC","OK","SC","TN","TX","VA","WV"}
for s in states:
    region_map[s] = "Northeast" if s in ne else "Midwest" if s in mw else "South" if s in south else "West"

county_state = rng.choice(states, size=N_COUNTY)
county_region = np.array([region_map[s] for s in county_state])

# Region utilization multipliers (consistent with known national pattern: South/West somewhat higher)
region_util_mult = {"Northeast": 0.92, "Midwest": 0.97, "South": 1.12, "West": 1.03}
region_mult_arr = np.array([region_util_mult[r] for r in county_region])

rucc = rng.integers(1, 10, size=N_COUNTY)  # Rural-Urban Continuum Code 1-9
poverty_rate = np.clip(rng.normal(13, 5, N_COUNTY), 3, 35)
median_income = np.clip(rng.normal(68000, 16000, N_COUNTY) - (rucc * 900), 32000, 140000)
pct_dual = np.clip(rng.normal(19, 7, N_COUNTY) + (poverty_rate * 0.15), 3, 55)
pct_85plus = np.clip(rng.normal(10, 3, N_COUNTY), 2, 22)
hcc_risk = np.clip(rng.normal(1.0, 0.18, N_COUNTY) + (pct_85plus - 10) * 0.01, 0.55, 1.75)
suppliers_per_10k = np.clip(rng.normal(8.5, 3.2, N_COUNTY) - (rucc * 0.35), 0.5, 22)

noise = rng.normal(0, 28, N_COUNTY)
dme_spending = (
    120 + 140 * hcc_risk + 1.6 * pct_dual + 2.1 * pct_85plus
    - 0.0006 * median_income - 1.3 * suppliers_per_10k + noise
) * region_mult_arr
dme_spending = np.clip(dme_spending, 60, None)

util_noise = rng.normal(0, 9, N_COUNTY)
dme_utilization = np.clip((38 + 55 * hcc_risk + 0.9 * pct_dual + util_noise) * region_mult_arr, 8, None)

county_df = pd.DataFrame({
    "county_id": [f"C{str(i).zfill(4)}" for i in range(N_COUNTY)],
    "state": county_state,
    "census_region": county_region,
    "dme_spending_per_beneficiary": dme_spending.round(2),
    "dme_utilization_rate": dme_utilization.round(2),
    "avg_hcc_risk_score": hcc_risk.round(3),
    "pct_dual_eligible": pct_dual.round(2),
    "pct_age_85plus": pct_85plus.round(2),
    "median_household_income": median_income.round(0),
    "rucc_code": rucc,
    "poverty_rate": poverty_rate.round(2),
    "suppliers_per_10k_beneficiaries": suppliers_per_10k.round(2),
})

# ---- Inject realistic data quality issues (to demonstrate cleaning pipeline) ----
# 1. Missing values (MCAR, ~4% in a few columns)
for col, frac in [("median_household_income", 0.035), ("avg_hcc_risk_score", 0.02),
                   ("suppliers_per_10k_beneficiaries", 0.03)]:
    idx = rng.choice(county_df.index, size=int(frac * N_COUNTY), replace=False)
    county_df.loc[idx, col] = np.nan

# 2. Duplicate rows (~1.2%)
dup_idx = rng.choice(county_df.index, size=5, replace=False)
county_df = pd.concat([county_df, county_df.loc[dup_idx]], ignore_index=True)

# 3. Extreme outliers (simulate rare data-entry/edge-case counties, ~1%)
out_idx = rng.choice(county_df.index, size=4, replace=False)
county_df.loc[out_idx, "dme_spending_per_beneficiary"] *= rng.uniform(3.2, 4.5, size=4)

county_df.to_csv("/home/claude/interim/data/pilot_county_geographic.csv", index=False)
print("County pilot file:", county_df.shape)

# ---------------- SUPPLIER-LEVEL PILOT DATA : RQ3, RQ4 ----------------
N_SUP = 650  # exceeds RQ3 minimum (385)
hcpcs_categories = ["Wheelchairs", "Oxygen Equipment", "CPAP/Respiratory", "Hospital Beds",
                     "Diabetic Supplies", "Orthotics/Prosthetics", "Enteral/Parenteral Nutrition"]
sup_state = rng.choice(states, size=N_SUP)
sup_category = rng.choice(hcpcs_categories, size=N_SUP, p=[0.18,0.14,0.16,0.10,0.22,0.12,0.08])

base_benes = rng.lognormal(mean=3.6, sigma=1.0, size=N_SUP)
base_benes = np.clip(base_benes, 11, 4000)  # CMS suppresses n<=10
avg_payment_per_claim = np.clip(rng.normal(185, 70, N_SUP), 25, 900)
claims_per_bene = np.clip(rng.normal(3.1, 1.1, N_SUP), 1, 12)
total_payment = base_benes * claims_per_bene * avg_payment_per_claim

# Inject a genuine outlier subpopulation (~3%) with implausibly high billing -- this is
# exactly the kind of pattern RQ3's anomaly detection is designed to surface.
anomaly_idx = rng.choice(N_SUP, size=int(0.035 * N_SUP), replace=False)
total_payment[anomaly_idx] *= rng.uniform(6, 14, size=len(anomaly_idx))
avg_payment_per_claim[anomaly_idx] *= rng.uniform(2.5, 5, size=len(anomaly_idx))

supplier_df = pd.DataFrame({
    "supplier_id": [f"S{str(i).zfill(5)}" for i in range(N_SUP)],
    "state": sup_state,
    "hcpcs_category": sup_category,
    "total_beneficiaries_served": base_benes.round(0).astype(int),
    "claims_per_beneficiary": claims_per_bene.round(2),
    "avg_payment_per_claim": avg_payment_per_claim.round(2),
    "total_medicare_payment": total_payment.round(2),
    "known_anomaly_flag_for_validation": np.isin(np.arange(N_SUP), anomaly_idx).astype(int),
})

# Inject missing + duplicate issues for the supplier file too
for col, frac in [("avg_payment_per_claim", 0.025), ("claims_per_beneficiary", 0.018)]:
    idx = rng.choice(supplier_df.index, size=int(frac * N_SUP), replace=False)
    supplier_df.loc[idx, col] = np.nan
dup_idx2 = rng.choice(supplier_df.index, size=7, replace=False)
supplier_df = pd.concat([supplier_df, supplier_df.loc[dup_idx2]], ignore_index=True)

supplier_df.to_csv("/home/claude/interim/data/pilot_supplier.csv", index=False)
print("Supplier pilot file:", supplier_df.shape)
