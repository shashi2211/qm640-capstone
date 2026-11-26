"""
Master pipeline for the QM640 Final Report, FULL (untruncated) data version.
Loads the complete CMS files the student uploaded in parts, plus the real
ACS 5-Year county file, and reruns the entire cleaning -> DB -> EDA ->
sample-size -> modeling chain.
"""
import glob
import numpy as np
import pandas as pd
import sqlite3
import json
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_style("whitegrid")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})
DATA = "data/"
FIG = "figures/"
RESULTS = {}

def safe_title(t, w=46): return "\n".join(textwrap.wrap(t, width=w))
def save(n): plt.tight_layout(); plt.savefig(FIG+n, dpi=200, bbox_inches="tight"); plt.close()

def load_parts(pattern, usecols=None):
    files = sorted(glob.glob(DATA + pattern))
    dfs = [pd.read_csv(f, usecols=usecols, low_memory=False) for f in files]
    return pd.concat(dfs, ignore_index=True), files

# ============================================================
# 1. LOAD FULL RAW FILES
# ============================================================
geo_raw, geo_files = load_parts("geographic_variation_full_part_*.csv")
sup_raw, sup_files = load_parts("Medicare_DME_Devices_Supplies_by_Supplier_2024_part_*.csv")
ref_cols = ["Rfrg_Prvdr_State_Abrvtn", "RBCS_Desc", "HCPCS_CD", "Tot_Suplrs", "Tot_Suplr_Clms", "Tot_Suplr_Srvcs", "Avg_Suplr_Mdcr_Pymt_Amt"]
ref_raw, ref_files = load_parts("Medicare_DME_Devices_Supplies_by_Referring_Provider_and_Service_2024_part_*.csv", usecols=ref_cols)
acs = pd.read_csv(DATA + "acs5_2023_county.csv")

RESULTS["raw_shapes"] = {"geographic_variation_files": len(geo_files), "geographic_variation_rows": geo_raw.shape,
                          "dme_supplier_files": len(sup_files), "dme_supplier_rows": sup_raw.shape,
                          "dme_referring_provider_files": len(ref_files), "dme_referring_provider_rows": ref_raw.shape,
                          "acs_rows": acs.shape}
print("Loaded:", RESULTS["raw_shapes"])

region_map = {}
ne = {"CT","ME","MA","NH","NJ","NY","PA","RI","VT"}
mw = {"IL","IN","IA","KS","MI","MN","MO","NE","ND","OH","SD","WI"}
south = {"AL","AR","DE","FL","GA","KY","LA","MD","MS","NC","OK","SC","TN","TX","VA","WV","DC"}
for s_ in list(ne)+list(mw)+list(south):
    region_map[s_] = "Northeast" if s_ in ne else "Midwest" if s_ in mw else "South"
def to_region(st):
    if st in region_map: return region_map[st]
    if st in ["AK","AZ","CA","CO","HI","ID","MT","NV","NM","OR","UT","WA","WY"]: return "West"
    return np.nan

# ============================================================
# 2. GEOGRAPHIC (COUNTY) CLEANING -- now from the FULL multi-year file
# ============================================================
YEAR = 2024
county = geo_raw[(geo_raw.BENE_GEO_LVL=="County") & (geo_raw.BENE_AGE_LVL=="All") & (geo_raw.YEAR==YEAR)].copy()
n_county_raw = len(county)
county["state"] = county["BENE_GEO_DESC"].str.split("-").str[0]
county["census_region"] = county["state"].apply(to_region)
county["fips5"] = county["BENE_GEO_CD"].dropna().astype(float).astype(int).astype(str).str.zfill(5)

key_cols = ["DME_MDCR_STDZD_PYMT_PC","DME_EVNTS_PER_1000_BENES","BENE_AVG_AGE","BENE_DUAL_PCT",
            "BENE_FEML_PCT","BENE_RACE_BLACK_PCT","BENE_RACE_HSPNC_PCT","BENES_TOTAL_CNT"]
suppressed_counts = {}
county_str = geo_raw[(geo_raw.BENE_GEO_LVL=="County") & (geo_raw.BENE_AGE_LVL=="All") & (geo_raw.YEAR==YEAR)].copy()
for c in key_cols:
    suppressed_counts[c] = int((county_str[c].astype(str) == "*").sum())
    county[c] = pd.to_numeric(county[c].astype(str).str.replace("*","", regex=False).replace("", np.nan), errors="coerce")

cleaning_log = []
cleaning_log.append(("CMS small-cell suppression", ", ".join(key_cols), "CMS '*' marker", "NaN -> median imputation", f"suppressed counts: {suppressed_counts}"))

dupes = county.duplicated(subset=["BENE_GEO_CD"]).sum()
county = county.drop_duplicates(subset=["BENE_GEO_CD"]).reset_index(drop=True)
cleaning_log.append(("Duplicates (county)", "BENE_GEO_CD", "pandas .duplicated()", f"removed {dupes}", "exact-key duplicates"))

num_cols_county = ["DME_MDCR_STDZD_PYMT_PC","DME_EVNTS_PER_1000_BENES","BENE_AVG_AGE","BENE_DUAL_PCT","BENE_FEML_PCT","BENE_RACE_BLACK_PCT","BENE_RACE_HSPNC_PCT"]
miss_pct = {}
for c in num_cols_county:
    miss_pct[c] = round(county[c].isna().mean()*100, 2)
    county[c] = county[c].fillna(county[c].median())

q1,q3 = county["DME_MDCR_STDZD_PYMT_PC"].quantile([0.25,0.75]); iqr=q3-q1
hi = q3+1.5*iqr
n_out = int((county["DME_MDCR_STDZD_PYMT_PC"]>hi).sum())
county.loc[county["DME_MDCR_STDZD_PYMT_PC"]>hi, "DME_MDCR_STDZD_PYMT_PC"] = hi
cleaning_log.append(("Outliers (county)", "DME_MDCR_STDZD_PYMT_PC", "IQR rule (1.5xIQR)", f"capped {n_out}", "winsorized upper fence"))

n_before_region = len(county)
county = county.dropna(subset=["census_region"]).reset_index(drop=True)
cleaning_log.append(("Unmapped territory rows", "census_region", "null check", f"dropped {n_before_region-len(county)}", "ANOVA needs the 4 standard regions"))

# --- Merge REAL ACS income/poverty by county FIPS ---
acs["fips5"] = acs["fips5"].astype(str).str.zfill(5)
# ACS/Census Bureau uses sentinel codes (e.g., -666666666) for suppressed/unavailable cells;
# these must be treated as missing, not as literal (wildly incorrect) values.
acs.loc[acs["median_hh_income"] < 0, "median_hh_income"] = np.nan
acs.loc[(acs["poverty_rate"] < 0) | (acs["poverty_rate"] > 1), "poverty_rate"] = np.nan
n_before_acs = len(county)
county = county.merge(acs[["fips5","median_hh_income","poverty_rate"]], on="fips5", how="left")
n_acs_missing = int(county["median_hh_income"].isna().sum())
n_acs_sentinel = int((county["median_hh_income"] < 0).sum())
county["median_hh_income"] = county["median_hh_income"].fillna(county["median_hh_income"].median())
county["poverty_rate"] = county["poverty_rate"].fillna(county["poverty_rate"].median())
cleaning_log.append(("ACS merge (income/poverty)", "median_hh_income, poverty_rate", "left-join on 5-digit county FIPS",
                      f"{n_acs_missing} of {n_before_acs} counties unmatched (median-imputed)", "real ACS 5-Year county file, replacing the earlier CPS-extract error"))

RESULTS["county_cleaning"] = {"n_raw": n_county_raw, "n_final": len(county), "duplicates_removed": int(dupes),
                               "missing_pct": miss_pct, "outliers_capped": n_out, "acs_unmatched": n_acs_missing}

# ============================================================
# 3. SUPPLIER CLEANING -- now from the FULL supplier file (60,060 raw rows)
# ============================================================
n_sup_raw = len(sup_raw)
supplier = sup_raw[sup_raw["DME_Sprsn_Ind"].isna()].copy()
n_sup_dme_active = len(supplier)
dupes_s = supplier.duplicated(subset=["Suplr_NPI"]).sum()
supplier = supplier.drop_duplicates(subset=["Suplr_NPI"]).reset_index(drop=True)

sup_cols = ["DME_Tot_Suplr_Benes","DME_Tot_Suplr_Clms","DME_Tot_Suplr_Srvcs","DME_Suplr_Mdcr_Pymt_Amt","DME_Suplr_Mdcr_Stdzd_Pymt_Amt","Bene_Avg_Risk_Scre"]
miss_pct_s = {}
for c in sup_cols:
    miss_pct_s[c] = round(supplier[c].isna().mean()*100, 2)
    supplier[c] = supplier[c].fillna(supplier[c].median())
cleaning_log.append(("DME-inactive suppliers excluded", "DME_Sprsn_Ind", "CMS DME suppression indicator", f"excluded {n_sup_raw-n_sup_dme_active} of {n_sup_raw}", "out-of-scope general suppliers"))
cleaning_log.append(("Duplicates (supplier)", "Suplr_NPI", "pandas .duplicated()", f"removed {dupes_s}", "one row per NPI"))
cleaning_log.append(("Missing values (supplier)", ", ".join(sup_cols), "null audit", "median imputation", f"{miss_pct_s}"))

# --- State-level supplier density using the FULL state rows (should now cover far more states) ---
state_benes = geo_raw[(geo_raw.BENE_GEO_LVL=="State") & (geo_raw.BENE_AGE_LVL=="All") & (geo_raw.YEAR==YEAR)].copy()
state_benes = state_benes[["BENE_GEO_DESC","BENES_TOTAL_CNT"]].rename(columns={"BENE_GEO_DESC":"state"})
state_benes["BENES_TOTAL_CNT"] = pd.to_numeric(state_benes["BENES_TOTAL_CNT"], errors="coerce")
n_states_with_real_data = state_benes["state"].nunique()
sup_counts_by_state = supplier.groupby("Suplr_Prvdr_State_Abrvtn").size().rename("n_suppliers").reset_index()
sup_counts_by_state.columns = ["state","n_suppliers"]
density = sup_counts_by_state.merge(state_benes, on="state", how="inner")
density["suppliers_per_10k"] = density["n_suppliers"] / density["BENES_TOTAL_CNT"] * 10000
density = density[np.isfinite(density["suppliers_per_10k"])]
county = county.merge(density[["state","suppliers_per_10k"]], on="state", how="left")
n_missing_density = int(county["suppliers_per_10k"].isna().sum())
county["suppliers_per_10k"] = county["suppliers_per_10k"].fillna(county["suppliers_per_10k"].median())
cleaning_log.append(("State coverage (access proxy)", "suppliers_per_10k", "left-join audit",
                      f"{n_missing_density} of {len(county)} imputed; {n_states_with_real_data} states had real State-level 2024 rows",
                      "full multi-year file now covers far more states than the earlier truncated extract"))

county.to_csv(DATA + "clean_county_geographic.csv", index=False)
supplier.to_csv(DATA + "clean_supplier.csv", index=False)
RESULTS["supplier_cleaning"] = {"n_raw": n_sup_raw, "n_dme_active": n_sup_dme_active, "n_final": len(supplier),
                                 "duplicates_removed": int(dupes_s), "missing_pct": miss_pct_s, "n_states_with_real_state_rows": n_states_with_real_data}
RESULTS["cleaning_log"] = cleaning_log

# ============================================================
# 4. DATABASE SETUP
# ============================================================
conn = sqlite3.connect(DATA + "qm640_capstone.db")
county.to_sql("county_geographic", conn, if_exists="replace", index=False)
supplier.to_sql("dme_supplier", conn, if_exists="replace", index=False)
cur = conn.cursor()
cur.execute("CREATE INDEX IF NOT EXISTS idx_county_region ON county_geographic(census_region)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_supplier_state ON dme_supplier(Suplr_Prvdr_State_Abrvtn)")
sample_query = """
    SELECT census_region, COUNT(*) AS n_counties,
           ROUND(AVG(DME_MDCR_STDZD_PYMT_PC), 2) AS avg_spending,
           ROUND(AVG(DME_EVNTS_PER_1000_BENES), 2) AS avg_utilization
    FROM county_geographic GROUP BY census_region ORDER BY avg_spending DESC;
"""
db_check = pd.read_sql(sample_query, conn)
db_check.to_csv(DATA + "db_sample_query_output.csv", index=False)
RESULTS["database"] = {"engine": "SQLite 3", "tables": [f"county_geographic ({len(county)} rows)", f"dme_supplier ({len(supplier)} rows)"],
                        "sample_query_result": db_check.to_dict(orient="records")}
conn.close()

with open(DATA + "results_stage1.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("STAGE 1 DONE")
print(json.dumps(RESULTS, indent=2, default=str)[:3000])
