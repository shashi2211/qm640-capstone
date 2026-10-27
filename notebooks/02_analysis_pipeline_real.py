"""
Real-data analysis pipeline for the QM640 DME capstone interim report.
Loads the student's own downloaded (truncated, <25MB each) CMS extracts:
  - 2014-2024_Original_Medicare_Geographic_Variation_data.csv
  - DME_Device_Supplier.csv
  - DME_Referring_Provider_Supplier.csv
and runs the full cleaning, EDA, sample-size, and modeling pipeline for RQ1-RQ4.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import json, textwrap

sns.set_style("whitegrid")
plt.rcParams.update({"axes.titlesize": 11, "axes.labelsize": 10, "figure.titlesize": 11})
DATA = "/home/claude/interim_v2/data/"
FIG = "/home/claude/interim_v2/figures/"
RESULTS = {}

def safe_title(text, width=42):
    return "\n".join(textwrap.wrap(text, width=width))

# ============================================================
# 1. LOAD REAL CMS FILES
# ============================================================
geo_raw = pd.read_csv(DATA + "2014-2024_Original_Medicare_Geographic_Variation_data.csv", low_memory=False)
sup_raw = pd.read_csv(DATA + "DME_Device_Supplier.csv", low_memory=False)
# ref_raw loaded only for literature/context use (category breakdown not usable for per-supplier HHI; see limitations)
ref_raw = pd.read_csv(DATA + "DME_Referring_Provider_Supplier.csv", low_memory=False)

RESULTS["raw_shapes"] = {"geographic_variation": list(geo_raw.shape), "dme_supplier": list(sup_raw.shape),
                          "dme_referring_provider": list(ref_raw.shape)}

region_map = {}
ne = {"CT","ME","MA","NH","NJ","NY","PA","RI","VT"}
mw = {"IL","IN","IA","KS","MI","MN","MO","NE","ND","OH","SD","WI"}
south = {"AL","AR","DE","FL","GA","KY","LA","MD","MS","NC","OK","SC","TN","TX","VA","WV","DC"}
for s in list(ne)+list(mw)+list(south):
    region_map[s] = "Northeast" if s in ne else "Midwest" if s in mw else "South"

def to_region(st):
    if st in region_map: return region_map[st]
    if st in ["AK","AZ","CA","CO","HI","ID","MT","NV","NM","OR","UT","WA","WY"]: return "West"
    return np.nan

# ============================================================
# 2. GEOGRAPHIC (COUNTY) FILE: CLEANING
# ============================================================
YEAR = 2024
county = geo_raw[(geo_raw.BENE_GEO_LVL=="County") & (geo_raw.BENE_AGE_LVL=="All") & (geo_raw.YEAR==YEAR)].copy()
n_county_raw = len(county)
county["state"] = county["BENE_GEO_DESC"].str.split("-").str[0]
county["census_region"] = county["state"].apply(to_region)

key_cols = ["DME_MDCR_STDZD_PYMT_PC","DME_EVNTS_PER_1000_BENES","BENE_AVG_AGE","BENE_DUAL_PCT",
            "BENE_FEML_PCT","BENE_RACE_BLACK_PCT","BENE_RACE_HSPNC_PCT","BENES_TOTAL_CNT"]

cleaning_log = []
# CMS suppresses small cells with "*"; convert to NaN
suppressed_counts = {}
for c in key_cols:
    county[c] = pd.to_numeric(county[c].astype(str).str.replace("*","", regex=False).replace("", np.nan), errors="coerce")
    n_sup = geo_raw.loc[county.index, c].astype(str).eq("*").sum() if c in geo_raw.columns else 0
# Recompute suppression counts properly (before numeric coercion wiped the "*" marker)
county_str = geo_raw[(geo_raw.BENE_GEO_LVL=="County") & (geo_raw.BENE_AGE_LVL=="All") & (geo_raw.YEAR==YEAR)].copy()
for c in key_cols:
    suppressed_counts[c] = int((county_str[c].astype(str) == "*").sum())

cleaning_log.append(("CMS small-cell suppression", ", ".join(key_cols), "CMS '*' marker (counts below privacy threshold)",
                      f"Converted to NaN then median-imputed; counts suppressed per column: {suppressed_counts}",
                      "CMS suppresses cells with very few beneficiaries to protect privacy; median imputation preserves the county sample size needed for RQ1/RQ2 power"))

dupes = county.duplicated(subset=["BENE_GEO_CD"]).sum()
county = county.drop_duplicates(subset=["BENE_GEO_CD"]).reset_index(drop=True)
cleaning_log.append(("Duplicates", "BENE_GEO_CD", "pandas .duplicated()", f"Removed {dupes} duplicate county rows", "Each county should appear once per year/age-level slice"))

missing_pct = {}
for c in key_cols:
    pct = county[c].isna().mean()*100
    missing_pct[c] = round(pct,2)
    county[c] = county[c].fillna(county[c].median())

# Outlier capping on the RQ1 DV
q1,q3 = county["DME_MDCR_STDZD_PYMT_PC"].quantile([0.25,0.75]); iqr=q3-q1
hi = q3+1.5*iqr
n_out = int((county["DME_MDCR_STDZD_PYMT_PC"]>hi).sum())
county.loc[county["DME_MDCR_STDZD_PYMT_PC"]>hi, "DME_MDCR_STDZD_PYMT_PC"] = hi
cleaning_log.append(("Outliers", "DME_MDCR_STDZD_PYMT_PC", "IQR rule (1.5xIQR)", f"Capped {n_out} high-end county outliers",
                      "Preserves full county sample for RQ1/RQ2 power while limiting undue leverage"))

# Missing region (territories not mapped) dropped
n_before_region = len(county)
county = county.dropna(subset=["census_region"]).reset_index(drop=True)
cleaning_log.append(("Unmapped territory rows", "census_region", "Null check after state->region mapping",
                      f"Dropped {n_before_region-len(county)} rows (U.S. territories outside the 4 Census regions)",
                      "RQ2's ANOVA requires the four standard Census Bureau regions"))

RESULTS["county_cleaning"] = {"n_raw": n_county_raw, "n_after_dedup_and_region_filter": len(county),
                               "duplicates_removed": int(dupes), "suppressed_counts": suppressed_counts,
                               "missing_pct_after_suppression_conversion": missing_pct, "outliers_capped": n_out}

# ============================================================
# 3. SUPPLIER FILE: CLEANING (RQ3) + STATE-LEVEL ACCESS PROXY (RQ1/RQ2) + HHI (RQ4)
# ============================================================
n_sup_raw = len(sup_raw)
# Keep only suppliers with real (non-suppressed) DME-specific activity
supplier = sup_raw[sup_raw["DME_Sprsn_Ind"].isna()].copy()
n_sup_dme_active = len(supplier)

dupes_s = supplier.duplicated(subset=["Suplr_NPI"]).sum()
supplier = supplier.drop_duplicates(subset=["Suplr_NPI"]).reset_index(drop=True)

sup_cols = ["DME_Tot_Suplr_Benes","DME_Tot_Suplr_Clms","DME_Tot_Suplr_Srvcs","DME_Suplr_Mdcr_Pymt_Amt","DME_Suplr_Mdcr_Stdzd_Pymt_Amt"]
miss_pct_s = {}
for c in sup_cols:
    pct = supplier[c].isna().mean()*100
    miss_pct_s[c] = round(pct,2)
    supplier[c] = supplier[c].fillna(supplier[c].median())

cleaning_log.append(("DME-inactive suppliers excluded", "DME_Sprsn_Ind and DME_* fields", "CMS DME suppression indicator",
                      f"Excluded {n_sup_raw-n_sup_dme_active} of {n_sup_raw} rows lacking confirmed DME billing activity",
                      "This study is specifically about DME suppliers; rows without DME activity (or DME-suppressed rows) are out of scope"))
cleaning_log.append(("Duplicates (supplier)", "Suplr_NPI", "pandas .duplicated()", f"Removed {dupes_s} duplicate NPI rows", "Each supplier NPI should appear once"))
cleaning_log.append(("Missing values (supplier)", ", ".join(sup_cols), "pandas .isna() audit", "Median imputation per column",
                      f"Residual small-cell suppression within DME-active suppliers; missingness: {miss_pct_s}"))

RESULTS["supplier_cleaning"] = {"n_raw": n_sup_raw, "n_dme_active": n_sup_dme_active, "n_after_dedup": len(supplier),
                                 "duplicates_removed": int(dupes_s), "missing_pct": miss_pct_s}
RESULTS["cleaning_log"] = cleaning_log

# --- State-level supplier density (access proxy): real supplier counts / real state beneficiary totals ---
state_benes = geo_raw[(geo_raw.BENE_GEO_LVL=="State") & (geo_raw.BENE_AGE_LVL=="All") & (geo_raw.YEAR==YEAR)].copy()
state_benes = state_benes[["BENE_GEO_DESC","BENES_TOTAL_CNT"]].rename(columns={"BENE_GEO_DESC":"state"})
state_benes["BENES_TOTAL_CNT"] = pd.to_numeric(state_benes["BENES_TOTAL_CNT"], errors="coerce")

sup_counts_by_state = supplier.groupby("Suplr_Prvdr_State_Abrvtn").size().rename("n_suppliers").reset_index()
sup_counts_by_state.columns = ["state","n_suppliers"]
density = sup_counts_by_state.merge(state_benes, on="state", how="inner")
density["suppliers_per_10k"] = density["n_suppliers"] / density["BENES_TOTAL_CNT"] * 10000
density = density[np.isfinite(density["suppliers_per_10k"])]

county = county.merge(density[["state","suppliers_per_10k"]], on="state", how="left")
n_missing_density = county["suppliers_per_10k"].isna().sum()
county["suppliers_per_10k"] = county["suppliers_per_10k"].fillna(county["suppliers_per_10k"].median())
cleaning_log.append(("State coverage gap (access proxy)", "suppliers_per_10k", "Left-join audit against state_benes table",
                      f"{n_missing_density} of {len(county)} counties in states absent from the truncated State-level extract; median-imputed",
                      "The truncated Geographic Variation file does not include a State-level row for every state; documented as a data-completeness limitation"))

county.to_csv("/home/claude/interim_v2/data/clean_county_geographic.csv", index=False)
supplier.to_csv("/home/claude/interim_v2/data/clean_supplier.csv", index=False)

RESULTS["sample_size_summary"] = {
    "RQ1_required": 114, "RQ1_achieved": len(county),
    "RQ2_required": 180, "RQ2_achieved": len(county),
    "RQ3_required": 385, "RQ3_achieved": len(supplier),
}

# ============================================================
# 4. EDA
# ============================================================
desc_cols = ["DME_MDCR_STDZD_PYMT_PC","DME_EVNTS_PER_1000_BENES","BENE_AVG_AGE","BENE_DUAL_PCT",
             "BENE_FEML_PCT","BENE_RACE_BLACK_PCT","BENE_RACE_HSPNC_PCT","suppliers_per_10k"]
desc = county[desc_cols].describe().T.round(2)
desc.to_csv("/home/claude/interim_v2/data/eda_descriptive_stats.csv")
RESULTS["descriptive_stats"] = desc.to_dict(orient="index")

plt.figure(figsize=(6,4))
sns.histplot(county["DME_MDCR_STDZD_PYMT_PC"], kde=True, color="#2E6E9E")
plt.title(safe_title("Distribution of Standardized DME Spending per Beneficiary (Real CMS Data, 2024)"))
plt.xlabel("Standardized DME Medicare Payment per Capita ($)"); plt.ylabel("Number of Counties")
plt.tight_layout(); plt.savefig(FIG+"fig1_spending_distribution.png", dpi=150, bbox_inches="tight"); plt.close()

order = ["Northeast","Midwest","South","West"]
plt.figure(figsize=(6,4))
sns.boxplot(data=county, x="census_region", y="DME_EVNTS_PER_1000_BENES", order=order, palette="Blues")
plt.title(safe_title("DME Utilization Rate by Census Region (Real CMS Data, 2024)"))
plt.xlabel("Census Region"); plt.ylabel("DME Events per 1,000 Beneficiaries")
plt.tight_layout(); plt.savefig(FIG+"fig2_utilization_by_region.png", dpi=150, bbox_inches="tight"); plt.close()

corr_cols = ["DME_MDCR_STDZD_PYMT_PC","BENE_AVG_AGE","BENE_DUAL_PCT","BENE_FEML_PCT",
             "BENE_RACE_BLACK_PCT","BENE_RACE_HSPNC_PCT","suppliers_per_10k"]
plt.figure(figsize=(6.5,5.5))
sns.heatmap(county[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title(safe_title("Correlation Matrix of Key Predictors and DME Spending (Real Data)"))
plt.tight_layout(); plt.savefig(FIG+"fig3_correlation_heatmap.png", dpi=150, bbox_inches="tight"); plt.close()

plt.figure(figsize=(6,4))
sns.regplot(data=county, x="BENE_DUAL_PCT", y="DME_MDCR_STDZD_PYMT_PC", scatter_kws={"alpha":0.4,"s":18}, line_kws={"color":"red"})
plt.title(safe_title("Dual-Eligible Share vs. DME Spending per Beneficiary (Real Data)"))
plt.xlabel("Percent Dual-Eligible Beneficiaries"); plt.ylabel("DME Spending per Beneficiary ($)")
plt.tight_layout(); plt.savefig(FIG+"fig4_risk_vs_spending.png", dpi=150, bbox_inches="tight"); plt.close()

r_dual, p_dual = stats.pearsonr(county["BENE_DUAL_PCT"], county["DME_MDCR_STDZD_PYMT_PC"])
RESULTS["eda_dual_spending_corr"] = {"r": round(r_dual,3), "p": round(p_dual,6)}

# ============================================================
# 5. RQ1: Multiple Regression + Random Forest (REAL DATA)
# Note: the truncated Geographic Variation extract does not include an HCC risk
# score field, so BENE_AVG_AGE and BENE_DUAL_PCT serve as the real-data health-need
# proxies in its place (documented in Interim Limitations).
# ============================================================
county_m = pd.get_dummies(county, columns=["census_region"], drop_first=True)
region_dummy_cols = [c for c in county_m.columns if c.startswith("census_region_")]
features_rq1 = ["BENE_AVG_AGE","BENE_DUAL_PCT","BENE_FEML_PCT","BENE_RACE_BLACK_PCT",
                 "BENE_RACE_HSPNC_PCT","suppliers_per_10k"] + region_dummy_cols
X = county_m[features_rq1].astype(float)
y = county_m["DME_MDCR_STDZD_PYMT_PC"].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_sm = sm.add_constant(X_train)
ols = sm.OLS(y_train, X_train_sm).fit()
ols_table = pd.DataFrame({"coef": ols.params.round(3), "std_err": ols.bse.round(3), "t": ols.tvalues.round(3), "p_value": ols.pvalues.round(4)})
ols_table.to_csv("/home/claude/interim_v2/data/rq1_ols_summary.csv")

X_test_sm = sm.add_constant(X_test)
y_pred = ols.predict(X_test_sm)
r2_ols = r2_score(y_test, y_pred); rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred)); mae_ols = mean_absolute_error(y_test, y_pred)
n_t,k_t = X_test.shape
adj_r2_ols = 1-(1-r2_ols)*(n_t-1)/(n_t-k_t-1)

vif_data = pd.DataFrame({"feature": X_train.columns, "VIF":[variance_inflation_factor(X_train.values,i) for i in range(X_train.shape[1])]})
vif_data.to_csv("/home/claude/interim_v2/data/rq1_vif.csv", index=False)

rf = RandomForestRegressor(n_estimators=400, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf); rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf)); mae_rf = mean_absolute_error(y_test, y_pred_rf)
importances = pd.Series(rf.feature_importances_, index=features_rq1).sort_values(ascending=False)
importances.to_csv("/home/claude/interim_v2/data/rq1_rf_importance.csv")

plt.figure(figsize=(6,4))
importances.plot(kind="barh", color="#2E6E9E"); plt.gca().invert_yaxis()
plt.title(safe_title("Random Forest Feature Importance for DME Spending (RQ1, Real Data)"))
plt.xlabel("Relative Importance")
plt.tight_layout(); plt.savefig(FIG+"fig5_rf_importance.png", dpi=150, bbox_inches="tight"); plt.close()

RESULTS["RQ1"] = {"ols_r2_train": round(ols.rsquared,3), "ols_adj_r2_train": round(ols.rsquared_adj,3),
                   "test_r2_ols": round(r2_ols,3), "test_adj_r2_ols": round(adj_r2_ols,3),
                   "test_rmse_ols": round(rmse_ols,2), "test_mae_ols": round(mae_ols,2),
                   "test_r2_rf": round(r2_rf,3), "test_rmse_rf": round(rmse_rf,2), "test_mae_rf": round(mae_rf,2),
                   "significant_predictors_p05": list(ols.pvalues[ols.pvalues<0.05].index),
                   "top3_rf_features": list(importances.index[:3]), "max_vif": round(vif_data["VIF"].max(),2)}

# ============================================================
# 6. RQ2: ANOVA (utilization + access) + K-Means (REAL DATA)
# ============================================================
groups_util = [county[county.census_region==r]["DME_EVNTS_PER_1000_BENES"].values for r in order]
f_util,p_util = stats.f_oneway(*groups_util)
groups_acc = [county[county.census_region==r]["suppliers_per_10k"].values for r in order]
f_acc,p_acc = stats.f_oneway(*groups_acc)

def eta_sq(f, groups):
    k=len(groups); N=sum(len(g) for g in groups); dfb,dfw=k-1,N-k
    return (f*dfb)/(f*dfb+dfw)

RESULTS["RQ2a_utilization_anova"] = {"F": round(f_util,3), "p": round(p_util,6), "eta_sq": round(eta_sq(f_util,groups_util),3)}
RESULTS["RQ2b_access_anova"] = {"F": round(f_acc,3), "p": round(p_acc,6), "eta_sq": round(eta_sq(f_acc,groups_acc),3)}

cluster_feats = ["DME_MDCR_STDZD_PYMT_PC","DME_EVNTS_PER_1000_BENES","BENE_DUAL_PCT","BENE_AVG_AGE","suppliers_per_10k"]
Xc = StandardScaler().fit_transform(county[cluster_feats])
sils=[]
K_range=range(2,7)
for k in K_range:
    km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(Xc)
    sils.append(silhouette_score(Xc,km.labels_))

plt.figure(figsize=(6,4))
inertias=[KMeans(n_clusters=k,n_init=10,random_state=42).fit(Xc).inertia_ for k in K_range]
plt.plot(list(K_range), inertias, marker="o")
plt.title(safe_title("Elbow Method for Optimal K (RQ2 Clustering, Real Data)"))
plt.xlabel("Number of Clusters (k)"); plt.ylabel("Inertia (WCSS)")
plt.tight_layout(); plt.savefig(FIG+"fig6_elbow.png", dpi=150, bbox_inches="tight"); plt.close()

best_k = list(K_range)[int(np.argmax(sils))]
km_final = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Xc)
county["cluster"] = km_final.labels_
cluster_profile = county.groupby("cluster")[cluster_feats].mean().round(2)
cluster_profile.to_csv("/home/claude/interim_v2/data/rq2_cluster_profile.csv")

plt.figure(figsize=(6,4))
sns.scatterplot(data=county, x="suppliers_per_10k", y="DME_MDCR_STDZD_PYMT_PC", hue="cluster", palette="Set2", s=24, alpha=0.7)
plt.title(safe_title(f"County Clusters (k={best_k}): Access Proxy vs. Spending (Real Data)"))
plt.xlabel("Suppliers per 10,000 Beneficiaries (State-Level Proxy)"); plt.ylabel("DME Spending per Beneficiary ($)")
plt.tight_layout(); plt.savefig(FIG+"fig7_clusters.png", dpi=150, bbox_inches="tight"); plt.close()

RESULTS["RQ2_kmeans"] = {"best_k": best_k, "silhouette": round(max(sils),3), "cluster_sizes": county["cluster"].value_counts().sort_index().to_dict()}

# ============================================================
# 7. RQ3: Real supplier outlier detection (no synthetic ground truth now)
# ============================================================
sup_feats = ["DME_Tot_Suplr_Benes","DME_Tot_Suplr_Clms","DME_Suplr_Mdcr_Pymt_Amt","DME_Suplr_Mdcr_Stdzd_Pymt_Amt"]
sdat = supplier[sup_feats].copy()
q1s,q3s = sdat["DME_Suplr_Mdcr_Pymt_Amt"].quantile([0.25,0.75]); iqrs=q3s-q1s
iqr_flag = (sdat["DME_Suplr_Mdcr_Pymt_Amt"] > q3s+1.5*iqrs).astype(int)
iso = IsolationForest(contamination=0.035, random_state=42)
iso_flag = (iso.fit_predict(StandardScaler().fit_transform(sdat)) == -1).astype(int)
supplier["iqr_flag"]=iqr_flag; supplier["iso_flag"]=iso_flag
jaccard = (supplier["iqr_flag"] & supplier["iso_flag"]).sum() / ((supplier["iqr_flag"] | supplier["iso_flag"]).sum()+1e-9)

plt.figure(figsize=(6,4))
sns.scatterplot(data=supplier, x="DME_Tot_Suplr_Benes", y="DME_Suplr_Mdcr_Pymt_Amt", hue="iso_flag",
                 palette={0:"#2E6E9E",1:"#D62728"}, s=18, alpha=0.6)
plt.title(safe_title("Supplier Billing: Isolation Forest Flags (RQ3, Real Data)"))
plt.xlabel("DME Beneficiaries Served"); plt.ylabel("DME Medicare Payment ($)")
plt.tight_layout(); plt.savefig(FIG+"fig8_rq3_outliers.png", dpi=150, bbox_inches="tight"); plt.close()

RESULTS["RQ3_preliminary"] = {"n_suppliers": len(supplier), "pct_flagged_iqr": round(iqr_flag.mean()*100,2),
                               "pct_flagged_iso": round(iso_flag.mean()*100,2), "jaccard_similarity": round(jaccard,3),
                               "max_dme_payment_observed": round(float(sdat["DME_Suplr_Mdcr_Pymt_Amt"].max()),2)}

# ============================================================
# 8. RQ4: Preliminary HHI (STATE level only; category-level breakdown pending -- see limitations)
# ============================================================
hhi_rows=[]
for st, g in supplier.groupby("Suplr_Prvdr_State_Abrvtn"):
    total = g["DME_Suplr_Mdcr_Pymt_Amt"].sum()
    if total<=0 or len(g)<2: continue
    shares = g["DME_Suplr_Mdcr_Pymt_Amt"]/total*100
    hhi_rows.append({"state": st, "n_suppliers": len(g), "HHI": round((shares**2).sum(),1)})
hhi_df = pd.DataFrame(hhi_rows)
hhi_df.to_csv("/home/claude/interim_v2/data/rq4_hhi_preliminary.csv", index=False)

plt.figure(figsize=(6,4))
sns.histplot(hhi_df["HHI"], bins=20, color="#2E6E9E")
plt.axvline(1500, color="orange", linestyle="--", label="Moderate (1,500)")
plt.axvline(2500, color="red", linestyle="--", label="High (2,500)")
plt.title(safe_title("Distribution of State-Level DME Supplier HHI (RQ4, Preliminary, Real Data)"))
plt.xlabel("Herfindahl-Hirschman Index"); plt.ylabel("Number of States"); plt.legend()
plt.tight_layout(); plt.savefig(FIG+"fig9_hhi_distribution.png", dpi=150, bbox_inches="tight"); plt.close()

RESULTS["RQ4_preliminary"] = {"n_states": len(hhi_df), "mean_hhi": round(hhi_df["HHI"].mean(),1),
                               "pct_highly_concentrated": round((hhi_df["HHI"]>2500).mean()*100,2),
                               "pct_unconcentrated": round((hhi_df["HHI"]<1500).mean()*100,2)}

with open("/home/claude/interim_v2/data/results.json","w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(json.dumps(RESULTS, indent=2, default=str))
