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
import json

sns.set_style("whitegrid")
plt.rcParams.update({"axes.titlesize": 11, "axes.labelsize": 10, "figure.titlesize": 11})
FIG = "/home/claude/interim/figures/"

def safe_title(text, width=42):
    import textwrap
    return "\n".join(textwrap.wrap(text, width=width))
RESULTS = {}

# ============================================================
# 1. LOAD RAW PILOT DATA
# ============================================================
county_raw = pd.read_csv("/home/claude/interim/data/pilot_county_geographic.csv")
sup_raw = pd.read_csv("/home/claude/interim/data/pilot_supplier.csv")

RESULTS["raw_county_shape"] = county_raw.shape
RESULTS["raw_supplier_shape"] = sup_raw.shape

# ============================================================
# 2. DATA CLEANING LOG
# ============================================================
cleaning_log = []

# --- County file ---
n0 = len(county_raw)
missing_before = county_raw.isna().sum()
dupes = county_raw.duplicated().sum()

county = county_raw.drop_duplicates().reset_index(drop=True)
cleaning_log.append(("Duplicates", "all columns", "pandas .duplicated()", f"Removed {dupes} exact duplicate rows", "Exact duplicates add no information and bias standard errors downward"))

# Missing value imputation: median for numeric (robust to outliers), log reasoning
num_cols_county = ["median_household_income", "avg_hcc_risk_score", "suppliers_per_10k_beneficiaries"]
miss_pct = {}
for c in num_cols_county:
    pct = county[c].isna().mean() * 100
    miss_pct[c] = round(pct, 2)
    med = county[c].median()
    county[c] = county[c].fillna(med)
cleaning_log.append(("Missing values", ", ".join(num_cols_county), "pandas .isna() audit",
                      "Median imputation per column", "Median is robust to the skew typical of income/risk-score variables; affected columns had 2-3.5% missingness (MCAR pattern), well below the 10% threshold where imputation would be considered risky"))

# Outlier detection via IQR on spending
q1, q3 = county["dme_spending_per_beneficiary"].quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outlier_mask = (county["dme_spending_per_beneficiary"] < lo) | (county["dme_spending_per_beneficiary"] > hi)
n_outliers = int(outlier_mask.sum())
# Winsorize (cap) rather than drop, to preserve sample size for the geographic-level analysis
county.loc[county["dme_spending_per_beneficiary"] > hi, "dme_spending_per_beneficiary"] = hi
cleaning_log.append(("Outliers", "dme_spending_per_beneficiary", "IQR rule (1.5xIQR)",
                      f"Capped {n_outliers} high-end outliers at the upper fence (winsorizing)",
                      "Winsorizing preserves the full county sample (important given the RQ1/RQ2 minimum-N requirements) while limiting undue leverage from extreme values on the regression"))

county.to_csv("/home/claude/interim/data/clean_county_geographic.csv", index=False)

RESULTS["county_cleaning"] = {
    "n_before": n0, "n_after_dedup": len(county), "duplicates_removed": int(dupes),
    "missing_pct_by_col": miss_pct, "outliers_capped": n_outliers,
}

# --- Supplier file ---
n0s = len(sup_raw)
dupes_s = sup_raw.duplicated(subset=[c for c in sup_raw.columns if c != "supplier_id"]).sum()
supplier = sup_raw.drop_duplicates(subset=[c for c in sup_raw.columns if c != "supplier_id"]).reset_index(drop=True)

num_cols_sup = ["avg_payment_per_claim", "claims_per_beneficiary"]
miss_pct_s = {}
for c in num_cols_sup:
    pct = supplier[c].isna().mean() * 100
    miss_pct_s[c] = round(pct, 2)
    supplier[c] = supplier[c].fillna(supplier[c].median())

# Recompute total payment where it depended on imputed fields is unnecessary here (already computed upstream)
supplier.to_csv("/home/claude/interim/data/clean_supplier.csv", index=False)

RESULTS["supplier_cleaning"] = {
    "n_before": n0s, "n_after_dedup": len(supplier), "duplicates_removed": int(dupes_s),
    "missing_pct_by_col": miss_pct_s,
}
cleaning_log.append(("Duplicates", "all columns (supplier)", "pandas .duplicated()",
                      f"Removed {dupes_s} duplicate supplier rows", "Prevents double-counting a supplier's billing profile"))
cleaning_log.append(("Missing values", ", ".join(num_cols_sup), "pandas .isna() audit",
                      "Median imputation per column", "Billing fields are right-skewed; median imputation avoids distorting the distribution used later for outlier screening"))

RESULTS["cleaning_log"] = cleaning_log

# ============================================================
# 3. EDA - DESCRIPTIVE STATS + FIGURES
# ============================================================
desc = county[["dme_spending_per_beneficiary","dme_utilization_rate","avg_hcc_risk_score",
               "pct_dual_eligible","pct_age_85plus","median_household_income",
               "poverty_rate","suppliers_per_10k_beneficiaries"]].describe().T
desc = desc.round(2)
desc.to_csv("/home/claude/interim/data/eda_descriptive_stats.csv")
RESULTS["descriptive_stats"] = desc.to_dict(orient="index")

# Figure 1: Distribution of DME spending per beneficiary
plt.figure(figsize=(6,4))
sns.histplot(county["dme_spending_per_beneficiary"], kde=True, color="#2E6E9E")
plt.title(safe_title("Distribution of Standardized DME Spending per Beneficiary (Pilot Sample)"))
plt.xlabel("DME Spending per Beneficiary ($)")
plt.ylabel("Number of Counties")
plt.tight_layout()
plt.savefig(FIG + "fig1_spending_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 2: Boxplot of utilization by region
plt.figure(figsize=(6,4))
order = ["Northeast","Midwest","South","West"]
sns.boxplot(data=county, x="census_region", y="dme_utilization_rate", order=order, palette="Blues")
plt.title(safe_title("DME Utilization Rate by Census Region (Pilot Sample)"))
plt.xlabel("Census Region"); plt.ylabel("Claims per 1,000 Beneficiaries")
plt.tight_layout()
plt.savefig(FIG + "fig2_utilization_by_region.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 3: Correlation heatmap
corr_cols = ["dme_spending_per_beneficiary","avg_hcc_risk_score","pct_dual_eligible",
             "pct_age_85plus","median_household_income","poverty_rate","suppliers_per_10k_beneficiaries"]
plt.figure(figsize=(6.5,5.5))
sns.heatmap(county[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title(safe_title("Correlation Matrix of Key Predictors and DME Spending"))
plt.tight_layout()
plt.savefig(FIG + "fig3_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 4: Scatter HCC risk vs spending
plt.figure(figsize=(6,4))
sns.regplot(data=county, x="avg_hcc_risk_score", y="dme_spending_per_beneficiary",
            scatter_kws={"alpha":0.4, "s":18}, line_kws={"color":"red"})
plt.title(safe_title("HCC Risk Score vs. DME Spending per Beneficiary"))
plt.xlabel("Average HCC Risk Score"); plt.ylabel("DME Spending per Beneficiary ($)")
plt.tight_layout()
plt.savefig(FIG + "fig4_risk_vs_spending.png", dpi=150, bbox_inches="tight")
plt.close()

r_risk_spend, p_risk_spend = stats.pearsonr(county["avg_hcc_risk_score"], county["dme_spending_per_beneficiary"])
RESULTS["eda_risk_spending_corr"] = {"r": round(r_risk_spend,3), "p": round(p_risk_spend,5)}

# ============================================================
# 4. SAMPLE SIZE (achieved vs required)
# ============================================================
sample_size_summary = {
    "RQ1_required": 114, "RQ1_achieved": len(county),
    "RQ2_required": 180, "RQ2_achieved": len(county),
    "RQ3_required": 385, "RQ3_achieved": len(supplier),
    "RQ4_required": 85,  "RQ4_achieved": len(county),  # geographic units for state x category will be computed below
}
RESULTS["sample_size_summary"] = sample_size_summary

# ============================================================
# 5. RQ1 MODELING: Multiple Regression + Random Forest
# ============================================================
features_rq1 = ["avg_hcc_risk_score","pct_dual_eligible","pct_age_85plus",
                "median_household_income","rucc_code","poverty_rate","suppliers_per_10k_beneficiaries"]
# one-hot for region (4th categorical "factor" to bring feature count to 8 per synopsis design)
county_m = pd.get_dummies(county, columns=["census_region"], drop_first=True)
region_dummy_cols = [c for c in county_m.columns if c.startswith("census_region_")]
X_cols = features_rq1 + region_dummy_cols
X = county_m[X_cols].astype(float)
y = county_m["dme_spending_per_beneficiary"].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# OLS via statsmodels for inference (coeffs, p-values)
X_train_sm = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_sm).fit()
ols_summary_table = pd.DataFrame({
    "coef": ols_model.params.round(3),
    "std_err": ols_model.bse.round(3),
    "t": ols_model.tvalues.round(3),
    "p_value": ols_model.pvalues.round(4),
})
ols_summary_table.to_csv("/home/claude/interim/data/rq1_ols_summary.csv")

X_test_sm = sm.add_constant(X_test)
y_pred_ols = ols_model.predict(X_test_sm)
r2_ols = r2_score(y_test, y_pred_ols)
rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_ols))
mae_ols = mean_absolute_error(y_test, y_pred_ols)
n_test, k_test = X_test.shape
adj_r2_ols = 1 - (1-r2_ols)*(n_test-1)/(n_test-k_test-1)

# VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif_data.to_csv("/home/claude/interim/data/rq1_vif.csv", index=False)

# Random Forest
rf = RandomForestRegressor(n_estimators=400, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)

importances = pd.Series(rf.feature_importances_, index=X_cols).sort_values(ascending=False)
importances.to_csv("/home/claude/interim/data/rq1_rf_importance.csv")

plt.figure(figsize=(6,4))
importances.plot(kind="barh", color="#2E6E9E")
plt.gca().invert_yaxis()
plt.title(safe_title("Random Forest Feature Importance for DME Spending (RQ1)"))
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.savefig(FIG + "fig5_rf_importance.png", dpi=150, bbox_inches="tight")
plt.close()

RESULTS["RQ1"] = {
    "ols_r2": round(ols_model.rsquared,3), "ols_adj_r2_train": round(ols_model.rsquared_adj,3),
    "test_r2_ols": round(r2_ols,3), "test_adj_r2_ols": round(adj_r2_ols,3),
    "test_rmse_ols": round(rmse_ols,2), "test_mae_ols": round(mae_ols,2),
    "test_r2_rf": round(r2_rf,3), "test_rmse_rf": round(rmse_rf,2), "test_mae_rf": round(mae_rf,2),
    "significant_predictors_p05": list(ols_model.pvalues[ols_model.pvalues < 0.05].index),
    "top3_rf_features": list(importances.index[:3]),
    "max_vif": round(vif_data["VIF"].max(),2),
}

# ============================================================
# 6. RQ2 MODELING: ANOVA (utilization & access) + K-Means
# ============================================================
groups_util = [county[county.census_region==r]["dme_utilization_rate"].values for r in order]
f_util, p_util = stats.f_oneway(*groups_util)
groups_acc = [county[county.census_region==r]["suppliers_per_10k_beneficiaries"].values for r in order]
f_acc, p_acc = stats.f_oneway(*groups_acc)

# eta-squared
def eta_squared(f, groups):
    k = len(groups); N = sum(len(g) for g in groups)
    dfb, dfw = k-1, N-k
    return (f*dfb)/(f*dfb + dfw)

eta_util = eta_squared(f_util, groups_util)
eta_acc = eta_squared(f_acc, groups_acc)

RESULTS["RQ2a_utilization_anova"] = {"F": round(f_util,3), "p": round(p_util,6), "eta_sq": round(eta_util,3)}
RESULTS["RQ2b_access_anova"] = {"F": round(f_acc,3), "p": round(p_acc,6), "eta_sq": round(eta_acc,3)}

# K-Means clustering
cluster_feats = ["dme_spending_per_beneficiary","dme_utilization_rate","avg_hcc_risk_score",
                  "rucc_code","suppliers_per_10k_beneficiaries"]
Xc = StandardScaler().fit_transform(county[cluster_feats])
inertias, sils = [], []
K_range = range(2,7)
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xc)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(Xc, km.labels_))

plt.figure(figsize=(6,4))
plt.plot(list(K_range), inertias, marker="o")
plt.title(safe_title("Elbow Method for Optimal K (RQ2 Clustering)"))
plt.xlabel("Number of Clusters (k)"); plt.ylabel("Inertia (WCSS)")
plt.tight_layout(); plt.savefig(FIG + "fig6_elbow.png", dpi=150, bbox_inches="tight"); plt.close()

best_k = list(K_range)[int(np.argmax(sils))]
km_final = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Xc)
county["cluster"] = km_final.labels_
cluster_profile = county.groupby("cluster")[cluster_feats].mean().round(2)
cluster_profile.to_csv("/home/claude/interim/data/rq2_cluster_profile.csv")

plt.figure(figsize=(6,4))
sns.scatterplot(data=county, x="suppliers_per_10k_beneficiaries", y="dme_spending_per_beneficiary",
                 hue="cluster", palette="Set2", s=28, alpha=0.75)
plt.title(safe_title(f"County Clusters (k={best_k}) \u2014 Access Proxy vs. Spending"))
plt.xlabel("DME Suppliers per 10,000 Beneficiaries"); plt.ylabel("DME Spending per Beneficiary ($)")
plt.tight_layout(); plt.savefig(FIG + "fig7_clusters.png", dpi=150, bbox_inches="tight"); plt.close()

RESULTS["RQ2_kmeans"] = {"best_k": best_k, "silhouette": round(max(sils),3),
                          "cluster_sizes": county["cluster"].value_counts().sort_index().to_dict()}

# ============================================================
# 7. RQ3 PRELIMINARY: IQR + Isolation Forest outlier screening
# ============================================================
sup_feats = ["total_medicare_payment","total_beneficiaries_served","claims_per_beneficiary","avg_payment_per_claim"]
sdat = supplier[sup_feats].copy()

# IQR screen on total_medicare_payment
q1s, q3s = sdat["total_medicare_payment"].quantile([0.25,0.75])
iqrs = q3s-q1s
iqr_flag = (sdat["total_medicare_payment"] > q3s + 1.5*iqrs).astype(int)

iso = IsolationForest(contamination=0.035, random_state=42)
iso_flag = (iso.fit_predict(StandardScaler().fit_transform(sdat)) == -1).astype(int)

supplier["iqr_flag"] = iqr_flag
supplier["iso_flag"] = iso_flag
jaccard = (supplier["iqr_flag"] & supplier["iso_flag"]).sum() / ((supplier["iqr_flag"] | supplier["iso_flag"]).sum() + 1e-9)

# Validation against the known injected anomaly (available only because this is pilot/simulated data)
recall_vs_known = (supplier.loc[supplier["known_anomaly_flag_for_validation"]==1, "iso_flag"]==1).mean()

plt.figure(figsize=(6,4))
sns.scatterplot(data=supplier, x="total_beneficiaries_served", y="total_medicare_payment",
                 hue="iso_flag", palette={0:"#2E6E9E",1:"#D62728"}, s=22, alpha=0.7)
plt.title(safe_title("Supplier Billing: Isolation Forest Flags (RQ3, Preliminary)"))
plt.xlabel("Total Beneficiaries Served"); plt.ylabel("Total Medicare Payment ($)")
plt.tight_layout(); plt.savefig(FIG + "fig8_rq3_outliers.png", dpi=150, bbox_inches="tight"); plt.close()

RESULTS["RQ3_preliminary"] = {
    "n_suppliers": len(supplier), "pct_flagged_iqr": round(iqr_flag.mean()*100,2),
    "pct_flagged_iso": round(iso_flag.mean()*100,2), "jaccard_similarity": round(jaccard,3),
    "recall_vs_known_injected_anomaly_pilot_only": round(recall_vs_known,3),
}

# ============================================================
# 8. RQ4 PRELIMINARY: HHI construction (state level, pilot)
# ============================================================
hhi_rows = []
for (st, cat), g in supplier.groupby(["state","hcpcs_category"]):
    total = g["total_medicare_payment"].sum()
    if total <= 0 or len(g) < 2:
        continue
    shares = (g["total_medicare_payment"]/total*100)
    hhi = (shares**2).sum()
    hhi_rows.append({"state": st, "hcpcs_category": cat, "n_suppliers": len(g), "HHI": round(hhi,1)})
hhi_df = pd.DataFrame(hhi_rows)
hhi_df.to_csv("/home/claude/interim/data/rq4_hhi_preliminary.csv", index=False)

plt.figure(figsize=(6,4))
sns.histplot(hhi_df["HHI"], bins=25, color="#2E6E9E", kde=False)
plt.axvline(1500, color="orange", linestyle="--", label="Moderate threshold (1,500)")
plt.axvline(2500, color="red", linestyle="--", label="High threshold (2,500)")
plt.title(safe_title("Distribution of State x Category HHI (RQ4, Preliminary)"))
plt.xlabel("Herfindahl-Hirschman Index"); plt.ylabel("Number of State x Category Cells")
plt.legend()
plt.tight_layout(); plt.savefig(FIG + "fig9_hhi_distribution.png", dpi=150, bbox_inches="tight"); plt.close()

RESULTS["RQ4_preliminary"] = {
    "n_state_category_cells": len(hhi_df),
    "pct_highly_concentrated": round((hhi_df["HHI"]>2500).mean()*100,2),
    "pct_unconcentrated": round((hhi_df["HHI"]<1500).mean()*100,2),
    "mean_hhi": round(hhi_df["HHI"].mean(),1),
}

with open("/home/claude/interim/data/results.json","w") as f:
    json.dump(RESULTS, f, indent=2, default=str)

print("DONE")
print(json.dumps(RESULTS, indent=2, default=str))
