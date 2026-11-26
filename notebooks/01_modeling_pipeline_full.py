import glob
import numpy as np
import pandas as pd
import json, textwrap
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

def safe_title(t, w=46): return "\n".join(textwrap.wrap(t, width=w))
def save(n): plt.tight_layout(); plt.savefig(FIG+n, dpi=200, bbox_inches="tight"); plt.close()

with open(DATA + "results_stage1.json") as f:
    RESULTS = json.load(f)

county = pd.read_csv(DATA + "clean_county_geographic.csv")
supplier = pd.read_csv(DATA + "clean_supplier.csv")
order = ["Northeast","Midwest","South","West"]

ref_cols = ["Rfrg_Prvdr_State_Abrvtn", "RBCS_Desc", "Tot_Suplr_Clms", "Avg_Suplr_Mdcr_Pymt_Amt"]
ref_files = sorted(glob.glob(DATA + "Medicare_DME_Devices_Supplies_by_Referring_Provider_and_Service_2024_part_*.csv"))
ref_raw = pd.concat([pd.read_csv(f, usecols=ref_cols, low_memory=False) for f in ref_files], ignore_index=True)

region_map = {}
ne = {"CT","ME","MA","NH","NJ","NY","PA","RI","VT"}
mw = {"IL","IN","IA","KS","MI","MN","MO","NE","ND","OH","SD","WI"}
south = {"AL","AR","DE","FL","GA","KY","LA","MD","MS","NC","OK","SC","TN","TX","VA","WV","DC"}
for s_ in list(ne)+list(mw)+list(south):
    region_map[s_] = "Northeast" if s_ in ne else "Midwest" if s_ in mw else "South"

# ============================================================
# EDA (overall + subgroup)
# ============================================================
desc_cols = ["DME_MDCR_STDZD_PYMT_PC","DME_EVNTS_PER_1000_BENES","BENE_AVG_AGE","BENE_DUAL_PCT",
             "BENE_FEML_PCT","BENE_RACE_BLACK_PCT","BENE_RACE_HSPNC_PCT","suppliers_per_10k","median_hh_income","poverty_rate"]
desc_overall = county[desc_cols].describe().T.round(2)
desc_overall.to_csv(DATA + "eda_overall_stats.csv")
region_stats = county.groupby("census_region").agg(
    n=("state","size"), n_states=("state","nunique"),
    spend_mean=("DME_MDCR_STDZD_PYMT_PC","mean"), spend_sd=("DME_MDCR_STDZD_PYMT_PC","std"),
    util_mean=("DME_EVNTS_PER_1000_BENES","mean"), util_sd=("DME_EVNTS_PER_1000_BENES","std"),
    access_mean=("suppliers_per_10k","mean"), access_sd=("suppliers_per_10k","std"),
    income_mean=("median_hh_income","mean"), poverty_mean=("poverty_rate","mean"),
).loc[order].round(2)
region_stats.to_csv(DATA + "eda_region_stats.csv")
RESULTS["eda"] = {"overall": desc_overall.to_dict(orient="index"), "by_region": region_stats.to_dict(orient="index")}

plt.figure(figsize=(6.5,4)); sns.histplot(county["DME_MDCR_STDZD_PYMT_PC"], kde=True, color="#2E6E9E", bins=30)
plt.title(safe_title("Distribution of Standardized DME Spending per Beneficiary (2024, Full Data)"))
plt.xlabel("Standardized DME Medicare payment per beneficiary ($)"); plt.ylabel("Number of counties"); save("fig1_spending_distribution.png")

plt.figure(figsize=(6.5,4)); sns.boxplot(data=county, x="census_region", y="DME_MDCR_STDZD_PYMT_PC", order=order, palette="Blues")
plt.title(safe_title("DME Spending per Beneficiary by Census Region")); plt.xlabel("Census region"); plt.ylabel("DME spending per beneficiary ($)")
save("fig2_spending_by_region.png")

plt.figure(figsize=(6.5,4)); sns.boxplot(data=county, x="census_region", y="DME_EVNTS_PER_1000_BENES", order=order, palette="Blues")
plt.title(safe_title("DME Utilization Rate by Census Region")); plt.xlabel("Census region"); plt.ylabel("DME events per 1,000 beneficiaries")
save("fig3_utilization_by_region.png")

corr_cols = ["DME_MDCR_STDZD_PYMT_PC","BENE_AVG_AGE","BENE_DUAL_PCT","BENE_FEML_PCT","BENE_RACE_BLACK_PCT","BENE_RACE_HSPNC_PCT","suppliers_per_10k","median_hh_income","poverty_rate"]
corr_labels = ["Spending","Avg. age","Dual-elig.","Female","Black","Hispanic","Suppliers/10k","Income","Poverty"]
cm = county[corr_cols].corr(); cm.index = corr_labels; cm.columns = corr_labels
plt.figure(figsize=(7,5.6)); sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-.6, vmax=1)
plt.title(safe_title("Correlation Matrix of Key Predictors and DME Spending")); plt.xticks(rotation=40, ha="right")
save("fig4_correlation_heatmap.png")

plt.figure(figsize=(6.5,4)); sns.regplot(data=county, x="poverty_rate", y="DME_MDCR_STDZD_PYMT_PC", scatter_kws={"alpha":.3,"s":14}, line_kws={"color":"red"})
plt.title(safe_title("Poverty Rate vs. DME Spending per Beneficiary")); plt.xlabel("County poverty rate"); plt.ylabel("DME spending per beneficiary ($)")
save("fig5_poverty_vs_spending.png")

r_pov, p_pov = stats.pearsonr(county["poverty_rate"], county["DME_MDCR_STDZD_PYMT_PC"])
r_inc, p_inc = stats.pearsonr(county["median_hh_income"], county["DME_MDCR_STDZD_PYMT_PC"])
RESULTS["eda_poverty_spending_corr"] = {"r": round(r_pov,3), "p": round(p_pov,6)}
RESULTS["eda_income_spending_corr"] = {"r": round(r_inc,3), "p": round(p_inc,6)}

# ============================================================
# RQ1: 11 predictors now (added median_hh_income, poverty_rate)
# ============================================================
county_m = pd.get_dummies(county, columns=["census_region"], drop_first=True)
region_dummy_cols = [c for c in county_m.columns if c.startswith("census_region_")]
FEATURES_RQ1 = ["BENE_AVG_AGE","BENE_DUAL_PCT","BENE_FEML_PCT","BENE_RACE_BLACK_PCT","BENE_RACE_HSPNC_PCT",
                 "suppliers_per_10k","median_hh_income","poverty_rate"] + region_dummy_cols
K_RQ1 = len(FEATURES_RQ1)
assert K_RQ1 == 11, f"Expected 11 predictors for RQ1, got {K_RQ1}"

X = county_m[FEATURES_RQ1].astype(float)
y = county_m["DME_MDCR_STDZD_PYMT_PC"].astype(float)
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42)

def eval_set(fn, X_, y_):
    pred = fn(X_)
    return r2_score(y_, pred), np.sqrt(mean_squared_error(y_, pred)), mean_absolute_error(y_, pred), pred

baseline_pred = np.full(len(y_test), y_train.mean())
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred)); baseline_mae = mean_absolute_error(y_test, baseline_pred)

Xtr_sm = sm.add_constant(X_train)
ols = sm.OLS(y_train, Xtr_sm).fit()
def ols_pred(Xd): return ols.predict(sm.add_constant(Xd, has_constant="add"))
ols_train = eval_set(ols_pred, X_train, y_train)
ols_val = eval_set(ols_pred, X_val, y_val)
ols_test = eval_set(ols_pred, X_test, y_test)

vif_data = pd.DataFrame({"feature": X_train.columns, "VIF":[variance_inflation_factor(X_train.values,i) for i in range(X_train.shape[1])]})
vif_data.to_csv(DATA + "rq1_vif.csv", index=False)

rf = RandomForestRegressor(n_estimators=500, max_depth=7, random_state=42)
rf.fit(X_train, y_train)
rf_train = eval_set(rf.predict, X_train, y_train)
rf_val = eval_set(rf.predict, X_val, y_val)
rf_test = eval_set(rf.predict, X_test, y_test)

importances = pd.Series(rf.feature_importances_, index=FEATURES_RQ1).sort_values(ascending=False)
importances.to_csv(DATA + "rq1_rf_importance.csv")
plt.figure(figsize=(6.5,4.4)); importances.plot(kind="barh", color="#2E6E9E"); plt.gca().invert_yaxis()
plt.title(safe_title("Random Forest Feature Importance for DME Spending (RQ1)")); plt.xlabel("Relative importance")
save("fig6_rf_importance.png")

resid_test = y_test - ols_pred(X_test)
plt.figure(figsize=(6.5,4.2)); plt.scatter(ols_pred(X_test), resid_test, alpha=0.4, s=14, color="#2E6E9E"); plt.axhline(0, color="red", linestyle="--")
plt.title(safe_title("RQ1 OLS Residuals vs. Fitted Values (Test Set)")); plt.xlabel("Fitted DME spending ($)"); plt.ylabel("Residual ($)")
save("fig7_rq1_residuals.png")

plt.figure(figsize=(6.5,4.2))
plt.scatter(y_test, ols_pred(X_test), alpha=0.35, s=14, color="#2E6E9E", label="OLS")
plt.scatter(y_test, rf.predict(X_test), alpha=0.3, s=14, color="#D4A017", label="Random Forest")
lims = [min(y_test.min(), ols_pred(X_test).min()), max(y_test.max(), ols_pred(X_test).max())]
plt.plot(lims, lims, color="gray", linestyle="--")
plt.title(safe_title("RQ1 Actual vs. Predicted DME Spending (Test Set)")); plt.xlabel("Actual ($)"); plt.ylabel("Predicted ($)"); plt.legend()
save("fig8_rq1_actual_vs_predicted.png")

RESULTS["RQ1"] = {
    "k_predictors": K_RQ1, "split": "60/20/20", "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
    "baseline_test_rmse": round(baseline_rmse,2), "baseline_test_mae": round(baseline_mae,2),
    "ols_train_r2": round(ols_train[0],3), "ols_train_rmse": round(ols_train[1],2), "ols_train_mae": round(ols_train[2],2),
    "ols_val_r2": round(ols_val[0],3), "ols_val_rmse": round(ols_val[1],2), "ols_val_mae": round(ols_val[2],2),
    "ols_test_r2": round(ols_test[0],3), "ols_test_rmse": round(ols_test[1],2), "ols_test_mae": round(ols_test[2],2),
    "rf_train_r2": round(rf_train[0],3), "rf_train_rmse": round(rf_train[1],2), "rf_train_mae": round(rf_train[2],2),
    "rf_val_r2": round(rf_val[0],3), "rf_val_rmse": round(rf_val[1],2), "rf_val_mae": round(rf_val[2],2),
    "rf_test_r2": round(rf_test[0],3), "rf_test_rmse": round(rf_test[1],2), "rf_test_mae": round(rf_test[2],2),
    "max_vif": round(vif_data["VIF"].max(),2),
    "significant_predictors_p05": list(ols.pvalues[ols.pvalues<0.05].index),
    "top3_rf_features": list(importances.index[:3]),
}

# ============================================================
# RQ2: ANOVA + K-Means
# ============================================================
groups_util = [county[county.census_region==r]["DME_EVNTS_PER_1000_BENES"].values for r in order]
f_util,p_util = stats.f_oneway(*groups_util)
groups_acc = [county[county.census_region==r]["suppliers_per_10k"].values for r in order]
f_acc,p_acc = stats.f_oneway(*groups_acc)
def eta_sq(f, groups):
    k=len(groups); N=sum(len(g) for g in groups); dfb,dfw=k-1,N-k
    return (f*dfb)/(f*dfb+dfw)

cluster_feats = ["DME_MDCR_STDZD_PYMT_PC","DME_EVNTS_PER_1000_BENES","BENE_DUAL_PCT","BENE_AVG_AGE","suppliers_per_10k","poverty_rate"]
Xc = StandardScaler().fit_transform(county[cluster_feats])
sils=[]; K_range=range(2,7)
for k in K_range:
    km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(Xc); sils.append(silhouette_score(Xc,km.labels_))
best_k = list(K_range)[int(np.argmax(sils))]
km_final = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Xc)
county["cluster"] = km_final.labels_
county.to_csv(DATA + "county_with_cluster.csv", index=False)

plt.figure(figsize=(6.5,4.2))
plt.plot(list(K_range), [KMeans(n_clusters=k,n_init=10,random_state=42).fit(Xc).inertia_ for k in K_range], marker="o")
plt.title(safe_title("Elbow Method for Optimal K (RQ2 Clustering)")); plt.xlabel("Number of clusters (k)"); plt.ylabel("Inertia (WCSS)")
save("fig9_elbow.png")

plt.figure(figsize=(6.5,4.2))
sns.scatterplot(data=county, x="suppliers_per_10k", y="DME_MDCR_STDZD_PYMT_PC", hue="cluster", palette="Set2", s=20, alpha=0.6)
plt.title(safe_title(f"County Clusters (k={best_k}): Access Proxy vs. Spending")); plt.xlabel("Suppliers per 10,000 beneficiaries"); plt.ylabel("DME spending per beneficiary ($)")
save("fig10_clusters.png")

RESULTS["RQ2a_utilization_anova"] = {"F": round(f_util,3), "p": round(p_util,6), "eta_sq": round(eta_sq(f_util,groups_util),3)}
RESULTS["RQ2b_access_anova"] = {"F": round(f_acc,3), "p": round(p_acc,6), "eta_sq": round(eta_sq(f_acc,groups_acc),3)}
RESULTS["RQ2_kmeans"] = {"best_k": best_k, "silhouette": round(max(sils),3), "cluster_sizes": county["cluster"].value_counts().sort_index().to_dict()}

# ============================================================
# RQ3: IQR + Isolation Forest + Mann-Whitney + chi-square + risk-score comparison
# ============================================================
sup_feats = ["DME_Tot_Suplr_Benes","DME_Tot_Suplr_Clms","DME_Suplr_Mdcr_Pymt_Amt","DME_Suplr_Mdcr_Stdzd_Pymt_Amt"]
sdat = supplier[sup_feats].copy()
q1s,q3s = sdat["DME_Suplr_Mdcr_Pymt_Amt"].quantile([0.25,0.75]); iqrs=q3s-q1s
iqr_flag = (sdat["DME_Suplr_Mdcr_Pymt_Amt"] > q3s+1.5*iqrs).astype(int)
iso = IsolationForest(contamination=0.035, random_state=42)
iso_flag = (iso.fit_predict(StandardScaler().fit_transform(sdat)) == -1).astype(int)
supplier["iqr_flag"]=iqr_flag; supplier["iso_flag"]=iso_flag
jaccard = (supplier["iqr_flag"] & supplier["iso_flag"]).sum() / ((supplier["iqr_flag"] | supplier["iso_flag"]).sum()+1e-9)

agree = pd.crosstab(supplier["iqr_flag"], supplier["iso_flag"], rownames=["IQR flag"], colnames=["Isolation Forest flag"])
agree.to_csv(DATA + "rq3_agreement_matrix.csv")
plt.figure(figsize=(5.2,4.2)); sns.heatmap(agree, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title(safe_title("RQ3 Method-Agreement Matrix: IQR vs. Isolation Forest Flags"))
plt.ylabel("IQR flag (0/1)"); plt.xlabel("Isolation Forest flag (0/1)"); save("fig11_rq3_agreement_matrix.png")

mw_results = {}
for c in sup_feats[:3]:
    u = stats.mannwhitneyu(supplier.loc[supplier.iso_flag==1, c], supplier.loc[supplier.iso_flag==0, c])
    mw_results[c] = {"U": float(u.statistic), "p": float(u.pvalue)}
u_risk = stats.mannwhitneyu(supplier.loc[supplier.iso_flag==1, "Bene_Avg_Risk_Scre"], supplier.loc[supplier.iso_flag==0, "Bene_Avg_Risk_Scre"])
mw_results["Bene_Avg_Risk_Scre"] = {"U": float(u_risk.statistic), "p": float(u_risk.pvalue)}
risk_flagged_mean = float(supplier.loc[supplier.iso_flag==1, "Bene_Avg_Risk_Scre"].mean())
risk_nonflagged_mean = float(supplier.loc[supplier.iso_flag==0, "Bene_Avg_Risk_Scre"].mean())

def reg(st):
    if st in region_map: return region_map[st]
    if st in ["AK","AZ","CA","CO","HI","ID","MT","NV","NM","OR","UT","WA","WY"]: return "West"
    return None
supplier["region"] = supplier["Suplr_Prvdr_State_Abrvtn"].apply(reg)
sr = supplier.dropna(subset=["region"])
ct = pd.crosstab(sr["region"], sr["iso_flag"])
chi2, chi_p, dof, _ = stats.chi2_contingency(ct)
cramers_v = np.sqrt(chi2 / len(sr) / (min(ct.shape)-1))

plt.figure(figsize=(6,4))
sns.scatterplot(data=supplier, x="DME_Tot_Suplr_Benes", y="DME_Suplr_Mdcr_Pymt_Amt", hue="iso_flag", palette={0:"#2E6E9E",1:"#D62728"}, s=14, alpha=0.5)
plt.title(safe_title("Supplier Billing: Isolation Forest Flags (RQ3)")); plt.xlabel("DME beneficiaries served"); plt.ylabel("DME Medicare payment ($)")
save("fig12_rq3_outliers.png")

RESULTS["RQ3"] = {"n_suppliers": len(supplier), "pct_flagged_iqr": round(iqr_flag.mean()*100,2), "pct_flagged_iso": round(iso_flag.mean()*100,2),
                   "jaccard_similarity": round(jaccard,3), "max_dme_payment_observed": round(float(sdat["DME_Suplr_Mdcr_Pymt_Amt"].max()),2),
                   "mann_whitney": mw_results, "risk_score_flagged_mean": round(risk_flagged_mean,3), "risk_score_nonflagged_mean": round(risk_nonflagged_mean,3),
                   "chi_square": {"chi2": round(chi2,3), "p": round(chi_p,6), "dof": int(dof), "cramers_v": round(cramers_v,3)}}

# ============================================================
# RQ4a: Supplier-level state HHI (primary, supply-side)
# ============================================================
hhi_rows=[]
for st, g in supplier.groupby("Suplr_Prvdr_State_Abrvtn"):
    total = g["DME_Suplr_Mdcr_Pymt_Amt"].sum()
    if total<=0 or len(g)<2: continue
    shares = g["DME_Suplr_Mdcr_Pymt_Amt"]/total*100
    hhi_rows.append({"state": st, "n_suppliers": len(g), "HHI": round((shares**2).sum(),1)})
hhi_df = pd.DataFrame(hhi_rows)
hhi_df.to_csv(DATA + "rq4_hhi_state.csv", index=False)

plt.figure(figsize=(6.5,4.2))
sns.histplot(hhi_df["HHI"], bins=25, color="#2E6E9E")
plt.axvline(1500, color="orange", linestyle="--", label="Moderate (1,500)"); plt.axvline(2500, color="red", linestyle="--", label="High (2,500)")
plt.title(safe_title("Distribution of State-Level DME Supplier HHI")); plt.xlabel("HHI"); plt.ylabel("Number of states"); plt.legend()
save("fig13_hhi_state_distribution.png")

# ============================================================
# RQ4b: Referring-provider state x RBCS-category HHI (NEW -- satisfies state x category granularity)
# ============================================================
ref = ref_raw[ref_raw["RBCS_Desc"].str.startswith("DME-", na=False)].copy()
ref["region"] = ref["Rfrg_Prvdr_State_Abrvtn"]
ref["est_payment"] = ref["Avg_Suplr_Mdcr_Pymt_Amt"] * ref["Tot_Suplr_Clms"]
cat_hhi_rows = []
for (st, cat), g in ref.groupby(["Rfrg_Prvdr_State_Abrvtn","RBCS_Desc"]):
    total = g["est_payment"].sum()
    if total<=0 or len(g)<2: continue
    shares = g["est_payment"]/total*100
    cat_hhi_rows.append({"state": st, "category": cat, "n_referring_providers": len(g), "HHI": round((shares**2).sum(),1)})
cat_hhi_df = pd.DataFrame(cat_hhi_rows)
cat_hhi_df.to_csv(DATA + "rq4_hhi_state_category_referring.csv", index=False)

plt.figure(figsize=(6.5,4.2))
sns.histplot(cat_hhi_df["HHI"], bins=30, color="#6A4C93")
plt.axvline(1500, color="orange", linestyle="--", label="Moderate (1,500)"); plt.axvline(2500, color="red", linestyle="--", label="High (2,500)")
plt.title(safe_title("Distribution of State x Category Referring-Provider HHI")); plt.xlabel("HHI"); plt.ylabel("Number of state x category cells"); plt.legend()
save("fig14_hhi_category_distribution.png")

RESULTS["RQ4_supplier_state"] = {"n_states": len(hhi_df), "mean_hhi": round(hhi_df["HHI"].mean(),1),
                   "pct_highly_concentrated": round((hhi_df["HHI"]>2500).mean()*100,2), "pct_unconcentrated": round((hhi_df["HHI"]<1500).mean()*100,2)}
RESULTS["RQ4_referring_provider_category"] = {"n_cells": len(cat_hhi_df), "mean_hhi": round(cat_hhi_df["HHI"].mean(),1),
                   "pct_highly_concentrated": round((cat_hhi_df["HHI"]>2500).mean()*100,2), "pct_unconcentrated": round((cat_hhi_df["HHI"]<1500).mean()*100,2),
                   "n_categories": ref["RBCS_Desc"].nunique(), "n_states": ref["Rfrg_Prvdr_State_Abrvtn"].nunique()}

with open(DATA + "results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("STAGE 2 DONE")
print(json.dumps(RESULTS["RQ1"], indent=2, default=str))
print(json.dumps(RESULTS["RQ4_referring_provider_category"], indent=2, default=str))
