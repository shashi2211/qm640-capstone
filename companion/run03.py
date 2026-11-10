import numpy as np, pandas as pd, json
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
D="data/"
c=pd.read_csv(D+"clean_county_geographic.csv"); s=pd.read_csv(D+"clean_supplier.csv")
out={}
cols={"DME_MDCR_STDZD_PYMT_PC":"Spending per beneficiary ($)","DME_EVNTS_PER_1000_BENES":"Utilization (events/1,000)",
"BENE_AVG_AGE":"Average age","BENE_DUAL_PCT":"Dual-eligible share","BENE_FEML_PCT":"Female share",
"BENE_RACE_BLACK_PCT":"Black share","BENE_RACE_HSPNC_PCT":"Hispanic share","suppliers_per_10k":"Suppliers per 10k"}
rows=[]
for k,v in cols.items():
    x=c[k]; rows.append([v,len(x),x.mean(),x.std(),x.median(),x.min(),x.max(),stats.skew(x)])
d1=pd.DataFrame(rows,columns=["var","n","mean","sd","median","min","max","skew"]); d1.to_csv(D+"t_desc_overall.csv",index=False); print(d1.round(3))
order=["Northeast","Midwest","South","West"]
g=c.groupby("census_region").agg(n=("state","size"),states=("state","nunique"),sp_m=("DME_MDCR_STDZD_PYMT_PC","mean"),sp_sd=("DME_MDCR_STDZD_PYMT_PC","std"),sp_med=("DME_MDCR_STDZD_PYMT_PC","median"),
 ut_m=("DME_EVNTS_PER_1000_BENES","mean"),ut_sd=("DME_EVNTS_PER_1000_BENES","std"),ut_med=("DME_EVNTS_PER_1000_BENES","median"),
 sd_m=("suppliers_per_10k","mean"),sd_sd=("suppliers_per_10k","std")).loc[order]; g.to_csv(D+"t_desc_region.csv"); print(g.round(2))
# state-level supplier-density: how many counties have real vs imputed
med=c["suppliers_per_10k"].median(); print("imputed share",(c.suppliers_per_10k==med).mean())
# cluster profile with n
# supplier descriptives
sc=["DME_Tot_Suplr_Benes","DME_Tot_Suplr_Clms","DME_Suplr_Mdcr_Pymt_Amt","DME_Suplr_Mdcr_Stdzd_Pymt_Amt"]
rows=[[k,len(s[k]),s[k].mean(),s[k].std(),s[k].median(),s[k].quantile(.25),s[k].quantile(.75),s[k].max(),stats.skew(s[k])] for k in sc]
d2=pd.DataFrame(rows,columns=["var","n","mean","sd","median","q1","q3","max","skew"]); d2.to_csv(D+"t_desc_supplier.csv",index=False); print(d2.round(1))
print("iqr/iso flags",s.iqr_flag.mean() if 'iqr_flag' in s else 'na')
# supplier by flag? flags not saved in clean_supplier (saved before). skip
# OLS assumptions
cm=pd.get_dummies(c,columns=["census_region"],drop_first=True)
rc=[x for x in cm.columns if x.startswith("census_region_")]
F=["BENE_AVG_AGE","BENE_DUAL_PCT","BENE_FEML_PCT","BENE_RACE_BLACK_PCT","BENE_RACE_HSPNC_PCT","suppliers_per_10k"]+rc
X=cm[F].astype(float); y=cm["DME_MDCR_STDZD_PYMT_PC"].astype(float)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
ols=sm.OLS(ytr,sm.add_constant(Xtr)).fit()
res=ols.resid
bp=het_breuschpagan(res,ols.model.exog)
sh=stats.shapiro(res); dw=durbin_watson(res)
infl=ols.get_influence(); cook=infl.cooks_distance[0]
olsr=sm.OLS(ytr,sm.add_constant(Xtr)).fit(cov_type="HC3")
out["ols"]={"n_train":len(Xtr),"n_test":len(Xte),"bp_lm":bp[0],"bp_p":bp[1],"shapiro_W":sh[0],"shapiro_p":sh[1],"skew_res":stats.skew(res),"kurt_res":stats.kurtosis(res),"dw":dw,
 "max_cook":cook.max(),"n_cook_gt_4n":int((cook>4/len(Xtr)).sum()),"hc3":{k:[float(olsr.params[k]),float(olsr.bse[k]),float(olsr.pvalues[k])] for k in olsr.params.index},
 "f":float(ols.fvalue),"f_p":float(ols.f_pvalue),"df":[ols.df_model,ols.df_resid]}
# standardized betas
zb=sm.OLS((ytr-ytr.mean())/ytr.std(),sm.add_constant((Xtr-Xtr.mean())/Xtr.std())).fit().params
out["std_beta"]=zb.to_dict()
# perm importance on test set
rf=RandomForestRegressor(n_estimators=400,max_depth=6,random_state=42).fit(Xtr,ytr)
pi=permutation_importance(rf,Xte,yte,n_repeats=30,random_state=42,scoring="r2")
pit=pd.DataFrame({"feature":F,"mean":pi.importances_mean,"sd":pi.importances_std}).sort_values("mean",ascending=False); pit.to_csv(D+"t_perm_imp.csv",index=False); print(pit.round(4))
out["rf_train_r2"]=r2_score(ytr,rf.predict(Xtr))
# ANOVA assumptions
for nm,col in [("util","DME_EVNTS_PER_1000_BENES"),("acc","suppliers_per_10k")]:
    gs=[c[c.census_region==r][col].values for r in order]
    lev=stats.levene(*gs,center="median"); kw=stats.kruskal(*gs)
    resid=np.concatenate([x-x.mean() for x in gs]); shp=stats.shapiro(resid[:5000])
    # Welch anova
    import statsmodels.stats.oneway as ow
    w=ow.anova_oneway(gs,use_var="unequal",welch_correction=True)
    out["anova_"+nm]={"levene_W":lev.statistic,"levene_p":lev.pvalue,"kw_H":kw.statistic,"kw_p":kw.pvalue,"shapiro_W":shp.statistic,"shapiro_p":shp.pvalue,"welch_F":float(w.statistic),"welch_p":float(w.pvalue),"skew_resid":stats.skew(resid)}
    if nm=="util":
        tk=stats.tukey_hsd(*gs); out["tukey_util"]=[[order[i],order[j],float(tk.statistic[i,j]),float(tk.pvalue[i,j])] for i in range(4) for j in range(i+1,4)]
# power check RQ1 (noncentral F)
from scipy.stats import f as F_, ncf
def power(n,k,f2,a=.05):
    d1=k; d2=n-k-1; lam=f2*n; crit=F_.ppf(1-a,d1,d2); return 1-ncf.cdf(crit,d1,d2,lam)
for k in (8,9,10):
    n=k+2
    while power(n,k,.15)<.80: n+=1
    out[f"power_k{k}"]=n; print(k,n,50+8*k)
# anova power
from statsmodels.stats.power import FTestAnovaPower
out["anova_n_per_group"]=FTestAnovaPower().solve_power(effect_size=.25,alpha=.05,power=.8,k_groups=4)/4; print(out["anova_n_per_group"])
print(json.dumps(out,indent=1,default=float))
json.dump(out,open(D+"results_extra.json","w"),indent=1,default=float)
