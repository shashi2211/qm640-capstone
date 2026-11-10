import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.titlesize":12,"axes.labelsize":11,"xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":10})
sns.set_style("whitegrid")
D="data/"; O="figs/"
c=pd.read_csv(D+"clean_county_geographic.csv"); s=pd.read_csv(D+"clean_supplier.csv"); h=pd.read_csv(D+"rq4_hhi_preliminary.csv")
order=["Northeast","Midwest","South","West"]; BL="#2E6E9E"
def save(n): plt.tight_layout(); plt.savefig(O+n,dpi=300,bbox_inches="tight"); plt.close()
SP="DME_MDCR_STDZD_PYMT_PC"
fig,ax=plt.subplots(figsize=(6.5,4)); sns.histplot(c[SP],kde=True,color=BL,ax=ax,bins=25)
ax.set_xlabel("Standardized DME Medicare payment per beneficiary ($)"); ax.set_ylabel("Number of counties"); save("f1.png")
fig,ax=plt.subplots(figsize=(6.5,4)); sns.boxplot(data=c,x="census_region",y=SP,order=order,palette="Blues",ax=ax)
ax.set_xlabel("Census region"); ax.set_ylabel("DME spending per beneficiary ($)"); save("f2.png")
fig,ax=plt.subplots(figsize=(6.5,4)); sns.boxplot(data=c,x="census_region",y="DME_EVNTS_PER_1000_BENES",order=order,palette="Blues",ax=ax)
ax.set_xlabel("Census region"); ax.set_ylabel("DME events per 1,000 beneficiaries"); save("f3.png")
cc=[SP,"BENE_AVG_AGE","BENE_DUAL_PCT","BENE_FEML_PCT","BENE_RACE_BLACK_PCT","BENE_RACE_HSPNC_PCT","suppliers_per_10k"]
lab=["Spending","Avg. age","Dual-eligible","Female","Black","Hispanic","Suppliers/10k"]
cm=c[cc].corr(); cm.index=lab; cm.columns=lab
fig,ax=plt.subplots(figsize=(6.5,5.2)); sns.heatmap(cm,annot=True,fmt=".2f",cmap="coolwarm",center=0,vmin=-.6,vmax=1,ax=ax,annot_kws={"size":10}); plt.xticks(rotation=40,ha="right"); save("f4.png")
fig,ax=plt.subplots(figsize=(6.5,4)); sns.regplot(data=c,x="BENE_DUAL_PCT",y=SP,scatter_kws={"alpha":.4,"s":18},line_kws={"color":"red"},ax=ax)
ax.set_xlabel("Dual-eligible share of beneficiaries (proportion)"); ax.set_ylabel("DME spending per beneficiary ($)"); save("f5.png")
# RF
cmod=pd.get_dummies(c,columns=["census_region"],drop_first=True); rc=[x for x in cmod.columns if x.startswith("census_region_")]
F=["BENE_AVG_AGE","BENE_DUAL_PCT","BENE_FEML_PCT","BENE_RACE_BLACK_PCT","BENE_RACE_HSPNC_PCT","suppliers_per_10k"]+rc
nm={"BENE_AVG_AGE":"Average age","BENE_DUAL_PCT":"Dual-eligible %","BENE_FEML_PCT":"Female %","BENE_RACE_BLACK_PCT":"Black %","BENE_RACE_HSPNC_PCT":"Hispanic %","suppliers_per_10k":"Suppliers per 10k","census_region_Northeast":"Region: Northeast","census_region_South":"Region: South","census_region_West":"Region: West"}
X=cmod[F].astype(float); y=cmod[SP].astype(float); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
rf=RandomForestRegressor(n_estimators=400,max_depth=6,random_state=42).fit(Xtr,ytr)
imp=pd.Series(rf.feature_importances_,index=F).sort_values(); pi=permutation_importance(rf,Xte,yte,n_repeats=30,random_state=42,scoring="r2")
pim=pd.Series(pi.importances_mean,index=F); pis=pd.Series(pi.importances_std,index=F)
o=pim.sort_values().index
fig,axs=plt.subplots(1,2,figsize=(6.5,3.8),sharey=True)
axs[0].barh([nm[i] for i in o],imp[o],color=BL); axs[0].set_xlabel("Impurity-based importance"); axs[0].set_title("(a) Impurity-based")
axs[1].barh([nm[i] for i in o],pim[o],xerr=pis[o],color="#D98324"); axs[1].set_xlabel("Drop in test R\u00b2 when permuted"); axs[1].set_title("(b) Permutation (test set)"); save("f6.png")
# kmeans
cf=[SP,"DME_EVNTS_PER_1000_BENES","BENE_DUAL_PCT","BENE_AVG_AGE","suppliers_per_10k"]
Xc=StandardScaler().fit_transform(c[cf]); K=list(range(2,7)); inert=[];sil=[]
for k in K:
    km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(Xc); inert.append(km.inertia_); sil.append(silhouette_score(Xc,km.labels_))
print(dict(zip(K,np.round(sil,3))))
fig,axs=plt.subplots(1,2,figsize=(6.5,3.6))
axs[0].plot(K,inert,marker="o",color=BL); axs[0].set_xlabel("Number of clusters (k)"); axs[0].set_ylabel("Inertia (WCSS)"); axs[0].set_title("(a) Elbow")
axs[1].plot(K,sil,marker="o",color="#D98324"); axs[1].set_xlabel("Number of clusters (k)"); axs[1].set_ylabel("Mean silhouette score"); axs[1].set_title("(b) Silhouette"); save("f7.png")
km=KMeans(n_clusters=3,n_init=10,random_state=42).fit(Xc); c["cluster"]=km.labels_
fig,ax=plt.subplots(figsize=(6.5,4)); sns.scatterplot(data=c,x="suppliers_per_10k",y=SP,hue="cluster",palette="Set2",s=26,alpha=.75,ax=ax)
ax.set_xlabel("Suppliers per 10,000 beneficiaries (state-level proxy)"); ax.set_ylabel("DME spending per beneficiary ($)"); save("f8.png")
prof=c.groupby("cluster")[cf].mean(); prof["n"]=c.groupby("cluster").size(); prof.to_csv(D+"t_cluster_profile.csv"); print(prof.round(2))
print(c.groupby(["cluster","census_region"]).size().unstack(fill_value=0))
# RQ3
sf=["DME_Tot_Suplr_Benes","DME_Tot_Suplr_Clms","DME_Suplr_Mdcr_Pymt_Amt","DME_Suplr_Mdcr_Stdzd_Pymt_Amt"]
iso=IsolationForest(contamination=.035,random_state=42); s["iso"]=(iso.fit_predict(StandardScaler().fit_transform(s[sf]))==-1).astype(int)
q1,q3=s.DME_Suplr_Mdcr_Pymt_Amt.quantile([.25,.75]); s["iqr"]=(s.DME_Suplr_Mdcr_Pymt_Amt>q3+1.5*(q3-q1)).astype(int)
print(s.iso.mean(),s.iqr.mean(),(s.iso&s.iqr).sum()/((s.iso|s.iqr).sum()))
fig,ax=plt.subplots(figsize=(6.5,4)); s["pm"]=s.DME_Suplr_Mdcr_Pymt_Amt/1e6
sns.scatterplot(data=s[s.pm>0],x="DME_Tot_Suplr_Benes",y="pm",hue="iso",palette={0:BL,1:"#D62728"},s=14,alpha=.6,ax=ax)
ax.set_yscale("log"); ax.set_xlabel("DME beneficiaries served"); ax.set_ylabel("DME Medicare payment ($ millions, log scale)")
hd,lb=ax.get_legend_handles_labels(); ax.legend(hd,["Not flagged","Flagged"],title="Isolation Forest"); save("f9.png")
print(s.groupby("iso")[["DME_Tot_Suplr_Benes","DME_Tot_Suplr_Clms","DME_Suplr_Mdcr_Pymt_Amt"]].median())
fig,ax=plt.subplots(figsize=(6.5,4)); sns.histplot(h.HHI,bins=20,color=BL,ax=ax)
ax.axvline(1500,color="orange",ls="--",label="Moderate (1,500)"); ax.axvline(2500,color="red",ls="--",label="High (2,500)")
ax.set_xlabel("Herfindahl\u2013Hirschman Index"); ax.set_ylabel("Number of states"); ax.legend(); save("f10.png")
# OLS diagnostics
ols=sm.OLS(ytr,sm.add_constant(Xtr)).fit()
fig,axs=plt.subplots(1,2,figsize=(6.5,3.6))
axs[0].scatter(ols.fittedvalues,ols.resid,s=10,alpha=.5,color=BL); axs[0].axhline(0,color="red"); axs[0].set_xlabel("Fitted value ($)"); axs[0].set_ylabel("Residual ($)"); axs[0].set_title("(a) Residuals vs. fitted")
sm.qqplot(ols.resid,line="s",ax=axs[1],markersize=3,markerfacecolor=BL,markeredgecolor=BL); axs[1].set_title("(b) Normal Q\u2013Q plot"); save("f11.png")
c.to_csv(D+"county_with_cluster.csv",index=False)
# top states HHI count of states with HHI >2500
print((h.HHI>2500).sum(),(h.HHI<1500).sum(),len(h), h.HHI.describe())
