import numpy as np,pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest,RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
s=pd.read_csv('data/clean_supplier.csv')
sf=["DME_Tot_Suplr_Benes","DME_Tot_Suplr_Clms","DME_Suplr_Mdcr_Pymt_Amt","DME_Suplr_Mdcr_Stdzd_Pymt_Amt"]
s['iso']=(IsolationForest(contamination=.035,random_state=42).fit_predict(StandardScaler().fit_transform(s[sf]))==-1).astype(int)
ne={"CT","ME","MA","NH","NJ","NY","PA","RI","VT"};mw={"IL","IN","IA","KS","MI","MN","MO","NE","ND","OH","SD","WI"}
so={"AL","AR","DE","FL","GA","KY","LA","MD","MS","NC","OK","SC","TN","TX","VA","WV","DC"};we={"AK","AZ","CA","CO","HI","ID","MT","NV","NM","OR","UT","WA","WY"}
def reg(x): return "Northeast" if x in ne else "Midwest" if x in mw else "South" if x in so else "West" if x in we else None
s['reg']=s.Suplr_Prvdr_State_Abrvtn.map(reg); r=s.dropna(subset=['reg'])
for c in sf[:3]:
    u=stats.mannwhitneyu(r[r.iso==1][c],r[r.iso==0][c]); print(c,u.statistic,u.pvalue)
ct=pd.crosstab(r.reg,r.iso); print(ct); print((ct[1]/ct.sum(axis=1)*100).round(2))
chi=stats.chi2_contingency(ct); print(chi[0],chi[1],chi[2], np.sqrt(chi[0]/len(r)/ (min(ct.shape)-1)))
print(len(r),len(s))
c=pd.read_csv('data/county_with_cluster.csv'); cm=pd.get_dummies(c,columns=["census_region"],drop_first=True); rc=[x for x in cm if x.startswith('census_region_')]
F=["BENE_AVG_AGE","BENE_DUAL_PCT","BENE_FEML_PCT","BENE_RACE_BLACK_PCT","BENE_RACE_HSPNC_PCT","suppliers_per_10k"]+rc
X=cm[F].astype(float);y=cm.DME_MDCR_STDZD_PYMT_PC;Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
o=sm.OLS(ytr,sm.add_constant(Xtr)).fit(); print('ols train rmse',np.sqrt(mean_squared_error(ytr,o.predict(sm.add_constant(Xtr)))))
rf=RandomForestRegressor(n_estimators=400,max_depth=6,random_state=42).fit(Xtr,ytr); print('rf train rmse',np.sqrt(mean_squared_error(ytr,rf.predict(Xtr))))
# baseline mean predictor
print('mean baseline test rmse',np.sqrt(mean_squared_error(yte,np.full(len(yte),ytr.mean()))))
