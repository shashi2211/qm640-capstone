import pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import shutil
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11})
sns.set_style("whitegrid")
c=pd.read_csv("data/clean_county_geographic.csv"); order=["Northeast","Midwest","South","West"]
fig,axs=plt.subplots(1,2,figsize=(6.5,3.8))
sns.boxplot(data=c,x="census_region",y="DME_MDCR_STDZD_PYMT_PC",order=order,color="#9ECAE1",ax=axs[0]); axs[0].set_title("(a) Spending per beneficiary"); axs[0].set_ylabel("Dollars"); axs[0].set_xlabel("")
sns.boxplot(data=c,x="census_region",y="DME_EVNTS_PER_1000_BENES",order=order,color="#9ECAE1",ax=axs[1]); axs[1].set_title("(b) Utilization"); axs[1].set_ylabel("Events per 1,000 beneficiaries"); axs[1].set_xlabel("")
for a in axs: a.tick_params(axis="x",labelrotation=30)
plt.tight_layout(); plt.savefig("final/R2.png",dpi=300,bbox_inches="tight"); plt.close()
# workflow
fig,ax=plt.subplots(figsize=(6.5,5.6)); ax.set_xlim(0,10); ax.set_ylim(0,12); ax.axis("off")
def box(x,y,w,h,t,col):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.05",fc=col,ec="#333",lw=1)); ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=9.5)
def arr(x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=12,color="#333"))
box(0.3,10.2,9.4,1.4,"Data sources (CMS data.cms.gov, CSV)\nGeographic Variation  |  DME by Supplier  |  DME by Referring Provider","#DCE9F5")
box(0.3,8.2,9.4,1.3,"Preprocessing\nsuppression markers \u2192 NaN, median imputation, IQR capping, de-duplication, region mapping","#E8F0DC")
box(0.3,6.4,9.4,1.2,"Exploratory data analysis\ndescriptive statistics, distributions, subgroup and correlation analysis","#E8F0DC")
box(0.3,4.7,9.4,1.2,"Feature engineering\nregion dummies, state supplier density, payment shares, z-scores","#FBEFD5")
box(0.1,2.3,2.3,1.9,"RQ1\nOLS +\nRandom Forest","#F5DCDC"); box(2.6,2.3,2.3,1.9,"RQ2\nANOVA +\nK-Means","#F5DCDC"); box(5.1,2.3,2.3,1.9,"RQ3\nIQR +\nIsolation Forest","#F5DCDC"); box(7.6,2.3,2.3,1.9,"RQ4\nHHI by state","#F5DCDC")
box(0.3,0.3,9.4,1.4,"Evaluation and outputs\nhold-out R\u00b2/RMSE/MAE, F-tests, silhouette, Jaccard, HHI bands, tables and figures","#DCE9F5")
for y1,y2 in [(10.2,9.5),(8.2,7.6),(6.4,5.9)]: arr(5,y1,5,y2)
for x in (1.25,3.75,6.25,8.75): arr(x,4.7,x,4.25); arr(x,2.3,x,1.75)
plt.savefig("final/W.png",dpi=300,bbox_inches="tight"); plt.close()
for a,b in [("f1","D1"),("f4","H"),("f11","DG"),("f6","I"),("f8","CL"),("f9","Q3"),("f10","HH")]: shutil.copy(f"figs/{a}.png",f"final/{b}.png")
