import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans"})
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(9, 11))
ax.set_xlim(0, 10)
ax.set_ylim(0, 15.5)
ax.axis("off")


def wrap(text, width):
    return "\n".join(textwrap.wrap(text, width=width))


def box(x, y, w, h, title, detail, col, wrap_width=46, title_size=11, detail_size=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=col, ec="#333", lw=1))
    wrapped_detail = wrap(detail, wrap_width)
    n_lines = wrapped_detail.count("\n") + 1
    # Title near top, detail block centered in remaining space
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=title_size, fontweight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.12, wrapped_detail, ha="center", va="center",
            fontsize=detail_size, linespacing=1.4)


def arr(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, color="#333", lw=1.2))


# Stage boxes (x, y, w, h)
box(0.3, 13.6, 9.4, 1.5, "1. DATA SOURCE",
    "CMS data.cms.gov (CSV, full files): Geographic Variation | DME by Supplier | DME by Referring Provider | ACS 5-Year",
    "#DCE9F5", wrap_width=72)

box(0.3, 11.7, 9.4, 1.5, "2. PREPROCESSING",
    "Suppression markers to NaN, median imputation, IQR capping, de-duplication, region mapping, ACS sentinel cleanup",
    "#E8F0DC", wrap_width=72)

box(0.3, 9.8, 9.4, 1.5, "3. EXPLORATORY DATA ANALYSIS",
    "Descriptive statistics, distributions, subgroup and correlation analysis",
    "#E8F0DC", wrap_width=60)

box(0.3, 7.9, 9.4, 1.5, "4. FEATURE ENGINEERING",
    "Region dummies, state supplier density, income/poverty merge, payment shares, standardization",
    "#FBEFD5", wrap_width=65)

box(0.1, 5.5, 2.35, 2.0, "5. MODEL (RQ1)", "OLS + Random Forest", "#F5DCDC", wrap_width=16, title_size=10, detail_size=9.5)
box(2.6, 5.5, 2.35, 2.0, "5. MODEL (RQ2)", "ANOVA + K-Means", "#F5DCDC", wrap_width=16, title_size=10, detail_size=9.5)
box(5.1, 5.5, 2.35, 2.0, "5. MODEL (RQ3)", "IQR + Isolation Forest", "#F5DCDC", wrap_width=16, title_size=10, detail_size=9.5)
box(7.6, 5.5, 2.35, 2.0, "5. MODEL (RQ4)", "HHI, state + category", "#F5DCDC", wrap_width=16, title_size=10, detail_size=9.5)

box(0.3, 3.3, 9.4, 1.6, "6. EVALUATION",
    "Hold-out R2/RMSE/MAE vs. baseline, F-tests, silhouette, Jaccard, Mann-Whitney, chi-square, HHI bands",
    "#DCE9F5", wrap_width=70)

box(0.3, 1.1, 9.4, 1.6, "7. DEPLOYMENT",
    "SQLite database + scheduled batch scoring + program-integrity dashboard",
    "#E0D5EC", wrap_width=60)

# Arrows
arr(5, 13.6, 5, 13.2)
arr(5, 11.7, 5, 11.3)
arr(5, 9.8, 5, 9.4)
for x in (1.275, 3.775, 6.275, 8.775):
    arr(x, 7.9, x, 7.5)
    arr(x, 5.5, x, 4.9)
arr(5, 3.3, 5, 2.7)

plt.savefig("final/workflow_diagram.png", dpi=220, bbox_inches="tight")
print("saved")
