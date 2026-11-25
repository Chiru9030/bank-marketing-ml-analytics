"""
Task 2 — Data Visualization Dashboard
Bank Marketing Dataset
Generates: dashboard.png  (10-chart data insights dashboard)
Usage: python3 visualize.py
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────────────────────
DATA_FILE = "bank.csv"
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"'{DATA_FILE}' not found.\n"
        "Download from: https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset\n"
        "and place it in this directory."
    )

df = pd.read_csv(DATA_FILE, sep=",")
# Detect semicolon-separated file (alternative format)
if df.shape[1] == 1:
    df = pd.read_csv(DATA_FILE, sep=";")

print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Normalise target column name
target_col = None
for c in df.columns:
    if c.strip().lower() in ("deposit", "y"):
        target_col = c
        break
if target_col is None:
    raise KeyError("Cannot find target column 'deposit' or 'y' in dataset.")
if target_col != "deposit":
    df.rename(columns={target_col: "deposit"}, inplace=True)

df["deposit"] = df["deposit"].str.strip().str.lower()
# Binary numeric target for correlations
df["y"] = (df["deposit"] == "yes").astype(int)

# ── Palette ────────────────────────────────────────────────────────────────
C_YES  = "#2ecc71"   # subscribed
C_NO   = "#e74c3c"   # not subscribed
C_MAIN = "#3498db"   # generic blue
C_ACC  = "#f39c12"   # accent orange

plt.rcParams.update({
    "figure.facecolor": "#0d1b2a",
    "axes.facecolor":   "#0d1b2a",
    "axes.edgecolor":   "#2c4a6e",
    "axes.labelcolor":  "#c9d1d9",
    "text.color":       "#c9d1d9",
    "xtick.color":      "#8b9ab5",
    "ytick.color":      "#8b9ab5",
    "grid.color":       "#1e3045",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   10,
    "axes.titleweight": "bold",
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
})

# ── Figure layout (4 rows × 3 cols) ────────────────────────────────────────
fig = plt.figure(figsize=(20, 26))
fig.patch.set_facecolor("#0d1b2a")
fig.suptitle(
    "Bank Marketing Dataset — Data Insights Dashboard",
    fontsize=17, fontweight="bold", color="#ffffff", y=0.995
)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.38)

# ── Chart 1: Target Distribution ──────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
counts = df["deposit"].value_counts()
bars = ax1.bar(counts.index, counts.values, color=[C_NO, C_YES], width=0.5, edgecolor="#0d1b2a")
for bar, val in zip(bars, counts.values):
    pct = val / len(df) * 100
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
             f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8, fontweight="bold",
             color="#ffffff")
ax1.set_title("Target Class Distribution")
ax1.set_ylabel("Count")
ax1.set_ylim(0, counts.max() * 1.22)
ax1.grid(axis="y", alpha=0.4)

# ── Chart 2: Deposit Rate by Job ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1:])
job_rates = (df.groupby("job")["y"].mean() * 100).sort_values(ascending=True)
colors_j = [C_YES if v >= df["y"].mean()*100 else C_MAIN for v in job_rates.values]
hbars = ax2.barh(job_rates.index, job_rates.values, color=colors_j, edgecolor="#0d1b2a")
for bar, val in zip(hbars, job_rates.values):
    ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}%", va="center", fontsize=8, color="#ffffff")
avg_line = df["y"].mean() * 100
ax2.axvline(avg_line, color="#e74c3c", linestyle="--", linewidth=1.2, label=f"Overall avg: {avg_line:.1f}%")
ax2.legend(fontsize=7)
ax2.set_title("Deposit Rate by Job Type")
ax2.set_xlabel("Deposit Rate (%)")
ax2.set_xlim(0, job_rates.max() * 1.18)

# ── Chart 3: Age Distribution by Outcome ──────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
df[df["deposit"]=="yes"]["age"].plot.hist(ax=ax3, bins=30, alpha=0.7, color=C_YES, label="Subscribed")
df[df["deposit"]=="no"]["age"].plot.hist(ax=ax3, bins=30, alpha=0.7, color=C_NO, label="Not Subscribed")
ax3.set_title("Age Distribution by Outcome")
ax3.set_xlabel("Age")
ax3.set_ylabel("Count")
ax3.legend(fontsize=7)
ax3.grid(axis="y", alpha=0.4)

# ── Chart 4: Account Balance by Outcome ───────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
balance_data = [
    df[df["deposit"]=="no"]["balance"].clip(-500, 10000),
    df[df["deposit"]=="yes"]["balance"].clip(-500, 10000),
]
bp = ax4.boxplot(balance_data, labels=["Not Subscribed", "Subscribed"],
                 patch_artist=True, notch=False, widths=0.4,
                 medianprops=dict(color="#f1c40f", linewidth=2))
for patch, c in zip(bp["boxes"], [C_NO, C_YES]):
    patch.set_facecolor(c); patch.set_alpha(0.7)
for part in ["whiskers","caps","fliers"]:
    for el in bp[part]:
        el.set_color("#8b9ab5")
ax4.set_title("Account Balance by Outcome")
ax4.set_ylabel("Balance (€)")
ax4.grid(axis="y", alpha=0.4)

# ── Chart 5: Call Duration by Outcome ─────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
dur_data = [
    df[df["deposit"]=="no"]["duration"].clip(0, 3000),
    df[df["deposit"]=="yes"]["duration"].clip(0, 3000),
]
bp2 = ax5.boxplot(dur_data, labels=["Not Subscribed", "Subscribed"],
                  patch_artist=True, notch=False, widths=0.4,
                  medianprops=dict(color="#f1c40f", linewidth=2))
for patch, c in zip(bp2["boxes"], [C_NO, C_ACC]):
    patch.set_facecolor(c); patch.set_alpha(0.7)
for part in ["whiskers","caps","fliers"]:
    for el in bp2[part]:
        el.set_color("#8b9ab5")
ax5.set_title("Call Duration by Outcome")
ax5.set_ylabel("Duration (seconds)")
ax5.grid(axis="y", alpha=0.4)

# ── Chart 6: Deposit Rate & Contact Volume by Month ───────────────────────
ax6 = fig.add_subplot(gs[2, 0:2])
month_order = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
df["month_cat"] = pd.Categorical(df["month"].str.lower(), categories=month_order, ordered=True)
month_stats = df.groupby("month_cat", observed=True).agg(
    deposit_rate=("y", lambda x: x.mean()*100),
    n_contacts=("y", "count")
).reindex(month_order)
ax6.bar(month_stats.index, month_stats["deposit_rate"], color=C_MAIN, alpha=0.85, label="Deposit Rate %")
ax6b = ax6.twinx()
ax6b.plot(month_stats.index, month_stats["n_contacts"], color="#e74c3c", marker="o",
          linewidth=1.8, markersize=5, label="# Contacts")
ax6b.set_ylabel("# Contacts", color="#e74c3c")
ax6b.tick_params(axis="y", labelcolor="#e74c3c")
ax6.set_title("Deposit Rate & Contact Volume by Month")
ax6.set_ylabel("Deposit Rate (%)")
ax6.set_ylim(0, month_stats["deposit_rate"].max() * 1.25)

lines1, labels1 = ax6.get_legend_handles_labels()
lines2, labels2 = ax6b.get_legend_handles_labels()
ax6.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

# ── Chart 7: Deposit Rate by Previous Campaign Outcome ────────────────────
ax7 = fig.add_subplot(gs[2, 2])
pout_rates = (df.groupby("poutcome")["y"].mean() * 100).sort_values()
palette_pout = [C_NO, C_NO, C_MAIN, C_YES][:len(pout_rates)]
bars7 = ax7.bar(pout_rates.index, pout_rates.values,
                color=palette_pout, edgecolor="#0d1b2a")
for bar, val in zip(bars7, pout_rates.values):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold",
             color="#ffffff")
ax7.set_title("Deposit Rate by\nPrevious Campaign Outcome")
ax7.set_ylabel("Deposit Rate (%)")
ax7.set_ylim(0, pout_rates.max() * 1.22)
ax7.grid(axis="y", alpha=0.4)

# ── Chart 8: Deposit Rate by Contact Method ───────────────────────────────
ax8 = fig.add_subplot(gs[3, 0])
contact_rates = (df.groupby("contact")["y"].mean() * 100).sort_values()
palette_c = [C_NO if v < 30 else (C_MAIN if v < 52 else C_YES) for v in contact_rates.values]
bars8 = ax8.bar(contact_rates.index, contact_rates.values, color=palette_c, edgecolor="#0d1b2a", width=0.5)
for bar, val in zip(bars8, contact_rates.values):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#ffffff")
ax8.set_title("Deposit Rate by\nContact Method")
ax8.set_ylabel("Deposit Rate (%)")
ax8.set_ylim(0, contact_rates.max() * 1.25)
ax8.grid(axis="y", alpha=0.4)

# ── Chart 9: Deposit Rate by Education ───────────────────────────────────
ax9 = fig.add_subplot(gs[3, 1])
edu_rates = (df.groupby("education")["y"].mean() * 100).sort_values()
bars9 = ax9.barh(edu_rates.index, edu_rates.values, color=C_MAIN, edgecolor="#0d1b2a")
for bar, val in zip(bars9, edu_rates.values):
    ax9.text(val + 0.3, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}%", va="center", fontsize=8, color="#ffffff")
ax9.set_title("Deposit Rate by Education")
ax9.set_xlabel("Deposit Rate (%)")
ax9.set_xlim(0, edu_rates.max() * 1.2)

# ── Chart 10: Feature Correlation Heatmap ─────────────────────────────────
ax10 = fig.add_subplot(gs[3, 2])
num_cols = ["age", "balance", "duration", "campaign", "previous", "y"]
corr_df = df[num_cols].corr()
mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
sns.heatmap(
    corr_df, ax=ax10, mask=mask, annot=True, fmt=".2f", annot_kws={"size": 7},
    cmap="RdYlGn", center=0, vmin=-1, vmax=1,
    linewidths=0.5, linecolor="#0d1b2a", cbar_kws={"shrink": 0.7}
)
ax10.set_title("Feature Correlation\nHeatmap")
ax10.tick_params(axis="x", rotation=45)

# ── Save ───────────────────────────────────────────────────────────────────
out = "dashboard.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1b2a")
print(f"✅ Saved: {out}")
plt.close()
