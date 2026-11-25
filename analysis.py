"""
Task 1 — Classification Model Pipeline
Bank Marketing Dataset
Generates: model_dashboard.png  (performance dashboard)
Usage: python3 analysis.py
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────────────────────
DATA_FILE = "bank.csv"
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"'{DATA_FILE}' not found.\n"
        "Download from: https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset\n"
        "and place it in this directory as 'bank.csv'."
    )

df = pd.read_csv(DATA_FILE, sep=",")
if df.shape[1] == 1:
    df = pd.read_csv(DATA_FILE, sep=";")

print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Normalise target column
target_col = None
for c in df.columns:
    if c.strip().lower() in ("deposit", "y"):
        target_col = c; break
if target_col is None:
    raise KeyError("Cannot find target column 'deposit' or 'y'")
if target_col != "deposit":
    df.rename(columns={target_col: "deposit"}, inplace=True)
df["deposit"] = df["deposit"].str.strip().str.lower()

# ── Feature Engineering ────────────────────────────────────────────────────
print("Engineering features...")

cat_cols = ["job","marital","education","contact","month","poutcome","default","housing","loan"]
le = LabelEncoder()
for col in cat_cols:
    if col in df.columns:
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))

# Engineered features
df["has_previous"]     = (df["previous"] > 0).astype(int)
df["contacted_before"] = (df["pdays"] != -1).astype(int)
df["balance_log"]      = np.log1p(df["balance"].clip(lower=0))
df["age_group"]        = pd.cut(
    df["age"], bins=[0,30,40,50,60,200],
    labels=[0,1,2,3,4], right=False
).astype(int)

# Target
df["y"] = (df["deposit"] == "yes").astype(int)

# Feature set
base_num  = ["age","balance","duration","campaign","pdays","previous","day"]
eng_feats = ["has_previous","contacted_before","balance_log","age_group"]
enc_cols  = [c for c in df.columns if c.endswith("_enc")]
FEATURES  = base_num + eng_feats + enc_cols

# Ensure all feature cols exist
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"Feature set: {len(FEATURES)} features — {FEATURES}")

X = df[FEATURES]
y = df["y"]

# ── Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Models ────────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
    "Decision Tree":       (DecisionTreeClassifier(max_depth=6, random_state=42), False),
    "Random Forest":       (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), False),
    "Gradient Boosting":   (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
}

results   = {}
roc_data  = {}
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, (model, scaled) in models.items():
    print(f"Training {name}...")
    Xtr = X_train_sc if scaled else X_train
    Xte = X_test_sc  if scaled else X_test

    model.fit(Xtr, y_train)
    y_pred  = model.predict(Xte)
    y_prob  = model.predict_proba(Xte)[:, 1]
    cv_auc  = cross_val_score(model, Xtr, y_train, cv=cv, scoring="roc_auc")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val      = roc_auc_score(y_test, y_prob)

    results[name] = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1":        f1_score(y_test, y_pred),
        "ROC-AUC":   auc_val,
        "CV AUC":    cv_auc.mean(),
        "CV Std":    cv_auc.std(),
    }
    roc_data[name] = (fpr, tpr, auc_val)
    print(f"  Accuracy={results[name]['Accuracy']:.4f}  ROC-AUC={auc_val:.4f}  CV={cv_auc.mean():.4f}±{cv_auc.std():.4f}")

res_df = pd.DataFrame(results).T
best   = res_df["ROC-AUC"].idxmax()
print(f"\nBest model: {best}  (ROC-AUC={res_df.loc[best,'ROC-AUC']:.4f})")

# Best model confusion matrix
best_model, best_scaled = models[best]
Xte_best = X_test_sc if best_scaled else X_test
y_pred_best = best_model.predict(Xte_best)
cm = confusion_matrix(y_test, y_pred_best)

# Random Forest feature importance
rf_model = models["Random Forest"][0]
fi = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=True).tail(12)

# ── Dashboard ─────────────────────────────────────────────────────────────
print("Plotting model performance dashboard...")

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
})

fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor("#0d1b2a")
fig.suptitle("Bank Marketing — Classification Model Performance Report",
             fontsize=16, fontweight="bold", color="#ffffff", y=0.99)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

model_names   = list(results.keys())
short_names   = ["LR", "DT", "RF", "GB"]
metric_names  = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
COLORS        = ["#3b82f6","#f59e0b","#10b981","#a855f7"]

# ── Row 0, Left: Grouped Bar — all metrics ────────────────────────────────
ax0 = fig.add_subplot(gs[0, :2])
x   = np.arange(len(metric_names))
w   = 0.18
for i, (name, color) in enumerate(zip(model_names, COLORS)):
    vals = [results[name][m] for m in metric_names]
    ax0.bar(x + i*w - 1.5*w, vals, width=w, color=color, label=name, alpha=0.9, edgecolor="#0d1b2a")
ax0.set_xticks(x)
ax0.set_xticklabels(metric_names, fontsize=8)
ax0.set_ylim(0.6, 1.0)
ax0.set_ylabel("Score")
ax0.set_title("Model Performance Comparison")
ax0.legend(fontsize=7, loc="lower right")
ax0.grid(axis="y", alpha=0.4)

# ── Row 0, Right: Horizontal CV AUC bars ─────────────────────────────────
ax1 = fig.add_subplot(gs[0, 2])
y_pos  = np.arange(len(model_names))
cv_auc = [results[n]["CV AUC"] for n in model_names]
cv_std = [results[n]["CV Std"] for n in model_names]
ax1.barh(y_pos, cv_auc, color=COLORS, alpha=0.9, edgecolor="#0d1b2a", xerr=cv_std,
         error_kw=dict(ecolor="#ffffff", capsize=4, linewidth=1.2))
for i, (v, s) in enumerate(zip(cv_auc, cv_std)):
    ax1.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8, color="#ffffff", fontweight="bold")
ax1.set_yticks(y_pos)
ax1.set_yticklabels([n.replace(" ", "\n") for n in model_names], fontsize=8)
ax1.set_xlim(0.80, 0.97)
ax1.set_xlabel("ROC-AUC")
ax1.set_title("5-Fold CV ROC-AUC\n(mean ± std)")
ax1.grid(axis="x", alpha=0.4)

# ── Row 1, Left: ROC Curves ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot([0,1],[0,1], "w--", linewidth=0.8, alpha=0.5)
for name, color in zip(model_names, COLORS):
    fpr, tpr, auc_v = roc_data[name]
    ax2.plot(fpr, tpr, color=color, linewidth=1.8,
             label=f"{name.split()[0]} ({name.split()[-1][:2] if len(name.split())>1 else ''}) ({auc_v:.3f})")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curves")
ax2.legend(fontsize=6, loc="lower right")
ax2.set_xlim(0,1); ax2.set_ylim(0,1.02)
ax2.grid(alpha=0.4)

# ── Row 1, Middle: Confusion Matrix ───────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
labels_cm = ["No","Yes"]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3,
            xticklabels=labels_cm, yticklabels=labels_cm,
            annot_kws={"size":14,"weight":"bold"},
            linewidths=2, linecolor="#0d1b2a",
            cbar_kws={"shrink":0.8})
ax3.set_title(f"Confusion Matrix\n({best})")
ax3.set_xlabel("Predicted"); ax3.set_ylabel("Actual")

# ── Row 1, Right: Feature Importance ──────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
colors_fi = ["#10b981" if v > fi.median() else "#3b82f6" for v in fi.values]
ax4.barh(fi.index, fi.values, color=colors_fi, edgecolor="#0d1b2a")
ax4.set_title("Feature Importance\n(Random Forest)")
ax4.set_xlabel("Importance")
ax4.grid(axis="x", alpha=0.4)

# ── Row 2: Metrics Table ───────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, :])
ax5.axis("off")
headers = ["Model","Accuracy","Precision","Recall","F1 Score","ROC-AUC","CV AUC (±std)"]
rows = []
for name in model_names:
    r = results[name]
    rows.append([
        name,
        f"{r['Accuracy']:.4f}",
        f"{r['Precision']:.4f}",
        f"{r['Recall']:.4f}",
        f"{r['F1']:.4f}",
        f"{r['ROC-AUC']:.4f}",
        f"{r['CV AUC']:.4f} ± {r['CV Std']:.4f}",
    ])

tbl = ax5.table(
    cellText=rows, colLabels=headers,
    cellLoc="center", loc="center", bbox=[0,0.05,1,0.88]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for (row, col), cell in tbl.get_celld().items():
    cell.set_facecolor("#111827")
    cell.set_edgecolor("#1e3045")
    cell.set_text_props(color="#c9d1d9")
    if row == 0:
        cell.set_facecolor("#1a2d4a")
        cell.set_text_props(color="#60a5fa", fontweight="bold")
    if row > 0 and model_names[row-1] == best:
        cell.set_facecolor("#0d3025")
        cell.set_text_props(color="#34d399", fontweight="bold")

ax5.set_title(f"Model Performance Summary Table  (★ Best: {best} highlighted in green)",
              fontsize=10, pad=12, color="#ffffff")

# ── Save ───────────────────────────────────────────────────────────────────
out = "model_dashboard.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1b2a")
print(f"✅ Saved: {out}")
plt.close()
print("\nAll done! Results summary:")
print(res_df[["Accuracy","Precision","Recall","F1","ROC-AUC"]].round(4).to_string())
