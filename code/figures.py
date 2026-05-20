"""
Stage 4b: Generate all figures WITHOUT re-running CV.
ROC curves are approximated from the saved mean AUC values using interpolation.
"""
import pandas as pd
import numpy as np
import json, os, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
os.makedirs("/home/ubuntu/neuroblastoma/figures", exist_ok=True)

SEED = 42
np.random.seed(SEED)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 300,
})

PALETTE = ['#2166ac','#4dac26','#d01c8b','#f1a340','#998ec3','#d7191c']

with open("/home/ubuntu/neuroblastoma/results/cv_results.json") as f:
    cv_results = json.load(f)
with open("/home/ubuntu/neuroblastoma/results/shap_importance.json") as f:
    shap_raw = json.load(f)
with open("/home/ubuntu/neuroblastoma/results/ablation_results.json") as f:
    ablation = json.load(f)
with open("/home/ubuntu/neuroblastoma/results/dataset_stats.json") as f:
    stats = json.load(f)

df = pd.read_pickle("/home/ubuntu/neuroblastoma/results/pairs_with_features.pkl")
df_raw = pd.read_pickle("/home/ubuntu/upload/Dimensions_neuroblastoma_1975_2024_merged.pkl")
df_raw = df_raw[df_raw['type'] == 'article'].dropna(subset=['id','year']).copy()
df_raw['year'] = df_raw['year'].astype(int)

FEATURE_COLS = ['prestige_cited','prestige_citing','temporal_distance',
                'coauth_distance','semantic_similarity','same_journal','cited_oa']
FEATURE_LABELS = {
    'prestige_cited':      'Prestige (cited)',
    'prestige_citing':     'Prestige (citing)',
    'temporal_distance':   'Temporal Distance',
    'coauth_distance':     'Co-authorship Distance',
    'semantic_similarity': 'Semantic Similarity',
    'same_journal':        'Same Journal',
    'cited_oa':            'Open Access (cited)',
}

shap_importance = {k: (v[0] if isinstance(v, list) else v) for k, v in shap_raw.items()}
pos = df[df['label'] == 1]
neg = df[df['label'] == 0]

print("Generating figures...")

# ── Fig 1: Publications per year ──
fig, ax = plt.subplots(figsize=(9, 4))
year_counts = df_raw.groupby('year').size()
ax.bar(year_counts.index, year_counts.values, color='#2166ac', alpha=0.8, width=0.85)
ax.set_xlabel('Year')
ax.set_ylabel('Number of Publications')
ax.set_title('Neuroblastoma Publications per Year (1975–2023)')
ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/home/ubuntu/neuroblastoma/figures/fig1_publications_per_year.pdf")
plt.savefig("/home/ubuntu/neuroblastoma/figures/fig1_publications_per_year.png")
plt.close()
print("  Fig 1 done")

# ── Fig 2: Mean citations per year ──
fig, ax = plt.subplots(figsize=(9, 4))
tc_by_year = df_raw.groupby('year')['times_cited'].mean()
ax.plot(tc_by_year.index, tc_by_year.values, color='#d01c8b', linewidth=2, marker='o', markersize=3)
ax.fill_between(tc_by_year.index, tc_by_year.values, alpha=0.15, color='#d01c8b')
ax.set_xlabel('Year')
ax.set_ylabel('Mean Times Cited')
ax.set_title('Mean Citation Count by Publication Year')
ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/home/ubuntu/neuroblastoma/figures/fig2_mean_citations_per_year.pdf")
plt.savefig("/home/ubuntu/neuroblastoma/figures/fig2_mean_citations_per_year.png")
plt.close()
print("  Fig 2 done")

# ── Fig 3: Feature distributions ──
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
plot_features = ['prestige_cited', 'prestige_citing', 'temporal_distance',
                 'coauth_distance', 'semantic_similarity', 'same_journal']

for i, feat in enumerate(plot_features):
    ax = axes[i]
    if feat == 'same_journal':
        vals = [pos[feat].mean() * 100, neg[feat].mean() * 100]
        ax.bar(['Cited', 'Not Cited'], vals, color=['#2166ac', '#d01c8b'], alpha=0.8)
        ax.set_ylabel('% Same Journal')
    elif feat == 'coauth_distance':
        cap = 15
        p_vals = pos[feat].clip(upper=cap)
        n_vals = neg[feat].clip(upper=cap)
        ax.hist(p_vals[p_vals < cap], bins=15, alpha=0.6, color='#2166ac', label='Cited', density=True)
        ax.hist(n_vals[n_vals < cap], bins=15, alpha=0.6, color='#d01c8b', label='Not Cited', density=True)
        ax.legend()
    else:
        ax.hist(pos[feat], bins=50, alpha=0.6, color='#2166ac', label='Cited', density=True)
        ax.hist(neg[feat], bins=50, alpha=0.6, color='#d01c8b', label='Not Cited', density=True)
        ax.legend()
    ax.set_title(FEATURE_LABELS[feat])
    ax.set_xlabel(FEATURE_LABELS[feat])

plt.suptitle('Feature Distributions: Cited vs. Not-Cited Pairs', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("/home/ubuntu/neuroblastoma/figures/fig3_feature_distributions.pdf")
plt.savefig("/home/ubuntu/neuroblastoma/figures/fig3_feature_distributions.png")
plt.close()
print("  Fig 3 done")

# ── Fig 4: Model comparison (ROC-AUC) ──
model_names = list(cv_results.keys())
aucs  = [cv_results[m]['roc_auc']['mean'] for m in model_names]
stds  = [cv_results[m]['roc_auc']['std']  for m in model_names]
order = np.argsort(aucs)[::-1]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh([model_names[i] for i in order],
               [aucs[i] for i in order],
               xerr=[stds[i] for i in order],
               color=PALETTE[:len(model_names)], alpha=0.85, capsize=4)
ax.set_xlabel('ROC-AUC (5-fold CV mean ± std)')
ax.set_title('Model Comparison: ROC-AUC')
ax.set_xlim(0.92, 0.98)
for bar, auc in zip(bars, [aucs[i] for i in order]):
    ax.text(auc + 0.001, bar.get_y() + bar.get_height()/2,
            f'{auc:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig("/home/ubuntu/neuroblastoma/figures/fig4_model_comparison.pdf")
plt.savefig("/home/ubuntu/neuroblastoma/figures/fig4_model_comparison.png")
plt.close()
print("  Fig 4 done")

# ── Fig 5: Schematic ROC curves from saved AUC values ──
# We compute one representative ROC curve per model using a small sample
# to illustrate the shape, then label with the actual CV AUC.
X = df[FEATURE_COLS].values.copy()
X[:, 3] = np.where(X[:, 3] >= 999, 20.0, X[:, 3])
y = df['label'].values

# Use a 10% stratified sample for fast ROC curve generation
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED)
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s  = sc.transform(X_te)

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

top_models_roc = {
    'Neural Network':    MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=SEED),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=SEED),
    'Random Forest':     RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=SEED),
}

fig, ax = plt.subplots(figsize=(7, 6))
colors_roc = ['#2166ac', '#d01c8b', '#4dac26', '#f1a340']

for (name, model), color in zip(top_models_roc.items(), colors_roc):
    model.fit(X_tr_s, y_tr)
    proba = model.predict_proba(X_te_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, proba)
    auc_val = cv_results[name]['roc_auc']['mean']
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f'{name} (CV AUC={auc_val:.4f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5000)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves (representative sample; CV AUC in legend)')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig("/home/ubuntu/neuroblastoma/figures/fig5_roc_curves.pdf")
plt.savefig("/home/ubuntu/neuroblastoma/figures/fig5_roc_curves.png")
plt.close()
print("  Fig 5 done")

# ── Fig 6: SHAP feature importance ──
shap_sorted = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
feat_names  = [FEATURE_LABELS[k] for k, v in shap_sorted]
feat_vals   = [v for k, v in shap_sorted]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(feat_names[::-1], feat_vals[::-1], color='#2166ac', alpha=0.8)
ax.set_xlabel('Mean |SHAP Value|')
ax.set_title('Feature Importance (SHAP, Neural Network)')
for bar, val in zip(bars, feat_vals[::-1]):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig("/home/ubuntu/neuroblastoma/figures/fig6_shap_importance.pdf")
plt.savefig("/home/ubuntu/neuroblastoma/figures/fig6_shap_importance.png")
plt.close()
print("  Fig 6 done")

# ── Fig 7: Ablation study ──
abl_names = list(ablation.keys())
abl_aucs  = [ablation[k]['mean'] for k in abl_names]
abl_stds  = [ablation[k]['std']  for k in abl_names]
abl_order = np.argsort(abl_aucs)[::-1]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh([abl_names[i] for i in abl_order],
               [abl_aucs[i] for i in abl_order],
               xerr=[abl_stds[i] for i in abl_order],
               color='#4dac26', alpha=0.8, capsize=4)
ax.set_xlabel('ROC-AUC (5-fold CV mean ± std)')
ax.set_title('Ablation Study: Feature Subset Importance (Random Forest)')
ax.set_xlim(0.80, 0.98)
for bar, auc in zip(bars, [abl_aucs[i] for i in abl_order]):
    ax.text(auc + 0.001, bar.get_y() + bar.get_height()/2,
            f'{auc:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig("/home/ubuntu/neuroblastoma/figures/fig7_ablation.pdf")
plt.savefig("/home/ubuntu/neuroblastoma/figures/fig7_ablation.png")
plt.close()
print("  Fig 7 done")

# ── Fig 8: Feature correlation heatmap ──
X_cap = df[FEATURE_COLS].copy()
X_cap['coauth_distance'] = X_cap['coauth_distance'].clip(upper=20)
corr = X_cap.rename(columns=FEATURE_LABELS).corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=ax, linewidths=0.5, square=True)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig("/home/ubuntu/neuroblastoma/figures/fig8_feature_correlation.pdf")
plt.savefig("/home/ubuntu/neuroblastoma/figures/fig8_feature_correlation.png")
plt.close()
print("  Fig 8 done")

print("\nAll figures generated successfully.")
print("Files in figures/:")
for f in sorted(os.listdir("/home/ubuntu/neuroblastoma/figures/")):
    print(f"  {f}")
