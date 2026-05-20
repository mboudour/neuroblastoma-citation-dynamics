# Citation Dynamics in Neuroblastoma Research

A temporally-aware machine learning framework for predicting citation dynamics in neuroblastoma research literature (1975–2023).

## Repository Structure

```
├── paper/
│   ├── manuscript.tex       # LaTeX source of the manuscript
│   └── references.bib       # BibTeX reference file
├── figures/                 # All manuscript figures (PDF and PNG)
│   ├── fig1_publications_per_year.*
│   ├── fig2_mean_citations_per_year.*
│   ├── fig3_feature_distributions.*
│   ├── fig4_model_comparison.*
│   ├── fig5_roc_curves.*
│   ├── fig6_shap_importance.*
│   ├── fig7_ablation.*
│   └── fig8_feature_correlation.*
└── results/                 # Raw computed results (JSON)
    ├── cv_results.json          # 5-fold CV scores for all 6 models
    ├── shap_importance.json     # SHAP feature importances (Neural Network)
    ├── ablation_results.json    # Ablation study AUC scores
    └── dataset_stats.json       # Dataset summary statistics
```

## Dataset

51,652 neuroblastoma research articles (1975–2023) extracted from [Dimensions.ai](https://www.dimensions.ai/). Citation pairs: 529,902 (balanced: 264,951 positive + 264,951 negative), constructed with strict temporal causality enforcement.

## Key Results

| Model | ROC-AUC (5-fold CV) | F1-Score |
|---|---|---|
| Neural Network | **0.9719 ± 0.0004** | **0.9129** |
| Gradient Boosting | 0.9717 ± 0.0005 | 0.9129 |
| Random Forest | 0.9689 ± 0.0005 | 0.9106 |
| Logistic Regression | 0.9670 ± 0.0005 | 0.9059 |
| Linear SVM | 0.9670 ± 0.0005 | 0.9059 |
| k-NN (k=5) | 0.9502 ± 0.0008 | 0.9019 |

## Features

- **Semantic Similarity** (SBERT cosine similarity of title+abstract embeddings)
- **Prestige of Cited Paper** (cumulative citations up to year t−1)
- **Prestige of Citing Paper** (max author citation count up to year t−1)
- **Temporal Distance** (difference in publication years)
- **Co-authorship Distance** (Dijkstra shortest path, temporally restricted graph)
- **Same Journal** (binary)
- **Open Access** (binary)

All network and prestige features are computed strictly causally — using only data available prior to the citing paper's publication year.
