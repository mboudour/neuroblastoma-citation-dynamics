# Citation Dynamics in Neuroblastoma Research
### A Temporally-Aware Machine Learning Framework for Predicting Scientific Citations

---

## Overview

This repository contains the code outputs, figures, and results for a study on citation dynamics in the neuroblastoma research literature. We apply a machine learning framework to predict whether one scientific paper will cite another, using a corpus of 51,652 peer-reviewed articles published between 1975 and 2023. A central design principle of this work is **strict temporal causality**: all features are computed using only information available prior to the citing paper's publication year, eliminating the data leakage that affects many prior citation prediction studies.

---

## Research Questions

This study addresses four core research questions:

**RQ1 — Predictability:** To what extent are citation events in a specialized biomedical field (neuroblastoma) predictable from observable features of the papers and their authors?

**RQ2 — Temporal validity:** Does eliminating temporal data leakage — a methodological flaw common in the citation prediction literature — substantially reduce model performance, or do citation dynamics remain highly predictable under strict causal constraints?

**RQ3 — Feature importance:** Which mechanisms drive citation behavior? Specifically, what is the relative contribution of (a) semantic content similarity, (b) social proximity via co-authorship networks, and (c) prestige (the Matthew effect)?

**RQ4 — Feature interactions:** Is it the combination of semantic and structural network features that enables accurate prediction, or can a single feature class alone explain citation behavior?

---

## Dataset

The dataset was extracted from [Dimensions.ai](https://www.dimensions.ai/) — a comprehensive scholarly database — using the keyword **"neuroblastoma"** searched in titles and abstracts. Neuroblastoma is a pediatric cancer of the peripheral nervous system and one of the most intensively studied childhood malignancies, making it an ideal domain for studying citation dynamics within a focused but large research community.

| Property | Value |
| :--- | :--- |
| **Source** | Dimensions.ai |
| **Query keyword** | "neuroblastoma" (title or abstract) |
| **Publication type** | Peer-reviewed journal articles |
| **Time span** | 1975–2023 (49 years) |
| **Total articles** | 51,652 |
| **Abstract coverage** | 89.7% |
| **Reference list coverage** | 75.5% |
| **Mean citations per paper** | 34.75 |
| **Median citations per paper** | 15.0 |

The dataset exhibits the characteristic exponential growth of biomedical literature, with publication volume increasing from fewer than 100 articles per year in the 1970s to over 3,000 per year by the 2020s.

### Citation Pair Construction

The classification task is: given a pair of papers $(A, B)$ where $B$ was published no later than $A$, predict whether $A$ cites $B$.

| Split | Count |
| :--- | :--- |
| **Positive pairs** (observed citations within corpus) | 264,951 |
| **Negative pairs** (temporally valid non-citations) | 264,951 |
| **Total pairs** | 529,902 |

Negative pairs are sampled strictly: for each positive pair $(A, B)$, a negative pair $(A, B')$ is constructed by drawing $B'$ uniformly from all papers published on or before $A$'s publication year that $A$ does not actually cite. This prevents the model from exploiting impossible temporal orderings.

---

## Methodology

### Temporally-Aware Feature Engineering

Seven features are engineered across four conceptual categories. All features that depend on the state of the literature (prestige, co-authorship) are computed using a **rolling causal window**: for a citing paper published in year $t$, only data from years $\leq t-1$ are used.

| Feature | Category | Description |
| :--- | :--- | :--- |
| **Semantic Similarity** | Semantic | Cosine similarity of SBERT (all-MiniLM-L6-v2) embeddings of title + abstract |
| **Prestige of Cited Paper** | Prestige | log(1 + citations received by $B$ up to year $t_A - 1$) |
| **Prestige of Citing Paper** | Prestige | Max author citation count (up to year $t_A - 1$) across all authors of $A$ |
| **Temporal Distance** | Temporal | $t_A - t_B$ (difference in publication years) |
| **Co-authorship Distance** | Network | Dijkstra shortest path in the cumulative co-authorship graph up to year $t_A - 1$; capped at 20 for disconnected pairs |
| **Same Journal** | Metadata | Binary: 1 if $A$ and $B$ appear in the same journal |
| **Open Access** | Metadata | Binary: 1 if $B$ is Open Access |

The co-authorship graph contains **155,804 nodes** and **1,172,571 edges**. Of all citation pairs, **43.4%** have no co-authorship path between the author sets (disconnected in the social network).

### Machine Learning Models

Six models are evaluated using **5-fold stratified cross-validation** on the balanced dataset:

- Logistic Regression
- Linear Support Vector Machine (LinearSVC)
- k-Nearest Neighbours (k=5)
- Random Forest (100 trees)
- Gradient Boosting (100 estimators)
- Neural Network (MLP: 128→64 hidden units, ReLU activation)

### Explainability

SHAP (SHapley Additive exPlanations) values are computed for the best-performing model (Neural Network) to quantify global feature importance and interpret the contribution of each feature to individual predictions.

---

## Results

### Model Performance

| Model | ROC-AUC | F1-Score | Accuracy | Precision | Recall | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Neural Network** | **0.9719 ± 0.0004** | **0.9129 ± 0.0007** | **0.9130 ± 0.0008** | 0.9138 ± 0.0032 | 0.9120 ± 0.0031 | **0.8259** |
| Gradient Boosting | 0.9717 ± 0.0005 | 0.9129 ± 0.0009 | 0.9130 ± 0.0009 | **0.9142 ± 0.0016** | 0.9116 ± 0.0013 | 0.8260 |
| Random Forest | 0.9689 ± 0.0005 | 0.9106 ± 0.0008 | 0.9107 ± 0.0009 | 0.9112 ± 0.0015 | 0.9100 ± 0.0013 | 0.8213 |
| Logistic Regression | 0.9670 ± 0.0005 | 0.9059 ± 0.0010 | 0.9057 ± 0.0010 | 0.9036 ± 0.0017 | 0.9082 ± 0.0010 | 0.8114 |
| Linear SVM | 0.9670 ± 0.0005 | 0.9059 ± 0.0010 | 0.9057 ± 0.0011 | 0.9037 ± 0.0017 | 0.9082 ± 0.0011 | 0.8114 |
| k-NN (k=5) | 0.9502 ± 0.0008 | 0.9019 ± 0.0009 | 0.9019 ± 0.0010 | 0.9022 ± 0.0017 | 0.9015 ± 0.0015 | 0.8038 |

The Neural Network achieves the highest ROC-AUC (0.9719), closely followed by Gradient Boosting (0.9717). The strong performance of non-linear models relative to Logistic Regression and Linear SVM indicates the presence of complex, non-linear interactions between features.

### Feature Importance (SHAP)

| Feature | Mean \|SHAP Value\| | Rank |
| :--- | :--- | :--- |
| Semantic Similarity | 0.2007 | 1 |
| Prestige (cited) | 0.1398 | 2 |
| Temporal Distance | 0.0882 | 3 |
| Co-authorship Distance | 0.0530 | 4 |
| Prestige (citing) | 0.0117 | 5 |
| Same Journal | 0.0055 | 6 |
| Open Access | 0.0000 | 7 |

**Semantic similarity is the dominant predictor** — papers that are conceptually close are far more likely to be cited. Prestige of the cited paper (the Matthew effect) is the second strongest driver, confirming that highly cited papers attract disproportionately more citations. Temporal distance ranks third, reflecting the natural obsolescence of older literature. Social proximity (co-authorship distance) plays a significant but secondary role.

Open Access status has no measurable effect on citation probability in this corpus, likely because neuroblastoma is a well-funded clinical research area with broad institutional access.

### Ablation Study

| Feature Subset | ROC-AUC | Δ vs. All Features |
| :--- | :--- | :--- |
| **All features** | **0.9683 ± 0.0005** | — |
| No semantic similarity | 0.8917 ± 0.0012 | −7.7% |
| No prestige | 0.9090 ± 0.0006 | −5.9% |
| No co-authorship distance | 0.9629 ± 0.0004 | −0.6% |
| Temporal + semantic only | 0.8882 ± 0.0007 | −8.3% |
| Semantic similarity only | 0.8491 ± 0.0008 | −12.3% |
| Network features only | 0.8388 ± 0.0008 | −13.4% |

The ablation results directly answer **RQ4**: neither semantic nor network features alone are sufficient. The combination of what a paper is about (semantics) and who wrote it / how established it is (network + prestige) is essential for high predictive accuracy. Removing semantic similarity causes the largest single-feature drop (−7.7 pp), while network features alone yield only AUC = 0.8388.

---

## Key Findings

1. **Citation dynamics are highly predictable even under strict temporal causality.** The best model achieves ROC-AUC = 0.9719 with no access to future network states, demonstrating that the predictive signal is genuine and not an artifact of data leakage.

2. **Semantic content is the primary driver of citation.** Papers with high SBERT cosine similarity (mean = 0.569 for cited pairs vs. 0.279 for non-cited pairs) are far more likely to be cited, confirming that topical relevance is the fundamental mechanism of citation.

3. **The Matthew effect is real and strong.** Prestige of the cited paper is the second most important feature. Cited papers have a mean log-prestige of 4.45 vs. 2.59 for non-cited papers — a 72% difference — confirming preferential attachment in the neuroblastoma literature.

4. **Social proximity matters but is secondary.** Co-authorship distance is a significant predictor (rank 4 in SHAP), but 43.4% of all citation pairs have no co-authorship path at all, limiting the reach of this mechanism.

5. **Temporal distance shapes citation probability.** Cited pairs have a mean temporal distance of 8.32 years vs. 18.73 years for non-cited pairs, reflecting the recency bias in scientific citation practice.

6. **Non-linear models outperform linear ones**, suggesting that the interaction between semantic and structural features is not captured by additive linear combinations.

---

## Repository Structure

```
├── paper/                   # Reserved for author-compiled manuscript (PDF)
├── figures/                 # All 8 manuscript figures (PDF and PNG)
│   ├── fig1_publications_per_year.*
│   ├── fig2_mean_citations_per_year.*
│   ├── fig3_feature_distributions.*
│   ├── fig4_model_comparison.*
│   ├── fig5_roc_curves.*
│   ├── fig6_shap_importance.*
│   ├── fig7_ablation.*
│   └── fig8_feature_correlation.*
└── results/                 # Raw computed results (JSON)
    ├── cv_results.json          # 5-fold CV scores for all 6 models (all metrics)
    ├── shap_importance.json     # SHAP feature importances (Neural Network)
    ├── ablation_results.json    # Ablation study AUC scores
    └── dataset_stats.json       # Dataset summary statistics
```

---

## Dependencies

- Python 3.11
- pandas, numpy, scikit-learn, networkx
- sentence-transformers (SBERT: all-MiniLM-L6-v2)
- shap, matplotlib, seaborn
- Data source: [Dimensions.ai](https://www.dimensions.ai/) (API access required)

---

## Citation

If you use this work, please cite the manuscript (forthcoming).
