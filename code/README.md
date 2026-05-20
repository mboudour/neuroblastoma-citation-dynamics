# Pipeline Code

The analysis pipeline is divided into four sequential stages. Each stage saves its outputs to `results/` so that stages can be run independently.

## Execution Order

```
stage1_features.py  →  stage2_sbert.py  →  stage3_ml.py  →  figures.py
```

## Stage Descriptions

### `stage1_features.py` — Citation Pair Construction & Non-Semantic Features
- Loads the Dimensions.ai neuroblastoma dataset
- Filters to peer-reviewed articles (1975–2023)
- Constructs 264,951 positive citation pairs from observed reference lists
- Constructs 264,951 temporally valid negative pairs (random non-citations, $t_{B'} \leq t_A$)
- Builds an incremental co-authorship graph (year by year) and computes Dijkstra shortest-path distances using only data up to year $t_A - 1$
- Computes prestige features (cited and citing paper) causally
- Computes temporal distance, same-journal, and open-access features
- **Output:** `results/stage1_pairs.pkl` (529,902 pairs × 6 features)

### `stage2_sbert.py` — Semantic Similarity (SBERT)
- Loads `results/stage1_pairs.pkl`
- Encodes all unique paper titles + abstracts using SBERT (`all-MiniLM-L6-v2`)
- Computes cosine similarity for each citation pair
- **Output:** `results/pairs_with_features.pkl` (529,902 pairs × 7 features, complete feature matrix)

### `stage3_ml.py` — Model Training, Evaluation, SHAP & Ablation
- Loads `results/pairs_with_features.pkl`
- Trains and evaluates 6 models (Logistic Regression, LinearSVC, k-NN, Random Forest, Gradient Boosting, Neural Network) using 5-fold stratified cross-validation
- Computes SHAP values for the best model (Neural Network)
- Runs ablation study across 7 feature subsets using Random Forest
- **Outputs:**
  - `results/cv_results.json` — per-model CV scores (AUC, F1, accuracy, precision, recall, MCC, log-loss)
  - `results/shap_importance.json` — mean |SHAP| per feature
  - `results/ablation_results.json` — AUC per feature subset

### `figures.py` — Figure Generation
- Loads all result JSONs and the feature matrix
- Generates all 8 manuscript figures (saved as PDF and PNG to `figures/`)
- **Output:** `figures/fig1_*.pdf/png` through `figures/fig8_*.pdf/png`

## Requirements

```
pandas
numpy
scikit-learn
networkx
sentence-transformers
shap
matplotlib
seaborn
```

Install with:
```bash
pip install pandas numpy scikit-learn networkx sentence-transformers shap matplotlib seaborn
```

## Notes

- Stage 1 is the most memory-intensive step (~3 GB RAM for the co-authorship graph on this dataset). It is designed to free graph memory before Stage 2 begins.
- Stage 2 requires a GPU or significant CPU time for SBERT encoding of ~49,000 papers.
- The raw dataset (`Dimensions_neuroblastoma_1975_2024_merged.pkl`) is not included in this repository due to size. It must be obtained via the Dimensions.ai API.
