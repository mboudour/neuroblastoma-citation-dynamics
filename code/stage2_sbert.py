"""
Stage 2: Add SBERT semantic similarity to stage 1 pairs.
Run after stage1_features.py has completed and freed memory.
"""
import pandas as pd
import numpy as np
import os, json, gc, warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

print("=" * 60)
print("STAGE 2: SBERT Semantic Similarity")
print("=" * 60)

# Load stage 1 results
pairs_df = pd.read_pickle("/home/ubuntu/neuroblastoma/results/stage1_pairs.pkl")
print(f"Loaded pairs: {pairs_df.shape}")

# Load text data
df_raw = pd.read_pickle("/home/ubuntu/upload/Dimensions_neuroblastoma_1975_2024_merged.pkl")
df_raw = df_raw[df_raw['type'] == 'article'].dropna(subset=['id', 'year']).copy()
id_to_title    = dict(zip(df_raw['id'], df_raw['title'].fillna('')))
id_to_abstract = dict(zip(df_raw['id'], df_raw['abstract'].fillna('')))
del df_raw; gc.collect()

def get_text(pid):
    t = id_to_title.get(pid, '')
    a = id_to_abstract.get(pid, '')
    return (t + '. ' + a[:400]).strip() if (a and a != t) else t.strip()

unique_pids = list(set(pairs_df['citing_id'].tolist() + pairs_df['cited_id'].tolist()))
print(f"Unique papers to encode: {len(unique_pids)}")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

texts = [get_text(p) for p in unique_pids]
emb_list = []
for i in tqdm(range(0, len(texts), 512), desc="SBERT encoding"):
    emb_list.append(model.encode(texts[i:i+512], show_progress_bar=False, normalize_embeddings=True))
embeddings = np.vstack(emb_list)
pid_to_emb = dict(zip(unique_pids, embeddings))
del model, emb_list, embeddings, texts; gc.collect()

sem_sims = []
for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Cosine similarity"):
    e1 = pid_to_emb.get(row['citing_id'])
    e2 = pid_to_emb.get(row['cited_id'])
    sem_sims.append(float(np.dot(e1, e2)) if (e1 is not None and e2 is not None) else 0.0)
pairs_df['semantic_similarity'] = sem_sims
del pid_to_emb; gc.collect()

print(f"Semantic similarity stats:\n{pairs_df['semantic_similarity'].describe()}")

# Save final feature matrix
feature_cols = ['prestige_cited','prestige_citing','temporal_distance',
                'coauth_distance','semantic_similarity','same_journal','cited_oa','label']
pairs_df[feature_cols + ['citing_id','cited_id','citing_year','cited_year']].to_pickle(
    "/home/ubuntu/neuroblastoma/results/pairs_with_features.pkl")

print(f"\nFinal feature matrix saved: {pairs_df.shape}")
print(pairs_df[feature_cols].describe().to_string())

# Dataset stats
df2 = pd.read_pickle("/home/ubuntu/upload/Dimensions_neuroblastoma_1975_2024_merged.pkl")
df2 = df2[df2['type'] == 'article'].dropna(subset=['id','year']).copy()
df2['year'] = df2['year'].astype(int)

pos = pairs_df[pairs_df['label']==1]
neg = pairs_df[pairs_df['label']==0]

stats = {
    'n_articles': int(len(df2)),
    'year_min': int(df2['year'].min()),
    'year_max': int(df2['year'].max()),
    'n_positive_pairs': int(len(pos)),
    'n_negative_pairs': int(len(neg)),
    'n_total_pairs': int(len(pairs_df)),
    'pct_abstract': round(float(df2['abstract'].notna().mean() * 100), 1),
    'pct_reference_ids': round(float(df2['reference_ids'].notna().mean() * 100), 1),
    'mean_times_cited': round(float(df2['times_cited'].mean()), 2),
    'median_times_cited': round(float(df2['times_cited'].median()), 1),
    'pct_disconnected_coauth': round(float((pairs_df['coauth_distance']==999).mean() * 100), 1),
    'mean_temporal_distance_pos': round(float(pos['temporal_distance'].mean()), 2),
    'mean_temporal_distance_neg': round(float(neg['temporal_distance'].mean()), 2),
    'mean_semantic_sim_pos': round(float(pos['semantic_similarity'].mean()), 4),
    'mean_semantic_sim_neg': round(float(neg['semantic_similarity'].mean()), 4),
    'mean_prestige_cited_pos': round(float(pos['prestige_cited'].mean()), 4),
    'mean_prestige_cited_neg': round(float(neg['prestige_cited'].mean()), 4),
    'coauth_graph_nodes': 155804,
    'coauth_graph_edges': 1172571,
}
with open("/home/ubuntu/neuroblastoma/results/dataset_stats.json", 'w') as f:
    json.dump(stats, f, indent=2)
print("\nDataset stats:")
print(json.dumps(stats, indent=2))
print("\nStage 2 complete.")
