"""
Stage 1: Build citation pairs + all non-SBERT features.
Saves intermediate result, then exits to free memory for Stage 2.
"""
import pandas as pd
import numpy as np
import networkx as nx
import os, json, warnings, gc
from collections import defaultdict
from tqdm import tqdm

warnings.filterwarnings('ignore')
os.makedirs("/home/ubuntu/neuroblastoma/results", exist_ok=True)

SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

print("=" * 60)
print("STAGE 1: Loading data")
print("=" * 60)

df = pd.read_pickle("/home/ubuntu/upload/Dimensions_neuroblastoma_1975_2024_merged.pkl")
df = df[df['type'] == 'article'].copy()
df = df.dropna(subset=['id', 'year'])
df['year'] = df['year'].astype(int)
df.rename(columns={'category_for_y': 'category_for',
                   'concepts_y': 'concepts',
                   'concepts_scores_y': 'concepts_scores',
                   'open_access_y': 'open_access'}, inplace=True)

id_to_year    = dict(zip(df['id'], df['year']))
id_to_tc      = dict(zip(df['id'], df['times_cited'].fillna(0)))
id_to_journal = dict(zip(df['id'], df['journal.id'].fillna('')))
id_to_oa      = {}
for _, row in df.iterrows():
    oa = row.get('open_access')
    id_to_oa[row['id']] = 1 if (isinstance(oa, dict) and oa.get('is_oa', False)) else 0

all_ids = set(df['id'].tolist())
papers_by_year = df.groupby('year')['id'].apply(list).to_dict()
years_sorted   = sorted(papers_by_year.keys())
year_arrays    = {yr: np.array(pids, dtype=object) for yr, pids in papers_by_year.items()}

print(f"Articles: {len(df)} | Years: {df['year'].min()}–{df['year'].max()}")

# ── Citation pairs ──
print("\nBuilding citation pairs...")
positive_pairs  = []
citing_to_cited = defaultdict(set)

for _, row in tqdm(df.iterrows(), total=len(df), desc="Positive pairs"):
    citing_id   = row['id']
    citing_year = row['year']
    ref_ids     = row['reference_ids']
    if not isinstance(ref_ids, list):
        continue
    for cited_id in ref_ids:
        if cited_id in all_ids:
            cited_year = id_to_year.get(cited_id)
            if cited_year is not None and cited_year < citing_year:
                positive_pairs.append((citing_id, cited_id, citing_year, cited_year))
                citing_to_cited[citing_id].add(cited_id)

print(f"Positive pairs: {len(positive_pairs)}")

negative_pairs = []
for (citing_id, _, citing_year, _) in tqdm(positive_pairs, desc="Negative sampling"):
    eligible = [y for y in years_sorted if y < citing_year]
    if not eligible:
        continue
    for _ in range(15):
        y_neg  = rng.choice(eligible)
        pool   = year_arrays[y_neg]
        neg_id = rng.choice(pool)
        if neg_id != citing_id and neg_id not in citing_to_cited[citing_id]:
            negative_pairs.append((citing_id, neg_id, citing_year, int(y_neg)))
            break

print(f"Negative pairs: {len(negative_pairs)}")

pos_df = pd.DataFrame(positive_pairs, columns=['citing_id','cited_id','citing_year','cited_year'])
pos_df['label'] = 1
neg_df = pd.DataFrame(negative_pairs, columns=['citing_id','cited_id','citing_year','cited_year'])
neg_df['label'] = 0
pairs_df = pd.concat([pos_df, neg_df], ignore_index=True)
pairs_df = pairs_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"Total: {len(pairs_df)} | Pos: {pairs_df['label'].sum()} | Neg: {(pairs_df['label']==0).sum()}")

del year_arrays; gc.collect()

# ── Scalar features ──
pairs_df['prestige_cited']    = pairs_df['cited_id'].map(id_to_tc).fillna(0).apply(np.log1p)
pairs_df['prestige_citing']   = pairs_df['citing_id'].map(id_to_tc).fillna(0).apply(np.log1p)
pairs_df['temporal_distance'] = pairs_df['citing_year'] - pairs_df['cited_year']
pairs_df['same_journal'] = (
    pairs_df['citing_id'].map(id_to_journal) == pairs_df['cited_id'].map(id_to_journal)
).astype(int)
pairs_df.loc[pairs_df['citing_id'].map(id_to_journal) == '', 'same_journal'] = 0
pairs_df['cited_oa'] = pairs_df['cited_id'].map(id_to_oa).fillna(0).astype(int)

# ── Co-authorship distance ──
print("\nCo-authorship distance...")

def extract_author_ids(authors_list):
    if not isinstance(authors_list, list):
        return []
    ids = []
    for a in authors_list:
        if not isinstance(a, dict):
            continue
        rid = a.get('researcher_id')
        if rid and rid not in ([], ''):
            ids.append(str(rid))
            continue
        fn = a.get('first_name', '').strip()
        ln = a.get('last_name', '').strip()
        if fn or ln:
            ids.append(f"{fn}_{ln}".lower())
    return ids

id_to_authors = {row['id']: extract_author_ids(row['authors']) for _, row in df.iterrows()}
del df; gc.collect()

pairs_sorted = pairs_df.sort_values('citing_year').reset_index(drop=True)
G = nx.Graph()
prev_built = None

def add_year(yr):
    for pid in papers_by_year.get(yr, []):
        auths = id_to_authors.get(pid, [])
        for i in range(len(auths)):
            for j in range(i+1, len(auths)):
                a1, a2 = auths[i], auths[j]
                if G.has_edge(a1, a2):
                    G[a1][a2]['weight'] = G[a1][a2].get('weight', 1) + 1
                else:
                    G.add_edge(a1, a2, weight=1)

def coauth_dist(cng_id, ctd_id):
    a1s = [a for a in id_to_authors.get(cng_id, []) if G.has_node(a)][:3]
    a2s = [a for a in id_to_authors.get(ctd_id, []) if G.has_node(a)][:3]
    if not a1s or not a2s:
        return 999
    best = 999
    for a1 in a1s:
        for a2 in a2s:
            if a1 == a2:
                return 0
            try:
                d = nx.shortest_path_length(G, a1, a2)
                if d < best:
                    best = d
                if best <= 1:
                    return best
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
    return best

dist_results = []
for citing_year, group in tqdm(pairs_sorted.groupby('citing_year'), desc="Co-auth distances"):
    for yr in years_sorted:
        if yr >= citing_year:
            break
        if prev_built is None or yr > prev_built:
            add_year(yr)
            prev_built = yr
    for idx, row in group.iterrows():
        dist_results.append((idx, coauth_dist(row['citing_id'], row['cited_id'])))

dist_map = dict(dist_results)
pairs_sorted['coauth_distance'] = pairs_sorted.index.map(dist_map).fillna(999)
pairs_df = pairs_sorted.sort_index().reset_index(drop=True)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Disconnected (999): {(pairs_df['coauth_distance']==999).sum()} ({100*(pairs_df['coauth_distance']==999).mean():.1f}%)")

# Save stage 1 results
pairs_df.to_pickle("/home/ubuntu/neuroblastoma/results/stage1_pairs.pkl")
print(f"\nStage 1 saved. Shape: {pairs_df.shape}")
print("Stage 1 complete.")
