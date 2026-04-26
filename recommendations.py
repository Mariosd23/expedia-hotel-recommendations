"""
# To Run:
1. Download train.tsv.gz from: 
   https://github.com/ExpediaGroup/pkdd22-challenge-expediagroup/releases
2. Place it in same directory as python file
3. Run: recommendatios.py

Hotel Recommendation System - FULL COMPARISON
Vector dimensions: 16, 32, 64, 100
K values: 10, 100
Context windows: 1, 2, 3, 5

approximate completion time : 40 minutes 
"""

import pandas as pd
from collections import defaultdict
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

print("="*80)
print("LOADING DATA")
print("="*80)
print("Loading 10% of data (100K rows from train.tsv.gz)...")
full_data = pd.read_csv("train.tsv.gz", sep="\t", nrows=100_000)
print(f"✓ Loaded {len(full_data)} rows\n")

print("Splitting into 80% train / 20% test...")
split_idx = int(len(full_data) * 0.8)
train_full = full_data[:split_idx].copy()
test_full = full_data[split_idx:].copy()

train_full["click_list"] = train_full["clicks"].str.split(",").apply(lambda x: [int(h) for h in x if h])
test_full["click_list"] = test_full["clicks"].str.split(",").apply(lambda x: [int(h) for h in x if h])

print(f"✓ Train: {len(train_full)} rows (80%)")
print(f"✓ Test: {len(test_full)} rows (20%)\n")

print("="*80)
print("BUILDING MODELS")
print("="*80)

print("Building Markov...")
transitions = defaultdict(lambda: defaultdict(int))
for seq in train_full["click_list"]:
    if len(seq) < 2: 
        continue
    for i in range(len(seq) - 1):
        transitions[seq[i]][seq[i + 1]] += 1
print(f"✓ Built transitions for {len(transitions):,} hotels\n")

print("Building Popularity baseline...")
hotel_counts = defaultdict(int)
for seq in train_full["click_list"]:
    for hotel in seq:
        hotel_counts[hotel] += 1
popular_hotels = [h for h, _ in sorted(hotel_counts.items(), key=lambda x: x[1], reverse=True)]
print(f"✓ Ranked {len(popular_hotels):,} hotels\n")

print("Training Word2Vec models with different dimensions...")
str_sequences = [[str(h) for h in seq] for seq in train_full["click_list"] if len(seq) >= 2]
models = {}
for dim in [16, 32, 64, 100]:
    print(f"  Training dim={dim}...", end="", flush=True)
    model = Word2Vec(
        sentences=str_sequences, 
        vector_size=dim, 
        window=6, 
        min_count=2,
        workers=8, 
        epochs=15, 
        sg=1,
        negative=5,
        seed=42
    )
    models[dim] = model
    print(f" ✓ {len(model.wv.key_to_index)} hotels")
print()

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def eval_word2vec_context_window_fast(test_data, model, k, window_size=3):
    """Context Window approach: average last N hotels"""
    hits, total, rr = 0, 0, []
    for seq in tqdm(test_data["click_list"], total=len(test_data), 
                    desc=f"W2V-Context(w={window_size})@{k}", leave=False):
        if len(seq) < 2: 
            continue
        for i in range(len(seq) - 1):
            context_start = max(0, i - window_size + 1)
            context = seq[context_start:i+1]
            actual = seq[i + 1]
            
            vectors = []
            for h in context:
                if str(h) in model.wv:
                    vectors.append(model.wv[str(h)])
            
            if not vectors:
                rr.append(0)
                total += 1
                continue
            
            context_vector = np.mean(vectors, axis=0)
            
            try:
                preds = model.wv.similar_by_vector(context_vector, topn=k)
                pred_ids = [int(h) for h, _ in preds]
                
                if actual in pred_ids:
                    hits += 1
                    rr.append(1.0 / (pred_ids.index(actual) + 1))
                else:
                    rr.append(0)
            except:
                rr.append(0)
            total += 1
    
    return (hits/total*100 if total > 0 else 0), (sum(rr)/len(rr) if rr else 0)


def eval_word2vec_last(test_data, model, k):
    """Standard W2Vec: use only last hotel"""
    hits, total, rr = 0, 0, []
    for seq in tqdm(test_data["click_list"], total=len(test_data), desc=f"W2V-Last@{k}", leave=False):
        if len(seq) < 2: 
            continue
        for i in range(len(seq) - 1):
            current, actual = seq[i], seq[i + 1]
            if str(current) not in model.wv:
                rr.append(0)
                total += 1
                continue
            try:
                preds = [int(h) for h, _ in model.wv.most_similar(str(current), topn=k)]
                if actual in preds:
                    hits += 1
                    rr.append(1.0 / (preds.index(actual) + 1))
                else:
                    rr.append(0)
            except:
                rr.append(0)
            total += 1
    return (hits/total*100 if total > 0 else 0), (sum(rr)/len(rr) if rr else 0)


def eval_markov(test_data, transitions, k):
    hits, total, rr = 0, 0, []
    for seq in tqdm(test_data["click_list"], total=len(test_data), desc=f"Markov@{k}", leave=False):
        if len(seq) < 2: 
            continue
        for i in range(len(seq) - 1):
            current, actual = seq[i], seq[i + 1]
            if current not in transitions:
                rr.append(0)
                total += 1
                continue
            preds = [h for h, _ in sorted(transitions[current].items(), key=lambda x: x[1], reverse=True)[:k]]
            if actual in preds:
                hits += 1
                rr.append(1.0 / (preds.index(actual) + 1))
            else:
                rr.append(0)
            total += 1
    return (hits/total*100 if total > 0 else 0), (sum(rr)/len(rr) if rr else 0)


def eval_popularity(test_data, popular_hotels, k):
    hits, total, rr = 0, 0, []
    top_k = popular_hotels[:k]
    for seq in tqdm(test_data["click_list"], total=len(test_data), desc=f"Popularity@{k}", leave=False):
        if len(seq) < 2: 
            continue
        for i in range(len(seq) - 1):
            actual = seq[i + 1]
            if actual in top_k:
                hits += 1
                rr.append(1.0 / (top_k.index(actual) + 1))
            else:
                rr.append(0)
            total += 1
    return (hits/total*100 if total > 0 else 0), (sum(rr)/len(rr) if rr else 0)


# ============================================================================
# EVALUATION
# ============================================================================

print("="*80)
print("EVALUATING MODELS ON TEST DATA")
print("="*80)

results = {}

for k in [10, 100]:
    print(f"\n{'='*80}")
    print(f"K = {k}")
    print(f"{'='*80}\n")
    
    # Baselines
    m_r, m_mrr = eval_markov(test_full, transitions, k)
    results[f'Markov@{k}'] = (m_r, m_mrr)
    print(f"Markov@{k}:                         Hits={m_r:6.2f}%  MRR={m_mrr:.4f}")
    
    p_r, p_mrr = eval_popularity(test_full, popular_hotels, k)
    results[f'Popularity@{k}'] = (p_r, p_mrr)
    print(f"Popularity@{k}:                     Hits={p_r:6.2f}%  MRR={p_mrr:.4f}\n")
    
    # Test all dimensions for W2V-Last
    print("W2V-Last (baseline - different dimensions):")
    for dim in [16, 32, 64, 100]:
        r, mrr = eval_word2vec_last(test_full, models[dim], k)
        results[f'W2V-Last-{dim}@{k}'] = (r, mrr)
        print(f"  W2V-Last-{dim}@{k}:                Hits={r:6.2f}%  MRR={mrr:.4f}")
    
    # Test all dimensions + context windows
    print(f"\nW2V-Context (with context windows):")
    for dim in [16, 32, 64, 100]:
        print(f"  dim={dim}:")
        for window in [1, 2, 3, 5]:
            r, mrr = eval_word2vec_context_window_fast(test_full, models[dim], k, window_size=window)
            results[f'W2V-Context-{dim}-w{window}@{k}'] = (r, mrr)
            print(f"    w={window}: Hits={r:6.2f}%  MRR={mrr:.4f}")

# ============================================================================
# COMPREHENSIVE SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL RESULTS - ALL METHODS (RANKED BY HITS%)")
print("="*80)
print(f"{'Method':<40} {'Hits (%)':<12} {'MRR':<10}")
print("-"*80)
for method, (hits, mrr) in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
    print(f"{method:<40} {hits:>10.2f}  {mrr:>10.4f}")
print("="*80)

# ============================================================================
# DIMENSION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("DIMENSION ANALYSIS: Impact of Vector Size")
print("="*80)

for k in [10, 100]:
    print(f"\n{'='*80}")
    print(f"K = {k}")
    print(f"{'='*80}\n")
    
    # W2V-Last performance by dimension
    print("W2V-Last performance by dimension:")
    last_by_dim = {}
    for dim in [16, 32, 64, 100]:
        hits = results[f'W2V-Last-{dim}@{k}'][0]
        mrr = results[f'W2V-Last-{dim}@{k}'][1]
        last_by_dim[dim] = hits
        print(f"  dim={dim:3d}: {hits:6.2f}% Hits, {mrr:.4f} MRR")
    
    best_dim = max(last_by_dim, key=last_by_dim.get)
    print(f"\n  ✓ Best dimension: {best_dim} ({last_by_dim[best_dim]:.2f}%)")
    
    # Context window best for each dimension
    print(f"\nContext Window best performance by dimension:")
    for dim in [16, 32, 64, 100]:
        window_results = {}
        for w in [1, 2, 3, 5]:
            hits = results[f'W2V-Context-{dim}-w{w}@{k}'][0]
            window_results[w] = hits
        
        best_w = max(window_results, key=window_results.get)
        baseline = last_by_dim[dim]
        improvement = window_results[best_w] - baseline
        
        print(f"  dim={dim:3d}: w={best_w} → {window_results[best_w]:6.2f}% ({improvement:+.2f}% vs W2V-Last)")

# ============================================================================
# CONTEXT WINDOW EFFECTIVENESS BY K
# ============================================================================

print("\n" + "="*80)
print("CONTEXT WINDOW EFFECTIVENESS SUMMARY")
print("="*80)

for k in [10, 100]:
    print(f"\n{'='*80}")
    print(f"K = {k}")
    print(f"{'='*80}\n")
    
    print(f"{'Dimension':<12} {'W2V-Last':<15} {'Best Context':<15} {'Window':<10} {'Improvement':<12}")
    print("-"*80)
    
    for dim in [16, 32, 64, 100]:
        baseline = results[f'W2V-Last-{dim}@{k}'][0]
        
        window_results = {}
        for w in [1, 2, 3, 5]:
            hits = results[f'W2V-Context-{dim}-w{w}@{k}'][0]
            window_results[w] = hits
        
        best_w = max(window_results, key=window_results.get)
        best_hits = window_results[best_w]
        improvement = best_hits - baseline
        
        print(f"dim={dim:<8} {baseline:>10.2f}%    {best_hits:>10.2f}%    w={best_w:<8} {improvement:>+8.2f}%")

print("\n" + "="*80)