"""
Grid Search Hyperparameter Tuning for Dual-Tower DQN (DNaIR-LLM)
================================================================

Searches over key hyperparameters (lambda_cl, beta, alpha) to find
the optimal configuration on a given dataset.

Usage:
  python grid_search.py --dataset ml100k --episode_max 30 --step_max 5000 --j 8

Output:
  - Console ranking table
  - results/grid_search_results.csv
"""
import argparse
import csv
import itertools
import os
import random
import threading
import time

import numpy as np
import pandas as pd
import torch
import faiss

from model import environment, dqn
from model.semantic_bridge import SemanticBridge
from util.datasplit_util import data_split
from util.jsondict_util import load_dict
from util.metrics_util import (ndcg_metric, novelty_metric, ils_metric,
                                interdiv_metric, recall_metric, ltc_metric, mrmc_metric)
from util.popularity_util import item_popularity_generate
from util.quality_util import item_quality_generate
from util.simmatrix_util import sim_matrix_generate


# ════════════════════════════════════════════════
#  Core single-run evaluation (self-contained)
# ════════════════════════════════════════════════

def run_single_config(train_df, test_df, train_dict,
                      item_sim_dict, item_quality_dict, item_pop_dict,
                      max_item_id, item_list, mask_list,
                      semantic_bridge, faiss_index, device,
                      head_items_set, tail_items_set, cold_items_set,
                      args, lambda_cl, beta, alpha):
    """Train & evaluate ONE hyperparameter configuration. Returns a metrics dict."""
    
    _lock = threading.Lock()
    user_num = 0
    precision_list, ndcg_list, recall_list = [], [], []
    ndcg_cold_list, recall_cold_list = [], []
    novelty_list, coverage_list, ils_list = [], [], []
    interdiv_list, rec_lists_all = [], []

    def _recommender(ep_user):
        nonlocal user_num

        last_obs = train_dict[ep_user][-args.obswindow:]
        mask_list_copy = list(mask_list) + train_dict[ep_user][:-1]

        env = environment.Env(
            ep_user, train_dict[ep_user][-args.obswindow:],
            list(range(max_item_id)),
            item_sim_dict, item_pop_dict, item_quality_dict,
            mask_list_copy, args.topk,
            beta=beta  # ← Inject grid search beta
        )

        agent = dqn.DQN(
            n_states=args.obswindow,
            n_actions=env.n_actions,
            num_items=max_item_id,
            memory_capacity=args.memory,
            lr=args.lr,
            epsilon=args.epsilon,
            target_network_replace_freq=args.replace_freq,
            batch_size=args.batch,
            gamma=args.gamma,
            tau=args.tau,
            K=args.topk,
            semantic_bridge=semantic_bridge,
            faiss_index=faiss_index,
            embed_dim=64,
            hidden_dim=128,
            sem_dim=768,
            lambda_cl=lambda_cl,  # ← Inject grid search lambda_cl
            temperature=0.07,
            device=device
        )

        # Fill replay buffer
        user_df = train_df[train_df['user_id'] == ep_user]
        state_list = []
        for obs in user_df['item_id'].rolling(args.obswindow):
            if len(obs) != args.obswindow:
                continue
            state_list.append(list(obs))
        interaction_num = 0
        for s_idx in range(len(state_list) - 1):
            s = np.array(env.reset(state_list[s_idx]), dtype=int)
            a = int(state_list[s_idx + 1][0])
            s_, r, done = env.step(a)
            agent.store_transition(s, a, r, s_)
            interaction_num += 1

        if interaction_num <= 20:
            return

        # Train
        for step in range(args.step_max):
            agent.learn()

        # Recommend using Hybrid DSAG with injected alpha
        state = np.array(last_obs, dtype=int)
        s = env.reset(state)
        last_item = s[-1]

        item_sim_dict_cf = env.item_sim_matrix.get(str(last_item), {})
        all_emb = semantic_bridge.get_normalized_embeddings()
        last_item_idx = semantic_bridge.id_to_idx.get(int(last_item), None)
        if last_item_idx is not None:
            last_item_emb = all_emb[last_item_idx:last_item_idx+1]
            sem_sim_scores = (all_emb @ last_item_emb.T).flatten()
        else:
            sem_sim_scores = np.zeros(all_emb.shape[0])

        hybrid_sim_dict = {}
        for action in env.action_space:
            if action in env.mask_list:
                continue
            sim_cf = item_sim_dict_cf.get(str(action), 0.0)
            cand_idx = semantic_bridge.id_to_idx.get(int(action), None)
            if cand_idx is not None:
                sim_sem = (sem_sim_scores[cand_idx] + 1) / 2.0
            else:
                sim_sem = 0.5
            hybrid_sim = alpha * float(sim_cf) + (1.0 - alpha) * float(sim_sem)
            hybrid_sim_dict[action] = hybrid_sim

        sorted_I = sorted(hybrid_sim_dict.items(), key=lambda x: x[1], reverse=True)
        I_sim_list = [item_id for item_id, score in sorted_I[:env.K]]
        rec_list = agent.choose_action(s, env, I_sim_list)

        test_set = test_df.loc[test_df['user_id'] == ep_user, 'item_id'].tolist()

        with _lock:
            nonlocal user_num
            user_num += 1
            precision_list.append(len(set(rec_list) & set(test_set)) / len(rec_list))
            ndcg_list.append(ndcg_metric({ep_user: rec_list}, {ep_user: test_set}))
            recall_list.append(recall_metric({ep_user: rec_list}, {ep_user: test_set}))

            cold_test_set = list(set(test_set) & cold_items_set)
            if cold_test_set:
                cold_rec = [item for item in rec_list if item in cold_items_set]
                if cold_rec:
                    ndcg_cold_list.append(ndcg_metric({ep_user: cold_rec}, {ep_user: cold_test_set}))
                else:
                    ndcg_cold_list.append(0.0)
                recall_cold_list.append(recall_metric({ep_user: rec_list}, {ep_user: cold_test_set}))

            novelty_list.append(novelty_metric(rec_list, env.item_pop_dict))
            coverage_list.extend(rec_list)
            ils_list.append(ils_metric(rec_list, env.item_sim_matrix))
            rec_lists_all.append(rec_list)

    # Run episodes
    episodes = random.sample(list(train_dict.keys()), min(args.episode_max, len(train_dict)))

    if args.j == 1:
        for ep_user in episodes:
            _recommender(ep_user)
    else:
        from concurrent.futures import ThreadPoolExecutor, wait
        executor = ThreadPoolExecutor(max_workers=args.j)
        futures = [executor.submit(_recommender, ep_user) for ep_user in episodes]
        wait(futures)

    # Collect metrics
    metrics = {
        'lambda_cl': lambda_cl,
        'beta': beta,
        'alpha': alpha,
        'users': user_num,
        'precision': np.mean(precision_list) if precision_list else 0,
        'recall': np.mean(recall_list) if recall_list else 0,
        'ndcg': np.mean(ndcg_list) if ndcg_list else 0,
        'ndcg_cold': np.mean(ndcg_cold_list) if ndcg_cold_list else 0,
        'recall_cold': np.mean(recall_cold_list) if recall_cold_list else 0,
        'novelty': (1 - np.mean(novelty_list)) if novelty_list else 0,
        'coverage': len(set(coverage_list)) / max_item_id if coverage_list else 0,
        'ltc': ltc_metric(coverage_list, tail_items_set) if coverage_list else 0,
        'mrmc': mrmc_metric(rec_lists_all, head_items_set) if rec_lists_all else 0,
        'ils': np.mean(ils_list) if ils_list else 0,
    }
    return metrics


# ════════════════════════════════════════════════
#  Main Grid Search Driver
# ════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Grid Search for Dual-Tower DQN')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='ml100k')
    parser.add_argument('--obswindow', type=int, default=10)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--memory', type=int, default=20000)
    parser.add_argument('--replace_freq', type=int, default=99)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epsilon', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.90)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--episode_max', type=int, default=30,
                        help='Episodes per config (smaller = faster search)')
    parser.add_argument('--step_max', type=int, default=5000,
                        help='Steps per episode (smaller = faster search)')
    parser.add_argument('--j', type=int, default=1,
                        help='Worker threads (use 8-16 on CUDA servers)')
    args = parser.parse_args()

    # ── Search Space ──
    LAMBDA_CL_GRID = [0.01, 0.05, 0.1, 0.5]
    BETA_GRID = [0.2, 0.4, 0.6, 0.8]
    ALPHA_GRID = [0.3, 0.5, 0.7]

    all_combos = list(itertools.product(LAMBDA_CL_GRID, BETA_GRID, ALPHA_GRID))
    total_runs = len(all_combos)
    print(f"\n{'='*60}")
    print(f"  GRID SEARCH: {total_runs} configurations")
    print(f"  Search Space: λ_cl={LAMBDA_CL_GRID} × β={BETA_GRID} × α={ALPHA_GRID}")
    print(f"  Per-config: {args.episode_max} episodes × {args.step_max} steps, j={args.j}")
    print(f"{'='*60}\n")

    # ── Load Dataset ──
    dataset = args.dataset
    dat_path = f'./dataset/{dataset}/{dataset}.dat'
    df = pd.read_csv(dat_path, sep=',', names=['user_id', 'item_id', 'ratings', 'timestamp'])
    max_item_id = df['item_id'].max()
    item_list = df['item_id'].tolist()
    mask_list = list(set(list(range(max_item_id))) - set(item_list))

    mat_path = f'./dataset/{dataset}/{dataset}.mat'
    if not os.path.exists(mat_path):
        sim_matrix_generate(dat_path, mat_path)
    item_sim_dict = load_dict(mat_path)

    qua_path = f'./dataset/{dataset}/{dataset}.qua'
    if not os.path.exists(qua_path):
        item_quality_generate(dat_path, qua_path)
    item_quality_dict = load_dict(qua_path)

    pop_path = f'./dataset/{dataset}/{dataset}.pop'
    if not os.path.exists(pop_path):
        item_popularity_generate(dat_path, pop_path)
    item_pop_dict = load_dict(pop_path)

    train_path = f'./dataset/{dataset}/{dataset}.train'
    valid_path = f'./dataset/{dataset}/{dataset}.valid'
    test_path = f'./dataset/{dataset}/{dataset}.test'
    if not (os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path)):
        data_split(dat_path, train_path, valid_path, test_path)
    train_df = pd.read_csv(train_path, sep=',', names=['user_id', 'item_id', 'ratings', 'timestamp'])
    test_df = pd.read_csv(test_path, sep=',', names=['user_id', 'item_id', 'ratings', 'timestamp'])

    # ── Device Detection ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] Using: {device}")

    # ── Shared Resources ──
    npy_path = './data/processed/ml-100k/movie_embeddings.npy'
    semantic_bridge = SemanticBridge(npy_path, device)
    print(f"[SemanticBridge] Loaded {len(semantic_bridge.id_to_idx)} vectors")

    faiss_path = './data/processed/ml-100k/movie_index.faiss'
    faiss_index = faiss.read_index(faiss_path)
    print(f"[FAISS] Loaded {faiss_index.ntotal} vectors")

    # ── LLM Quality Scores ──
    llm_quality_path = './data/processed/ml-100k/quality_scores_llm.json'
    if os.path.exists(llm_quality_path):
        import json
        with open(llm_quality_path, 'r') as f:
            item_quality_dict = json.load(f)
        print(f"[LLM Reward] Loaded {len(item_quality_dict)} ψ̂_a scores")

    # ── Item Sets (using raw counts) ──
    raw_item_counts = train_df['item_id'].value_counts().to_dict()
    raw_items_sorted = sorted(raw_item_counts.items(), key=lambda x: x[1], reverse=True)
    num_head = int(len(raw_items_sorted) * 0.2)
    head_items_set = set([int(x[0]) for x in raw_items_sorted[:num_head]])
    tail_items_set = set([int(x[0]) for x in raw_items_sorted[num_head:]])
    cold_items_set = set([int(k) for k, v in raw_item_counts.items() if v <= 5])
    print(f"[Items] Head: {len(head_items_set)} | Tail: {len(tail_items_set)} | Cold(<=5): {len(cold_items_set)}")

    # ── Train Dict ──
    train_dict = {}
    for _, row in train_df.iterrows():
        train_dict.setdefault(int(row['user_id']), []).append(int(row['item_id']))

    # ── CSV Output ──
    os.makedirs('results', exist_ok=True)
    csv_path = f'results/grid_search_{dataset}.csv'
    fieldnames = ['rank', 'lambda_cl', 'beta', 'alpha', 'users',
                  'ndcg', 'precision', 'recall',
                  'ndcg_cold', 'recall_cold',
                  'novelty', 'coverage', 'ltc', 'mrmc', 'ils']

    all_results = []

    # ════════════════════════════════════════════════
    #  Execute Grid Search
    # ════════════════════════════════════════════════
    for run_idx, (lam, beta, alpha) in enumerate(all_combos, 1):
        print(f"\n{'─'*60}")
        print(f"  [{run_idx}/{total_runs}] λ_cl={lam}, β={beta}, α={alpha}")
        print(f"{'─'*60}")
        t_start = time.time()

        metrics = run_single_config(
            train_df, test_df, train_dict,
            item_sim_dict, item_quality_dict, item_pop_dict,
            max_item_id, item_list, mask_list,
            semantic_bridge, faiss_index, device,
            head_items_set, tail_items_set, cold_items_set,
            args, lam, beta, alpha
        )

        elapsed = time.time() - t_start
        all_results.append(metrics)

        print(f"  ⏱️  {elapsed:.1f}s | Users: {metrics['users']}")
        print(f"  NDCG={metrics['ndcg']:.4f} | Prec={metrics['precision']:.4f} | "
              f"Recall={metrics['recall']:.4f}")
        print(f"  NDCG_Cold={metrics['ndcg_cold']:.4f} | Recall_Cold={metrics['recall_cold']:.4f}")
        print(f"  Novelty={metrics['novelty']:.4f} | Coverage={metrics['coverage']:.3f} | "
              f"LtC={metrics['ltc']:.4f} | MRMC={metrics['mrmc']:.4f}")

    # ════════════════════════════════════════════════
    #  Ranking & Output
    # ════════════════════════════════════════════════

    # Composite Score: 40% NDCG + 30% Recall_Cold + 20% Coverage + 10% (1-MRMC)
    for r in all_results:
        r['composite'] = (0.4 * r['ndcg'] +
                          0.3 * r['recall_cold'] +
                          0.2 * r['coverage'] +
                          0.1 * (1 - r['mrmc']))

    all_results.sort(key=lambda x: x['composite'], reverse=True)

    print(f"\n{'='*60}")
    print(f"  🏆 GRID SEARCH RESULTS — Top 10 Configurations")
    print(f"{'='*60}")
    print(f"{'Rank':>4} | {'λ_cl':>6} | {'β':>4} | {'α':>4} | {'NDCG':>7} | "
          f"{'R_Cold':>7} | {'Cov%':>6} | {'MRMC':>6} | {'Score':>6}")
    print(f"{'─'*70}")

    for rank, r in enumerate(all_results[:10], 1):
        print(f"{rank:>4} | {r['lambda_cl']:>6.2f} | {r['beta']:>4.1f} | {r['alpha']:>4.1f} | "
              f"{r['ndcg']:>7.4f} | {r['recall_cold']:>7.4f} | "
              f"{r['coverage']*100:>5.1f}% | {r['mrmc']:>6.4f} | {r['composite']:>6.4f}")

    # Save CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for rank, r in enumerate(all_results, 1):
            r['rank'] = rank
            writer.writerow(r)

    print(f"\n📁 Full results saved to: {csv_path}")

    best = all_results[0]
    print(f"\n🥇 BEST CONFIG: λ_cl={best['lambda_cl']}, β={best['beta']}, α={best['alpha']}")
    print(f"   NDCG={best['ndcg']:.4f} | Recall_Cold={best['recall_cold']:.4f} | "
          f"Coverage={best['coverage']*100:.1f}% | MRMC={best['mrmc']:.4f}")


if __name__ == '__main__':
    main()
