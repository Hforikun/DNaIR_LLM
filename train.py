import os
import random
import threading
import numpy as np
import torch
import faiss
from concurrent.futures import ThreadPoolExecutor, wait
from model import environment, dqn
from model.semantic_bridge import SemanticBridge
from util.metrics_util import ndcg_metric, novelty_metric, ils_metric, interdiv_metric, recall_metric, ltc_metric, mrmc_metric

user_num = 0
precision, ndcg, novelty, coverage, ils, interdiv = [], [], [], [], [], []
recall, ndcg_cold, recall_cold = [], [], []
rec_lists_all = []

# ======================== Global Shared Resources ========================
# These are initialized once and shared across all threads/episodes
_semantic_bridge = None
_faiss_index = None
_device = None
_head_items_set = set()
_tail_items_set = set()
_lock = threading.Lock()
_cold_items_set = set()


def setInteraction(env, agent, ep_user, train_df, obswindow):
    """Store historical transitions into the agent's replay buffer."""
    user_df = train_df[train_df['user_id'] == ep_user]
    state_list = []
    for obs in user_df['item_id'].rolling(obswindow):
        if len(obs) != obswindow:
            continue
        state_list.append(list(obs))
    interaction_num = 0
    for s_idx in range(len(state_list) - 1):
        s = np.array(env.reset(state_list[s_idx]), dtype=int)
        a = int(state_list[s_idx + 1][0])
        s_, r, done = env.step(a)
        agent.store_transition(s, a, r, s_)
        interaction_num += 1
    return interaction_num


def recommend_offpolicy(env, agent, last_obs, alpha=0.5):
    """Generate recommendations using the Hybrid Semantic-Aware DSAG.
    
    Diversified Stochastic Action Generator (DSAG):
    Instead of only looking at items with >0 ID co-occurrence similarity (which traps 
    cold-start items), we do a global scan across the action space using a dynamic
    hybrid score: Sim_Hybrid = alpha * Sim_ID + (1 - alpha) * Sim_Semantic.
    """
    global _semantic_bridge
    state = np.array(last_obs, dtype=int)
    s = env.reset(state)
    last_item = s[-1]

    # 1. Get CF ID similarities (sparse dictionary)
    item_sim_dict_cf = env.item_sim_matrix.get(str(last_item), {})

    # 2. Get Semantic cosine similarities via fast NumPy dot product
    all_emb = _semantic_bridge.get_normalized_embeddings()  # (N, 768)
    last_item_idx = _semantic_bridge.id_to_idx.get(int(last_item), None)
    
    if last_item_idx is not None:
        last_item_emb = all_emb[last_item_idx:last_item_idx+1]
        sem_sim_scores = (all_emb @ last_item_emb.T).flatten()  # (N,)
    else:
        sem_sim_scores = np.zeros(all_emb.shape[0])

    # 3. Global scan: compute Hybrid Similarity for ALL valid actions
    hybrid_sim_dict = {}
    for action in env.action_space:
        if action in env.mask_list:
            continue
            
        # CF Similarity (0 if never co-occurred = cold start)
        sim_cf = item_sim_dict_cf.get(str(action), 0.0)
        
        # Semantic Similarity (CosSim = [-1, 1], shifted to [0, 1] for scale alignment)
        cand_idx = _semantic_bridge.id_to_idx.get(int(action), None)
        if cand_idx is not None:
            sim_sem = (sem_sim_scores[cand_idx] + 1) / 2.0  # Normalize to [0, 1]
        else:
            sim_sem = 0.5  # Neutral fallback

        # Dynamic Fusion
        hybrid_sim = alpha * float(sim_cf) + (1.0 - alpha) * float(sim_sem)
        hybrid_sim_dict[action] = hybrid_sim

    # 4. Sort globally and extract the Similarity Subset (I_sim_list)
    sorted_I = sorted(hybrid_sim_dict.items(), key=lambda x: x[1], reverse=True)
    I_sim_list = [item_id for item_id, score in sorted_I[:env.K]]

    # 5. Pass bounded similarity subset to DQN for final decision
    result = agent.choose_action(s, env, I_sim_list)
    return result


def trainAgent(agent, step_max):
    """Train the agent for a fixed number of steps with loss tracking."""
    losses = {'loss_total': [], 'loss_td': [], 'loss_infonce': [],
              'q_eval_mean': [], 'q_target_mean': []}
    log_interval = max(step_max // 5, 1)  # Log ~5 times during training

    for step in range(step_max):
        metrics = agent.learn()
        if metrics is not None:
            for k, v in metrics.items():
                losses[k].append(v)

        # Periodic logging
        if (step + 1) % log_interval == 0 and losses['loss_total']:
            recent = min(log_interval, len(losses['loss_total']))
            avg_total = np.mean(losses['loss_total'][-recent:])
            avg_td = np.mean(losses['loss_td'][-recent:])
            avg_nce = np.mean(losses['loss_infonce'][-recent:])
            avg_qe = np.mean(losses['q_eval_mean'][-recent:])
            avg_qt = np.mean(losses['q_target_mean'][-recent:])
            print(f"    [Step {step+1}/{step_max}] "
                  f"L_total={avg_total:.4f} "
                  f"L_td={avg_td:.4f} "
                  f"L_nce={avg_nce:.4f} "
                  f"Q_eval={avg_qe:.3f} "
                  f"Q_tgt={avg_qt:.3f}")

    return losses


def recommender(ep_user, train_df, test_df, train_dict,
                item_sim_dict, item_quality_dict, item_pop_dict,
                max_item_id, mask_list, args):
    """Per-user training and evaluation pipeline."""
    global _semantic_bridge, _faiss_index, _device

    last_obs = train_dict[ep_user][-args.obswindow:]
    mask_list_copy = list(mask_list) + train_dict[ep_user][:-1]

    env = environment.Env(
        ep_user, train_dict[ep_user][-args.obswindow:],
        list(range(max_item_id)),
        item_sim_dict, item_pop_dict, item_quality_dict,
        mask_list_copy, args.topk
    )

    # Create Dual-Tower DQN agent with shared semantic bridge and FAISS index
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
        semantic_bridge=_semantic_bridge,
        faiss_index=_faiss_index,
        embed_dim=64,
        hidden_dim=128,
        sem_dim=768,
        lambda_cl=0.1,
        temperature=0.07,
        device=_device
    )

    interaction_num = setInteraction(env, agent, ep_user, train_df, args.obswindow)
    if interaction_num <= 20:
        return

    trainAgent(agent, args.step_max)
    rec_list = recommend_offpolicy(env, agent, last_obs)
    test_set = test_df.loc[test_df['user_id'] == ep_user, 'item_id'].tolist()

    # ── Thread-safe metric collection ──
    with _lock:
        global user_num
        user_num += 1

        # ── Accuracy Metrics ──
        precision.append(len(set(rec_list) & set(test_set)) / (len(rec_list)))
        ndcg.append(ndcg_metric({ep_user: rec_list}, {ep_user: test_set}))
        recall.append(recall_metric({ep_user: rec_list}, {ep_user: test_set}))

        # ── Strict Cold-Start Subset Metrics ──
        cold_test_set = list(set(test_set) & _cold_items_set)
        if cold_test_set:
            cold_rec_list = [item for item in rec_list if item in _cold_items_set]
            if cold_rec_list:
                ndcg_cold.append(ndcg_metric({ep_user: cold_rec_list}, {ep_user: cold_test_set}))
            else:
                ndcg_cold.append(0.0)
            recall_cold.append(recall_metric({ep_user: rec_list}, {ep_user: cold_test_set}))

        # ── Debiasing & Diversity Metrics ──
        novelty.append(novelty_metric(rec_list, env.item_pop_dict))
        coverage.extend(rec_list)
        ils.append(ils_metric(rec_list, env.item_sim_matrix))
        interdiv.append(rec_list)
        rec_lists_all.append(rec_list)

        print(f"  ✅ User {ep_user} done (total: {user_num})")


def train_dqn(train_df, test_df,
              item_sim_dict, item_quality_dict, item_pop_dict,
              max_item_id, item_list, mask_list, args):
    """Main training entry point for Dual-Tower DQN.
    
    Initializes shared resources (SemanticBridge, FAISS index, device)
    once, then distributes per-user training across threads.
    """
    global _semantic_bridge, _faiss_index, _device
    global _head_items_set, _tail_items_set, _cold_items_set

    # 1. Device detection (cloud GPU / Mac MPS / CPU)
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")
    print(f"[Device] Using: {_device}")

    # 2. Initialize Semantic Bridge (shared, read-only)
    npy_path = './data/processed/ml-100k/movie_embeddings.npy'
    _semantic_bridge = SemanticBridge(npy_path, _device)
    print(f"[SemanticBridge] Loaded {len(_semantic_bridge.id_to_idx)} item vectors, dim={_semantic_bridge.sem_dim}")

    # 3. Load FAISS index (shared, read-only Action Retriever)
    faiss_path = './data/processed/ml-100k/movie_index.faiss'
    _faiss_index = faiss.read_index(faiss_path)
    print(f"[FAISS] Loaded index with {_faiss_index.ntotal} vectors")

    # 4. Load LLM-generated quality scores (ψ̂_a) to upgrade dual reward
    llm_quality_path = './data/processed/ml-100k/quality_scores_llm.json'
    if os.path.exists(llm_quality_path):
        import json
        with open(llm_quality_path, 'r') as f:
            llm_quality_dict = json.load(f)
        # Override the original Bayesian quality dict with LLM zero-shot scores
        item_quality_dict = llm_quality_dict
        print(f"[LLM Reward] Loaded {len(llm_quality_dict)} semantic quality scores (ψ̂_a)")
    else:
        print(f"[LLM Reward] WARNING: {llm_quality_path} not found, using original quality_dict")

    # 5. Compute Item Sets for Evaluation (Head/Tail/Cold)
    #    CRITICAL: item_pop_dict contains Min-Max NORMALIZED values in [0,1].
    #    We must use RAW interaction counts from train_df for Head/Tail/Cold splits.
    raw_item_counts = train_df['item_id'].value_counts().to_dict()  # {item_id: raw_count}
    raw_items_sorted = sorted(raw_item_counts.items(), key=lambda x: x[1], reverse=True)
    num_head = int(len(raw_items_sorted) * 0.2)  # 20/80 Pareto Rule
    _head_items_set = set([int(x[0]) for x in raw_items_sorted[:num_head]])
    _tail_items_set = set([int(x[0]) for x in raw_items_sorted[num_head:]])
    _cold_items_set = set([int(k) for k, v in raw_item_counts.items() if v <= 5])
    print(f"[Items] Total: {max_item_id} | Head: {len(_head_items_set)} | Tail: {len(_tail_items_set)} | Cold(<=5): {len(_cold_items_set)}")

    # 6. Build per-user training dict
    train_dict = {}
    for index, row in train_df.iterrows():
        train_dict.setdefault(int(row['user_id']), list())
        train_dict[int(row['user_id'])].append(int(row['item_id']))

    # 7. Multi-threaded per-user training (or main thread if j=1)
    train_episodes = random.sample(list(train_dict.keys()), min(args.episode_max, len(train_dict)))
    print(f"[Training] Starting {len(train_episodes)} episodes with {args.j} workers...")

    if args.j == 1:
        # Run directly in the main python thread to bypass Mac OpenMP/FAISS ThreadPoolExecutor segfaults
        for ep_user in train_episodes:
            recommender(
                ep_user, train_df, test_df, train_dict,
                item_sim_dict, item_quality_dict, item_pop_dict,
                max_item_id, mask_list, args
            )
    else:
        futures = []
        executor = ThreadPoolExecutor(max_workers=args.j)
        for ep_user in train_episodes:
            future = executor.submit(
                recommender,
                ep_user, train_df, test_df, train_dict,
                item_sim_dict, item_quality_dict, item_pop_dict,
                max_item_id, mask_list, args
            )
            futures.append(future)
        wait(futures)

    # 8. Print 4-Dimensional Metrics
    print("\n===== Dual-Tower DQN Evaluation Results =====")
    print(f"Users evaluated: {user_num}")
    
    print("\n[Accuracy & Quality]")
    if precision: print(f"  Precision@{args.topk}: {np.mean(precision):.4f}")
    if recall:    print(f"  Recall@{args.topk}:    {np.mean(recall):.4f}")
    if ndcg:      print(f"  NDCG@{args.topk}:      {np.mean(ndcg):.4f}")
    
    print("\n[Strict Cold-Start Subset (interact <= 5)]")
    if ndcg_cold:   print(f"  NDCG_Cold@{args.topk}:   {np.mean(ndcg_cold):.4f}")
    if recall_cold: print(f"  Recall_Cold@{args.topk}: {np.mean(recall_cold):.4f}")

    print("\n[Debiasing & Diversity]")
    if rec_lists_all:
        try:
            mrmc = mrmc_metric(rec_lists_all, _head_items_set)
            print(f"  MRMC (Miscalibration): {mrmc:.4f}  (Lower is fairer)")
        except Exception as e:
            print(f"  MRMC (Miscalibration): Failed ({e})")
    
    if coverage:
        try:
            ltc = ltc_metric(coverage, _tail_items_set)
            print(f"  LtC (Long-tail Cov):   {ltc:.4f}  (Higher is more tail-focused)")
        except Exception as e:
            print(f"  LtC (Long-tail Cov):   Failed ({e})")
            
        print(f"  Catalog Coverage:      {len(set(coverage))}/{max_item_id} ({len(set(coverage))/max_item_id*100:.1f}%)")

    if novelty: print(f"  Novelty (Inverse Pop): {1 - np.mean(novelty):.4f}")
    if ils:     print(f"  ILS (Intra-list Sim):  {np.mean(ils):.4f}")

if __name__ == "__main__":
    pass
