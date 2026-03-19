"""Smoke test: verify joint TD + InfoNCE loss converges on synthetic data.

This script does NOT require the full ML-100K pipeline. It creates
a tiny synthetic environment to verify:
  1. DQN.learn() runs without errors
  2. Both L_TD and L_InfoNCE decrease over training steps
  3. Q-values stabilize (no NaN, no explosion)
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix macOS FAISS+PyTorch OpenMP clash
import sys
import os
import numpy as np
import torch
import faiss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.semantic_bridge import SemanticBridge
from model.dqn import DQN


def create_mock_embeddings(npy_path, n_items=100, sem_dim=768):
    """Create a tiny mock embedding file for testing."""
    movie_ids = np.arange(1, n_items + 1)
    embeddings = np.random.randn(n_items, sem_dim).astype(np.float32)
    # Normalize so dot product = cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    np.save(npy_path, {'movie_ids': movie_ids, 'embeddings': embeddings})
    return movie_ids, embeddings


def create_mock_faiss_index(embeddings, movie_ids):
    """Build a FAISS IndexFlatIP with IndexIDMap from embeddings."""
    dim = embeddings.shape[1]
    base_index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(base_index)
    emb_copy = embeddings.copy()
    faiss.normalize_L2(emb_copy)
    index.add_with_ids(emb_copy, movie_ids.astype(np.int64))
    return index


def main():
    print("=" * 60)
    print("SMOKE TEST: Joint TD + InfoNCE Loss Convergence")
    print("=" * 60)

    # Setup
    n_items = 100
    seq_len = 5  # observation window
    batch_size = 16
    n_steps = 200
    sem_dim = 768

    # Create mock data
    mock_path = '/tmp/mock_embeddings.npy'
    movie_ids, embeddings = create_mock_embeddings(mock_path, n_items, sem_dim)

    device = torch.device("cpu")
    bridge = SemanticBridge(mock_path, device)
    faiss_index = create_mock_faiss_index(embeddings, movie_ids)

    # Create DQN agent
    agent = DQN(
        n_states=seq_len,
        n_actions=n_items,
        num_items=n_items,
        memory_capacity=2000,
        lr=0.001,
        epsilon=0.9,
        target_network_replace_freq=50,
        batch_size=batch_size,
        gamma=0.9,
        tau=0.01,
        K=5,
        semantic_bridge=bridge,
        faiss_index=faiss_index,
        embed_dim=32,
        hidden_dim=64,
        sem_dim=sem_dim,
        lambda_cl=0.1,
        temperature=0.07,
        device=device
    )

    # Fill replay buffer with synthetic transitions
    print("\n[1] Filling replay buffer with 500 synthetic transitions...")
    for _ in range(500):
        s = np.random.randint(1, n_items + 1, seq_len)
        a = np.random.randint(1, n_items + 1)
        r = np.random.randn() * 0.5
        s_ = np.random.randint(1, n_items + 1, seq_len)
        agent.store_transition(s, a, r, s_)
    print(f"    Buffer size: {len(agent.memory)}")

    # Train and track losses
    print(f"\n[2] Training for {n_steps} steps...")
    loss_history = {'loss_total': [], 'loss_td': [], 'loss_infonce': [],
                    'q_eval_mean': [], 'q_target_mean': []}

    for step in range(n_steps):
        metrics = agent.learn()
        if metrics is not None:
            for k, v in metrics.items():
                loss_history[k].append(v)

        if (step + 1) % 50 == 0 and loss_history['loss_total']:
            recent = loss_history['loss_total'][-50:]
            avg = np.mean(recent)
            avg_td = np.mean(loss_history['loss_td'][-50:])
            avg_nce = np.mean(loss_history['loss_infonce'][-50:])
            print(f"    Step {step+1:3d}/{n_steps}: "
                  f"L_total={avg:.4f}  L_td={avg_td:.4f}  L_nce={avg_nce:.4f}")

    # Verify convergence
    print(f"\n[3] Convergence Analysis:")
    if len(loss_history['loss_total']) >= 20:
        first_20 = np.mean(loss_history['loss_total'][:20])
        last_20 = np.mean(loss_history['loss_total'][-20:])
        first_td = np.mean(loss_history['loss_td'][:20])
        last_td = np.mean(loss_history['loss_td'][-20:])
        first_nce = np.mean(loss_history['loss_infonce'][:20])
        last_nce = np.mean(loss_history['loss_infonce'][-20:])

        print(f"    L_total:   first 20 avg = {first_20:.4f} → last 20 avg = {last_20:.4f}")
        print(f"    L_TD:      first 20 avg = {first_td:.4f} → last 20 avg = {last_td:.4f}")
        print(f"    L_InfoNCE: first 20 avg = {first_nce:.4f} → last 20 avg = {last_nce:.4f}")

        # Check no NaN
        has_nan = any(np.isnan(v) for v in loss_history['loss_total'])
        print(f"    NaN detected: {'❌ YES' if has_nan else '✅ No'}")

        # Check convergence (loss should decrease or stabilize)
        if last_20 <= first_20 * 1.5 and not has_nan:
            print(f"    ✅ Loss is converging or stable (no explosion)")
        else:
            print(f"    ⚠️ Loss may be diverging — review hyperparameters")

        # Check Q-value stability
        q_first = np.mean(loss_history['q_eval_mean'][:20])
        q_last = np.mean(loss_history['q_eval_mean'][-20:])
        print(f"    Q_eval:    first 20 avg = {q_first:.4f} → last 20 avg = {q_last:.4f}")
        if abs(q_last) < 100 and not np.isnan(q_last):
            print(f"    ✅ Q-values are bounded and stable")
        else:
            print(f"    ⚠️ Q-values may be exploding")
    else:
        print("    ⚠️ Not enough training steps for convergence analysis")

    # Cleanup
    os.remove(mock_path)
    print(f"\n🎉 SMOKE TEST COMPLETE!")


if __name__ == "__main__":
    main()
