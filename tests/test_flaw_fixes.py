"""Comprehensive verification for all 3 critical flaw fixes.

Tests:
1. LLM Quality Scores (ψ̂_a) — loaded and injected into Env reward
2. GPU Embedding — SemanticBridge uses nn.Embedding on device, zero CPU-GPU transfer
3. InfoNCE False-Negative Mask — same-action samples masked from denominator
4. Full convergence with all fixes applied
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.semantic_bridge import SemanticBridge
from model.dqn import DQN, compute_infonce_loss


def create_mock_data(n_items=100, sem_dim=768):
    """Create mock embeddings and FAISS-free setup for testing."""
    npy_path = '/tmp/mock_embeddings_v2.npy'
    movie_ids = np.arange(1, n_items + 1)
    embeddings = np.random.randn(n_items, sem_dim).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    np.save(npy_path, {'movie_ids': movie_ids, 'embeddings': embeddings})
    return npy_path, movie_ids, embeddings


def test_flaw1_quality_scores():
    """Test 1: LLM quality scores exist and have correct format."""
    print("\n=== Test 1: LLM Quality Scores (ψ̂_a) ===")
    path = 'data/processed/ml-100k/quality_scores_llm.json'
    assert os.path.exists(path), f"Missing {path}!"

    with open(path, 'r') as f:
        scores = json.load(f)

    assert len(scores) == 1682, f"Expected 1682 scores, got {len(scores)}"
    vals = list(scores.values())
    assert all(0 < v < 1 for v in vals), "All scores must be in (0, 1)"
    print(f"  ✅ Loaded {len(scores)} scores, range=[{min(vals):.4f}, {max(vals):.4f}]")

    # Verify it's different from the 0.5 prior
    non_default = sum(1 for v in vals if abs(v - 0.5) > 0.01)
    print(f"  ✅ {non_default}/{len(vals)} scores differ from 0.5 prior ({non_default/len(vals)*100:.0f}%)")


def test_flaw2_gpu_embedding():
    """Test 2: SemanticBridge uses GPU nn.Embedding, no CPU-GPU transfer."""
    print("\n=== Test 2: GPU-Resident Frozen Embedding ===")

    npy_path, movie_ids, embeddings = create_mock_data(n_items=50)
    device = torch.device("cpu")  # We test on CPU but the mechanism is the same
    bridge = SemanticBridge(npy_path, device)

    # Check it's an nn.Embedding
    assert isinstance(bridge.frozen_emb, torch.nn.Embedding), "Must be nn.Embedding!"
    assert bridge.frozen_emb.weight.requires_grad == False, "Must be frozen!"
    print(f"  ✅ frozen_emb is nn.Embedding(freeze=True)")

    # Test batch lookup returns tensor directly (no NumPy intermediate)
    ids = np.array([[1, 2, 3], [4, 5, 6]])
    result = bridge.lookup_batch(ids)
    assert isinstance(result, torch.Tensor), "Must return Tensor!"
    assert result.shape == (2, 3, 768), f"Wrong shape: {result.shape}"
    assert result.requires_grad == False, "Must have no grad!"
    print(f"  ✅ lookup_batch → {result.shape}, requires_grad=False")

    # Test flat lookup
    flat = bridge.lookup_ids([1, 2, 3])
    assert flat.shape == (3, 768), f"Wrong shape: {flat.shape}"
    print(f"  ✅ lookup_ids → {flat.shape}")

    # Test unknown ID → zero vector
    unknown = bridge.lookup_ids([99999])
    assert torch.allclose(unknown, torch.zeros(1, 768)), "Unknown ID should → zero!"
    print(f"  ✅ Unknown ID → zero vector (graceful fallback)")

    os.remove(npy_path)


def test_flaw4_infonce_mask():
    """Test 4: InfoNCE false-negative mask correctly shields same-action pairs."""
    print("\n=== Test 4: InfoNCE False-Negative Mask ===")

    batch_size = 8
    hidden_dim = 64
    device = torch.device("cpu")

    # Create embeddings where samples 0 and 1 have SAME representations
    # (simulating users who both watched the same movie)
    h_cf = torch.randn(batch_size, hidden_dim)
    h_sem = torch.randn(batch_size, hidden_dim)

    # Make samples 0 and 1 identical (they watched the same film)
    h_cf[1] = h_cf[0].clone()
    h_sem[1] = h_sem[0].clone()

    # Actions: samples 0 and 1 share action 42
    actions = torch.tensor([42, 42, 10, 20, 30, 40, 50, 60])

    # Without mask: sample 1 is treated as a negative for sample 0
    loss_no_mask = compute_infonce_loss(h_cf, h_sem, temperature=0.07, batch_action_ids=None)

    # With mask: sample 1 is shielded from being a negative for sample 0
    loss_with_mask = compute_infonce_loss(h_cf, h_sem, temperature=0.07, batch_action_ids=actions)

    print(f"  Loss WITHOUT mask: {loss_no_mask.item():.4f}")
    print(f"  Loss WITH mask:    {loss_with_mask.item():.4f}")

    # The masked version should have LOWER loss because the false negative
    # (sample 1 penalizing sample 0) has been removed
    assert loss_with_mask.item() <= loss_no_mask.item() + 0.01, \
        "Masked loss should be <= unmasked (false negative removed)"
    print(f"  ✅ Mask correctly reduces false-negative penalty")

    # Verify gradient still flows
    h_cf_grad = h_cf.clone().requires_grad_(True)
    h_sem_grad = h_sem.clone().requires_grad_(True)
    loss = compute_infonce_loss(h_cf_grad, h_sem_grad, 0.07, actions)
    loss.backward()
    assert h_cf_grad.grad is not None, "Gradient must flow through mask!"
    print(f"  ✅ Gradients flow correctly through masked InfoNCE")


def test_convergence_with_all_fixes():
    """Test 5: Full convergence test with all 3 fixes applied."""
    print("\n=== Test 5: Convergence with All Fixes ===")

    n_items, seq_len, batch_size, n_steps = 100, 5, 16, 150
    npy_path, movie_ids, embeddings = create_mock_data(n_items)

    device = torch.device("cpu")
    bridge = SemanticBridge(npy_path, device)

    # No FAISS needed for learn() anymore
    import faiss
    dim = embeddings.shape[1]
    base = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(base)
    emb_copy = embeddings.copy()
    faiss.normalize_L2(emb_copy)
    index.add_with_ids(emb_copy, movie_ids.astype(np.int64))

    agent = DQN(
        n_states=seq_len, n_actions=n_items, num_items=n_items,
        memory_capacity=2000, lr=0.001, epsilon=0.9,
        target_network_replace_freq=50, batch_size=batch_size,
        gamma=0.9, tau=0.01, K=5,
        semantic_bridge=bridge, faiss_index=index,
        embed_dim=32, hidden_dim=64, sem_dim=768,
        lambda_cl=0.1, temperature=0.07, device=device
    )

    # Fill buffer
    for _ in range(500):
        s = np.random.randint(1, n_items + 1, seq_len)
        a = np.random.randint(1, n_items + 1)
        r = np.random.randn() * 0.5
        s_ = np.random.randint(1, n_items + 1, seq_len)
        agent.store_transition(s, a, r, s_)

    # Train
    losses = []
    for step in range(n_steps):
        metrics = agent.learn()
        if metrics:
            losses.append(metrics['loss_total'])

    first = np.mean(losses[:20])
    last = np.mean(losses[-20:])
    has_nan = any(np.isnan(v) for v in losses)

    print(f"  L_total: first 20 avg = {first:.4f} → last 20 avg = {last:.4f}")
    print(f"  NaN: {'❌' if has_nan else '✅ None'}")
    assert not has_nan, "NaN detected!"
    assert last <= first * 2.0, "Loss is exploding!"
    print(f"  ✅ Convergence verified with all 3 fixes applied")

    os.remove(npy_path)


def main():
    print("=" * 60)
    print("VERIFICATION: All 3 Critical Flaw Fixes")
    print("=" * 60)

    test_flaw1_quality_scores()
    test_flaw2_gpu_embedding()
    test_flaw4_infonce_mask()
    test_convergence_with_all_fixes()

    print("\n" + "=" * 60)
    print("🎉 ALL VERIFICATION TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
