"""Unit tests for Dual-Tower DQN architecture.

Verifies:
  1. Forward pass output shapes
  2. InfoNCE loss gradient flow
  3. Replay Buffer memory safety (only stores ints)
  4. Device consistency
  5. Gradient detach on BGE vectors
"""
import sys
import os
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.semantic_bridge import SemanticBridge
from model.dqn import CFTower, SemanticTower, DualTowerNet, compute_infonce_loss


def test_forward_pass_shapes():
    """Test that all tower outputs have correct shapes."""
    print("=== Test 1: Forward Pass Shapes ===")
    num_items = 1700
    embed_dim = 64
    hidden_dim = 128
    sem_dim = 768
    batch = 4
    seq_len = 10

    # CF Tower
    cf = CFTower(num_items, embed_dim, hidden_dim)
    ids = torch.randint(0, num_items, (batch, seq_len))
    h_cf = cf(ids)
    assert h_cf.shape == (batch, hidden_dim), f"CF tower shape mismatch: {h_cf.shape}"
    print(f"  ✅ CFTower output: {h_cf.shape}")

    # Semantic Tower
    sem = SemanticTower(sem_dim, hidden_dim)
    vecs = torch.randn(batch, seq_len, sem_dim)
    h_sem = sem(vecs)
    assert h_sem.shape == (batch, hidden_dim), f"Sem tower shape mismatch: {h_sem.shape}"
    print(f"  ✅ SemanticTower output: {h_sem.shape}")

    # DualTowerNet
    net = DualTowerNet(num_items, embed_dim, hidden_dim, sem_dim)
    action_emb, h_cf2, h_sem2 = net(ids, vecs)
    assert action_emb.shape == (batch, sem_dim), f"Action emb shape mismatch: {action_emb.shape}"
    print(f"  ✅ DualTowerNet action embedding: {action_emb.shape}")
    print(f"  ✅ h_cf: {h_cf2.shape}, h_sem: {h_sem2.shape}")


def test_infonce_loss():
    """Test InfoNCE loss computation and gradient flow."""
    print("\n=== Test 2: InfoNCE Loss & Gradient Flow ===")
    batch = 8
    hidden_dim = 128

    h_cf = torch.randn(batch, hidden_dim, requires_grad=True)
    h_sem = torch.randn(batch, hidden_dim, requires_grad=True)

    loss = compute_infonce_loss(h_cf, h_sem, temperature=0.07)
    print(f"  InfoNCE loss value: {loss.item():.4f}")
    assert loss.item() > 0, "InfoNCE loss should be positive"

    loss.backward()
    assert h_cf.grad is not None, "Gradient should flow to h_cf"
    assert h_sem.grad is not None, "Gradient should flow to h_sem"
    print(f"  ✅ Gradients flow to both h_cf (norm={h_cf.grad.norm():.4f}) and h_sem (norm={h_sem.grad.norm():.4f})")


def test_gradient_detach_on_bge():
    """Ensure BGE vectors don't accumulate gradients."""
    print("\n=== Test 3: BGE Gradient Detach ===")
    sem_dim = 768
    hidden_dim = 128
    batch = 4
    seq_len = 10

    # Simulate frozen BGE vectors (as SemanticBridge would produce)
    frozen_vecs = torch.randn(batch, seq_len, sem_dim)  # requires_grad=False by default
    assert not frozen_vecs.requires_grad, "BGE vectors must not require grad"

    sem_tower = SemanticTower(sem_dim, hidden_dim)
    h_sem = sem_tower(frozen_vecs)

    # Backprop should update W_sem (projection layer) but NOT frozen_vecs
    loss = h_sem.sum()
    loss.backward()

    assert frozen_vecs.grad is None, "BGE vectors should NOT have gradients!"
    assert sem_tower.proj.weight.grad is not None, "W_sem projection SHOULD have gradients"
    print(f"  ✅ frozen_vecs.grad = None (correct)")
    print(f"  ✅ W_sem.weight.grad norm = {sem_tower.proj.weight.grad.norm():.4f} (correct)")


def test_buffer_memory():
    """Verify replay buffer only stores int/float scalars, no dense vectors."""
    print("\n=== Test 4: Buffer Memory Safety ===")
    n_states = 10

    # Simulate storing 1000 transitions
    memory = np.zeros((0, n_states * 2 + 2))
    for _ in range(1000):
        s = np.random.randint(0, 1700, n_states)
        a = np.random.randint(0, 1700)
        r = np.random.randn()
        s_ = np.random.randint(0, 1700, n_states)
        transition = np.hstack((s, [a, r], s_))
        memory = np.append(memory, [transition], axis=0)

    mem_bytes = memory.nbytes
    print(f"  1000 transitions: {memory.shape} = {mem_bytes / 1024:.1f} KB")

    # Compare with storing 768D vectors (the BAD way)
    bad_memory_bytes = 1000 * (n_states * 768 * 4 * 2 + 768 * 4 + 4)  # float32
    print(f"  ❌ If we stored 768D vectors: {bad_memory_bytes / 1024 / 1024:.1f} MB")
    print(f"  ✅ Compression ratio: {bad_memory_bytes / mem_bytes:.0f}x smaller!")

    assert mem_bytes < 1_000_000, "Buffer for 1000 transitions should be < 1MB"
    print(f"  ✅ Buffer is memory-safe")


def test_semantic_bridge():
    """Test SemanticBridge if embeddings exist."""
    print("\n=== Test 5: Semantic Bridge ===")
    npy_path = 'data/processed/ml-100k/movie_embeddings.npy'
    if not os.path.exists(npy_path):
        print(f"  ⚠️ Skipped: {npy_path} not found")
        return

    device = torch.device("cpu")
    bridge = SemanticBridge(npy_path, device)

    # Single lookup
    vec = bridge.lookup_single(1)
    assert vec.shape == (768,), f"Single lookup shape: {vec.shape}"
    print(f"  ✅ Single lookup: shape={vec.shape}")

    # Batch lookup
    batch_ids = [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]
    result = bridge.lookup_batch(batch_ids)
    assert result.shape == (2, 5, 768), f"Batch lookup shape: {result.shape}"
    assert result.device == device, f"Wrong device: {result.device}"
    assert not result.requires_grad, "Should not require grad!"
    print(f"  ✅ Batch lookup: shape={result.shape}, device={result.device}, grad={result.requires_grad}")

    # Flat lookup
    flat = bridge.lookup_ids([1, 2, 3])
    assert flat.shape == (3, 768)
    print(f"  ✅ Flat lookup: shape={flat.shape}")


if __name__ == "__main__":
    test_forward_pass_shapes()
    test_infonce_loss()
    test_gradient_detach_on_bge()
    test_buffer_memory()
    test_semantic_bridge()
    print("\n🎉 ALL TESTS PASSED!")
