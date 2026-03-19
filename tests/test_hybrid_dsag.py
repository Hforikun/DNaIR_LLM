"""Verification Test for the Hybrid Semantic-Aware DSAG.

Proves that:
1. Pure CF similarity ignores cold-start items (0 co-occurrence).
2. Hybrid Similarity (CF + Semantic) successfully elevates cold-start items
   with high semantic similarity into the candidate pool.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train

class MockSemanticBridge:
    def __init__(self, num_items, sem_dim=768):
        self.num_items = num_items
        self.embeddings = np.random.randn(num_items, sem_dim).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        self._normalized_embeddings = self.embeddings / norms
        self.id_to_idx = {i: i for i in range(num_items)}
        
    def get_normalized_embeddings(self):
        return self._normalized_embeddings

class MockEnv:
    def __init__(self, action_space):
        self.action_space = action_space
        self.mask_list = []
        self.K = 10
        # Sparse CF matrix: item 0 has similarity with item 1 and 2 ONLY.
        # Item 3 is a PURE COLD START item (not in the matrix at all).
        self.item_sim_matrix = {
            '0': {
                '1': 0.8,
                '2': 0.6
            }
        }
        
    def reset(self, state):
        return state

class MockAgent:
    def choose_action(self, s, env, I_sim_list):
        # We just return the candidate list to inspect it
        return I_sim_list

def test_hybrid_dsag():
    print("=" * 60)
    print("TEST: Hybrid Semantic-Aware DSAG (Breaking Cold-Start)")
    print("=" * 60)
    
    num_items = 5
    action_space = list(range(num_items))
    env = MockEnv(action_space)
    # 💥 CRITICAL: Restrict K=2 so we can see who gets "filtered out" 💥
    env.K = 2  
    
    agent = MockAgent()
    bridge = MockSemanticBridge(num_items)
    train._semantic_bridge = bridge
    
    # Let's set deterministic semantic distances
    # Item 0 is the last watched item.
    # We want: 
    # Item 1: high CF (0.8), bad semantic (-1.0 -> norm 0.0)
    # Item 2: med CF (0.6), bad semantic (-1.0 -> norm 0.0)
    # Item 3: ZERO CF (0.0), PERFECT semantic (1.0 -> norm 1.0)
    
    vec_0 = np.ones(768)
    vec_bad = -np.ones(768)
    vec_good = np.ones(768)
    
    # Store them
    bridge._normalized_embeddings[0] = vec_0 / np.linalg.norm(vec_0)
    bridge._normalized_embeddings[1] = vec_bad / np.linalg.norm(vec_bad)
    bridge._normalized_embeddings[2] = vec_bad / np.linalg.norm(vec_bad)
    bridge._normalized_embeddings[3] = vec_good / np.linalg.norm(vec_good)
    bridge._normalized_embeddings[4] = vec_bad / np.linalg.norm(vec_bad)
    
    last_obs = [0]
    
    print("\n[Case 1] Pure CF (alpha = 1.0): Original DNaIR behavior")
    candidates_cf = train.recommend_offpolicy(env, agent, last_obs, alpha=1.0)
    print(f"Candidates retrieved (Top 2): {candidates_cf}")
    assert 3 not in candidates_cf, "Pure CF should ignore Item 3 entirely!"
    assert 1 in candidates_cf and 2 in candidates_cf, "Pure CF should retrieve Items 1 and 2."
    print("✅ Correct: Only items 1 and 2 (from CF matrix) were retrieved.")
        
    print("\n[Case 2] Hybrid Similarity (alpha = 0.5): LLM Enhanced behavior")
    candidates_hybrid = train.recommend_offpolicy(env, agent, last_obs, alpha=0.5)
    print(f"Candidates retrieved (Top 2): {candidates_hybrid}")
    assert 3 in candidates_hybrid, "Hybrid MUST retrieve Item 3 due to high semantic similarity!"
    
    rank = candidates_hybrid.index(3)
    print(f"✅ Correct: Cold-start Item 3 successfully bypassed the filter and ranked #{rank+1}!")
    print("\n🎉 HYBRID DSAG TEST PASSED!")

if __name__ == "__main__":
    test_hybrid_dsag()
