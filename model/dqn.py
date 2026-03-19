import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss


# ======================== Utility Functions ========================

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# ====================== InfoNCE Contrastive Loss ======================

def compute_infonce_loss(h_cf, h_sem, temperature=0.07, batch_action_ids=None):
    """Maximize mutual information between CF view and Semantic view,
    with false-negative masking to prevent dimension collapse.

    For a batch of N states, the diagonal entries (i, i) are positive pairs
    (same state's CF and Semantic representations). Off-diagonal entries
    are negative pairs UNLESS their true actions overlap (false negatives).

    Args:
        h_cf:  (batch, hidden_dim) — output of CF tower
        h_sem: (batch, hidden_dim) — output of Semantic tower
        temperature: controls sharpness of distribution (lower = harder negatives)
        batch_action_ids: (batch,) LongTensor of action IDs for false-neg masking.
            If two samples share the same action, they should NOT be pushed apart.

    Returns:
        InfoNCE loss scalar
    """
    h_cf_norm = F.normalize(h_cf, dim=-1)
    h_sem_norm = F.normalize(h_sem, dim=-1)
    # Similarity matrix: (batch, batch)
    logits = torch.matmul(h_cf_norm, h_sem_norm.T) / temperature

    # ── False-Negative Mask ──
    # If two samples performed the same action (e.g., both watched Interstellar),
    # they are semantically similar and should NOT be treated as negatives.
    if batch_action_ids is not None:
        # mask[i][j] = True if action_i == action_j AND i != j
        same_action = (batch_action_ids.unsqueeze(0) == batch_action_ids.unsqueeze(1))
        same_action.fill_diagonal_(False)  # Keep the positive diagonal intact
        # Set false-negative logits to -inf so they don't contribute to denominator
        logits = logits.masked_fill(same_action, float('-inf'))

    # Labels: diagonal is the positive pair
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


# ========================== Tower Modules ==========================

class CFTower(nn.Module):
    """Collaborative Filtering Tower: Embedding + GRU for ID sequence modeling."""

    def __init__(self, num_items, embed_dim, hidden_dim):
        super(CFTower, self).__init__()
        self.embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, item_id_seq):
        """
        Args:
            item_id_seq: (batch, seq_len) LongTensor of item IDs
        Returns:
            h_cf: (batch, hidden_dim)
        """
        x = self.embedding(item_id_seq)  # (batch, seq_len, embed_dim)
        _, h = self.gru(x)               # h: (1, batch, hidden_dim)
        return h.squeeze(0)              # (batch, hidden_dim)


class SemanticTower(nn.Module):
    """Semantic Tower: Projects frozen BGE 768D vectors + GRU."""

    def __init__(self, sem_dim, hidden_dim):
        super(SemanticTower, self).__init__()
        self.proj = nn.Linear(sem_dim, hidden_dim)  # Learnable projection W_sem
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, sem_vectors):
        """
        Args:
            sem_vectors: (batch, seq_len, 768) — frozen, no grad from BGE
        Returns:
            h_sem: (batch, hidden_dim)
        """
        x = F.relu(self.proj(sem_vectors))  # (batch, seq_len, hidden_dim)
        _, h = self.gru(x)
        return h.squeeze(0)


class DualTowerNet(nn.Module):
    """Dual-Tower DQN Network.

    Outputs a 768D Virtual Action Embedding instead of n_actions discrete Q-values.
    The action embedding is used to retrieve the nearest item via FAISS.
    """

    def __init__(self, num_items, embed_dim, hidden_dim, sem_dim=768):
        super(DualTowerNet, self).__init__()
        self.cf_tower = CFTower(num_items, embed_dim, hidden_dim)
        self.sem_tower = SemanticTower(sem_dim, hidden_dim)

        # Fusion MLP: concatenated dual-tower output → 768D action embedding
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sem_dim)  # Output: 768D for FAISS retrieval
        )

    def forward(self, item_id_seq, sem_vectors):
        """
        Args:
            item_id_seq: (batch, seq_len) LongTensor
            sem_vectors: (batch, seq_len, 768) FloatTensor (detached/frozen)
        Returns:
            action_embedding: (batch, 768) — virtual action vector
            h_cf: (batch, hidden_dim) — for InfoNCE
            h_sem: (batch, hidden_dim) — for InfoNCE
        """
        h_cf = self.cf_tower(item_id_seq)
        h_sem = self.sem_tower(sem_vectors)
        fused = torch.cat([h_cf, h_sem], dim=-1)  # (batch, 2*hidden_dim)
        action_embedding = self.fusion(fused)       # (batch, 768)
        return action_embedding, h_cf, h_sem


# ======================== DQN Agent ========================

class DQN(object):
    """Dual-Tower Deep Q-Network with FAISS-based Action Retrieval.

    Key design decisions:
      1. Replay Buffer stores ONLY int item IDs — zero dense float storage.
      2. Semantic vectors are looked up on-the-fly during learn() via SemanticBridge.
      3. FAISS is the Action Retriever: network outputs 768D ideal-action vector,
         FAISS index.search() returns the Top-K nearest item IDs.
      4. All tensors explicitly transferred to self.device.
      5. BGE vectors are frozen (requires_grad=False); gradients only flow
         through W_sem projection, CF Embedding, GRUs, and Fusion MLP.
    """

    def __init__(self, n_states, n_actions, num_items,
                 memory_capacity, lr, epsilon, target_network_replace_freq,
                 batch_size, gamma, tau, K,
                 semantic_bridge, faiss_index,
                 embed_dim=64, hidden_dim=128, sem_dim=768,
                 lambda_cl=0.1, temperature=0.07, device=None):
        self.n_states = n_states  # observation window length
        self.n_actions = n_actions
        self.num_items = num_items
        self.memory_capacity = memory_capacity
        self.lr = lr
        self.epsilon = epsilon
        self.replace_freq = target_network_replace_freq
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.K = K
        self.lambda_cl = lambda_cl
        self.temperature = temperature

        # Device setup
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Semantic Bridge (NumPy lookup, NOT FAISS)
        self.semantic_bridge = semantic_bridge

        # FAISS Index (Action Retriever)
        self.faiss_index = faiss_index

        # Dual-Tower Networks
        self.eval_net = DualTowerNet(num_items, embed_dim, hidden_dim, sem_dim).to(self.device)
        self.target_net = DualTowerNet(num_items, embed_dim, hidden_dim, sem_dim).to(self.device)
        hard_update(self.target_net, self.eval_net)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

        # Experience Replay Buffer — ONLY stores int IDs, no float vectors!
        # Each transition: [s_ids (n_states ints), action_id (1 int), reward (1 float), s_next_ids (n_states ints)]
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((0, self.n_states * 2 + 2))

    def choose_action(self, state, env, I_sim_list):
        """Choose action by scoring the DSAG subset (I_sim_list).

        Args:
            state: 1D array of item IDs (length = n_states)
            env: Environment instance
            I_sim_list: list of candidate item IDs from the Hybrid DSAG

        Returns:
            rec_list: list of K recommended item IDs
        """
        state_ids = np.array(state).reshape(1, -1)  # (1, seq_len)
        sem_vectors = self.semantic_bridge.lookup_batch(state_ids)  # (1, seq_len, 768)
        state_tensor = torch.tensor(state_ids, dtype=torch.long).to(self.device)

        with torch.no_grad():
            action_emb, _, _ = self.eval_net(state_tensor, sem_vectors)  # (1, 768)
            action_emb = F.normalize(action_emb, p=2, dim=-1)

        rec_list = []
        if np.random.uniform() < self.epsilon:
            # Exploit: Score the I_sim_list items and pick the Top-K
            # Look up embeddings for the candidates
            cand_ids = np.array(I_sim_list).reshape(1, -1)
            cand_embs = self.semantic_bridge.lookup_batch(cand_ids)[0]  # (N_cand, 768)
            cand_embs = F.normalize(cand_embs, p=2, dim=-1)
            
            # Compute cosine similarities (Q-values) between ideal action and candidates
            q_values = torch.matmul(action_emb, cand_embs.T).squeeze(0)  # (N_cand,)
            
            # Sort candidates by Q-value
            sorted_indices = torch.argsort(q_values, descending=True).cpu().numpy()
            
            for idx in sorted_indices:
                item_id = int(I_sim_list[idx])
                if item_id not in env.mask_list and item_id != -1:
                    rec_list.append(item_id)
                    env.mask_list.append(item_id)
                    if len(rec_list) >= self.K:
                        break

        # Fill remaining with random exploration from valid space
        valid_space = [i for i in env.action_space if i not in env.mask_list]
        while len(rec_list) < self.K and valid_space:
            rand_idx = np.random.randint(0, len(valid_space))
            action = int(valid_space.pop(rand_idx))
            rec_list.append(action)
            env.mask_list.append(action)

        return rec_list

    def store_transition(self, s, a, r, s_):
        """Store transition in replay buffer — ONLY int IDs and scalar reward."""
        transition = np.hstack((s, [a, r], s_))
        if len(self.memory) < self.memory_capacity:
            self.memory = np.append(self.memory, [transition], axis=0)
        else:
            index = self.memory_counter % self.memory_capacity
            self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        """Train the Dual-Tower DQN with joint TD + InfoNCE loss.

        Mathematical formulation:
          Q(s, a) = cosine_sim(f_eval(s), e_a)        — scalar Q-value via dot product
          Q_target = r + γ · max_{a'} Q_target(s', a') — Bellman equation
          max_{a'} is estimated by FAISS search on target net output

          L_total = L_TD + λ_cl · L_InfoNCE
          L_TD      = MSE(Q_eval, Q_target)            — temporal difference error
          L_InfoNCE = -log(exp(sim+)/Σexp(sim))        — cross-view alignment

        Key flow:
          1. Sample int-only batch from Buffer
          2. On-the-fly semantic lookup (detached, no grad from BGE)
          3. Forward through eval and target nets → 768D action embeddings
          4. Compute scalar Q-values via dot product with item embeddings
          5. Estimate max future Q via FAISS nearest-neighbor on target output
          6. Construct Bellman TD target and compute MSE loss
          7. Build in-batch positive/negative pairs for InfoNCE
          8. Joint backprop with gradient clipping

        Returns:
            dict with 'loss_total', 'loss_td', 'loss_infonce' for logging, or None
        """
        if len(self.memory) < self.batch_size:
            return None

        # Periodically hard-copy eval → target network
        if self.learn_step_counter % self.replace_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # ── Step 1: Sample mini-batch of pure int transitions ──
        sample_index = np.random.choice(len(self.memory), self.batch_size)
        batch_memory = self.memory[sample_index, :]

        batch_state_ids = batch_memory[:, :self.n_states].astype(int)       # (B, seq_len)
        batch_action_ids = batch_memory[:, self.n_states].astype(int)        # (B,)
        batch_reward = batch_memory[:, self.n_states + 1]                    # (B,)
        batch_state_next_ids = batch_memory[:, -self.n_states:].astype(int)  # (B, seq_len)

        # ── Step 2: On-the-fly semantic lookup (frozen, requires_grad=False) ──
        sem_s = self.semantic_bridge.lookup_batch(batch_state_ids)           # (B, seq_len, 768)
        sem_s_next = self.semantic_bridge.lookup_batch(batch_state_next_ids)

        # Convert IDs to device tensors
        s_ids = torch.tensor(batch_state_ids, dtype=torch.long).to(self.device)
        s_next_ids = torch.tensor(batch_state_next_ids, dtype=torch.long).to(self.device)
        reward_t = torch.tensor(batch_reward, dtype=torch.float32).to(self.device)  # (B,)

        # ── Step 3: Forward pass — Eval Net ──
        pred_emb, h_cf, h_sem = self.eval_net(s_ids, sem_s)  # pred_emb: (B, 768)

        # ── Step 4: Compute Q(s, a) = dot(normalize(pred_emb), normalize(e_a)) ──
        # Look up the actual action's semantic vector
        action_emb = self.semantic_bridge.lookup_ids(batch_action_ids)  # (B, 768)

        # L2 normalize both for cosine similarity (dot product of unit vectors)
        pred_emb_norm = F.normalize(pred_emb, dim=-1)
        action_emb_norm = F.normalize(action_emb, dim=-1)

        # Scalar Q-value: per-sample dot product
        q_eval = (pred_emb_norm * action_emb_norm).sum(dim=-1)  # (B,)

        # ── Step 5: Estimate max future Q via Target Net + NumPy dot product ──
        # NOTE: We use NumPy matrix multiply instead of FAISS search here because
        # on macOS ARM, faiss-cpu and torch both link libomp.dylib causing segfaults.
        # FAISS is reserved for choose_action() at inference time only.
        with torch.no_grad():
            target_emb, _, _ = self.target_net(s_next_ids, sem_s_next)  # (B, 768)

            # max_{a'} cos_sim(target(s'), e_{a'}) via brute-force dot product
            target_np = target_emb.cpu().numpy().astype(np.float32)
            # L2 normalize target embeddings
            norms = np.linalg.norm(target_np, axis=1, keepdims=True) + 1e-8
            target_np = target_np / norms
            # Dot product against all normalized item embeddings: (B, N)
            all_scores = target_np @ self.semantic_bridge.get_normalized_embeddings().T
            # Max over all items
            max_q_next = torch.tensor(
                all_scores.max(axis=1), dtype=torch.float32
            ).to(self.device)  # (B,)

        # ── Step 6: Bellman TD Target and TD Loss ──
        q_target = reward_t + self.gamma * max_q_next  # (B,)
        loss_td = F.mse_loss(q_eval, q_target.detach())

        # ── Step 7: InfoNCE Contrastive Loss (with false-negative mask) ──
        # Positive pair: (h_cf[i], h_sem[i]) — same state's CF and Semantic views
        # Negative pairs: (h_cf[i], h_sem[j]) for j ≠ i AND action_i ≠ action_j
        action_ids_tensor = torch.tensor(batch_action_ids, dtype=torch.long).to(self.device)
        loss_infonce = compute_infonce_loss(h_cf, h_sem, self.temperature, action_ids_tensor)

        # ── Step 8: Joint Loss + Backprop ──
        loss_total = loss_td + self.lambda_cl * loss_infonce

        self.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Soft update target network (Polyak averaging)
        soft_update(self.target_net, self.eval_net, self.tau)

        return {
            'loss_total': loss_total.item(),
            'loss_td': loss_td.item(),
            'loss_infonce': loss_infonce.item(),
            'q_eval_mean': q_eval.mean().item(),
            'q_target_mean': q_target.mean().item()
        }
