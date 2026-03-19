import numpy as np
import torch
import torch.nn as nn


class SemanticBridge:
    """GPU-resident frozen embedding bridge: item_id → 768D BGE vector.

    Design principles:
      1. NO FAISS here — FAISS is the Action Retriever, not a dictionary.
      2. Embeddings stored as nn.Embedding.from_pretrained(freeze=True) ON GPU.
         This eliminates ALL CPU→GPU data transfers during learn().
         MovieLens 100K: 1682 × 768 × 4 bytes = ~5.1MB — trivial for any GPU.
      3. lookup_batch() uses pure GPU indexing (self.frozen_emb(ids_tensor)).
      4. requires_grad=False enforced by freeze=True.
      5. CPU-side normalized matrix retained ONLY for DSAG hybrid similarity scan.
    """

    def __init__(self, npy_path, device):
        """
        Args:
            npy_path: Path to movie_embeddings.npy (dict with 'movie_ids' and 'embeddings')
            device: torch.device for GPU-resident embedding (cpu / cuda / mps)
        """
        data = np.load(npy_path, allow_pickle=True).item()
        movie_ids = data['movie_ids']
        embeddings = data['embeddings'].astype(np.float32)  # (N, 768)
        self.sem_dim = embeddings.shape[1]
        self.device = device

        # Build id → row index mapping for O(1) lookup
        self.id_to_idx = {int(mid): i for i, mid in enumerate(movie_ids)}
        self.n_items = len(movie_ids)

        # ── GPU-resident frozen embedding layer ──
        # Row 0 is reserved as a zero-padding vector for unknown IDs
        padded = np.vstack([np.zeros((1, self.sem_dim), dtype=np.float32), embeddings])
        self.frozen_emb = nn.Embedding.from_pretrained(
            torch.tensor(padded, dtype=torch.float32),
            freeze=True,       # requires_grad=False
            padding_idx=0      # Row 0 = zero vector for unknown IDs
        ).to(device)

        # Shift id_to_idx values by +1 to account for padding row
        self._id_to_row = {mid: idx + 1 for mid, idx in self.id_to_idx.items()}

        # ── CPU-side normalized matrix for DSAG hybrid similarity ──
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        self._normalized_embeddings = (embeddings / norms).astype(np.float32)

    def get_normalized_embeddings(self):
        """Return the full L2-normalized embedding matrix (N, 768) on CPU.
        Used ONLY by DSAG recommend_offpolicy() for hybrid similarity scan."""
        return self._normalized_embeddings

    def _ids_to_rows(self, item_ids):
        """Convert item IDs to row indices in the padded embedding table.
        Unknown IDs map to row 0 (zero vector)."""
        if isinstance(item_ids, np.ndarray):
            rows = np.vectorize(lambda x: self._id_to_row.get(int(x), 0))(item_ids)
            return torch.tensor(rows, dtype=torch.long).to(self.device)
        elif isinstance(item_ids, (list, tuple)):
            rows = [self._id_to_row.get(int(x), 0) for x in item_ids]
            return torch.tensor(rows, dtype=torch.long).to(self.device)
        else:
            row = self._id_to_row.get(int(item_ids), 0)
            return torch.tensor([row], dtype=torch.long).to(self.device)

    def lookup_single(self, item_id):
        """Look up a single item_id → (768,) NumPy array (CPU, for compatibility)."""
        idx = self.id_to_idx.get(int(item_id), None)
        if idx is not None:
            return self._normalized_embeddings[idx] * (
                np.linalg.norm(self._normalized_embeddings[idx]) + 1e-8
            )  # Un-normalize for raw embedding
        return np.zeros(self.sem_dim, dtype=np.float32)

    def lookup_batch(self, item_id_sequences):
        """Batch lookup: (batch, seq_len) int IDs → (batch, seq_len, 768) Tensor.

        PURE GPU operation — no CPU-GPU data transfer!
        Args:
            item_id_sequences: 2D array/list of item IDs, shape (batch, seq_len)
        Returns:
            torch.Tensor on self.device, dtype=float32, requires_grad=False
        """
        row_indices = self._ids_to_rows(item_id_sequences)  # (batch, seq_len) LongTensor on GPU
        return self.frozen_emb(row_indices)  # (batch, seq_len, 768) — pure GPU indexing

    def lookup_ids(self, item_ids):
        """Flat lookup: (batch,) int IDs → (batch, 768) Tensor.
        Used for action target vectors in the loss function.
        PURE GPU operation.
        """
        row_indices = self._ids_to_rows(item_ids)  # (batch,) LongTensor on GPU
        return self.frozen_emb(row_indices)  # (batch, 768)
