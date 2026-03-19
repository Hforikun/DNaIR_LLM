import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import time

def main():
    # 1. 路径定义
    embedding_path = "data/processed/ml-100k/movie_embeddings.npy"
    metadata_path = "data/processed/ml-100k/movies_metadata.csv"
    output_index_path = "data/processed/ml-100k/movie_index.faiss"

    if not os.path.exists(embedding_path):
        print(f"Error: {embedding_path} not found. Please run generate_semantic_embeddings.py first.")
        return

    # 2. 装载数据
    print(f"Loading embeddings from {embedding_path}...")
    start_time = time.time()
    data = np.load(embedding_path, allow_pickle=True).item()
    movie_ids = data['movie_ids'].astype(np.int64) # FAISS IDMap requires 64-bit strictly
    embeddings = data['embeddings'].astype(np.float32) # FAISS requires float32 strictly
    print(f"Loaded {len(embeddings)} vectors of dimension {embeddings.shape[1]} in {time.time() - start_time:.3f}s")
    
    # 3. 强制 L2 归一化 (为 InfoNCE 的点积/余弦相似度做准备)
    print("Applying L2 Normalization...")
    faiss.normalize_L2(embeddings)

    # 4. 构建 FAISS 点积索引 (IndexFlatIP) 并包裹 IndexIDMap 支持非连续 ID
    print("Building FAISS IndexFlatIP with IndexIDMap...")
    dimension = embeddings.shape[1]
    
    # Base index for inner product (since embeddings are normalized, IP == Cosine Sim)
    base_index = faiss.IndexFlatIP(dimension)
    
    # Wrap it to map custom movie IDs instead of sequential row numbers
    index = faiss.IndexIDMap(base_index)
    
    # 5. 注入向量与IDs
    index.add_with_ids(embeddings, movie_ids)
    print(f"Successfully added {index.ntotal} items to FAISS Index.")
    
    # 6. 持久化存储
    os.makedirs(os.path.dirname(output_index_path), exist_ok=True)
    faiss.write_index(index, output_index_path)
    print(f"Index successfully written to {output_index_path}")
    
    # 7. 闭环验证 (Mock RL state / query retrieval)
    print("\n--- FAISS Playground Retrieval Test ---")
    query_text = "A science fiction movie about time travel and space robots"
    print(f"Query: '{query_text}'")
    
    try:
        print("Loading BGE model to encode query...")
        model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        
        # BGE prefers the "Represent this sentence for searching relevant passages: " prefix 
        # for sparse/short queries fetching longer passages natively, though it's optional for basic semantic matching.
        query_encoded = model.encode([f"Represent this sentence for searching relevant passages: {query_text}"], convert_to_numpy=True)
        query_encoded = query_encoded.astype(np.float32)
        faiss.normalize_L2(query_encoded)  # Critical: query must also be L2 normalized!
        
        # Search the top 5
        k = 5
        start_search = time.time()
        distances, indices = index.search(query_encoded, k)
        search_time = (time.time() - start_search) * 1000  # ms
        
        # Cross reference the titles
        df_meta = pd.read_csv(metadata_path)
        
        print(f"\nTop-{k} Results Retrieval Time: {search_time:.3f} ms")
        for rank in range(k):
            m_id = indices[0][rank]
            dist = distances[0][rank]
            # Handle potential faiss -1 missing IDs
            if m_id != -1:
                match_row = df_meta[df_meta['movie_id'] == m_id]
                if not match_row.empty:
                    title = match_row.iloc[0]['clean_title']
                    genres = match_row.iloc[0]['genres_text']
                    print(f"  [{rank+1}] Movie ID {m_id:3d} (Score: {dist:.4f}) | {title} | {genres}")
                else:
                    print(f"  [{rank+1}] Movie ID {m_id:3d} (Score: {dist:.4f}) | Title not found in CSV")
                    
    except Exception as e:
        print(f"Test query failed: {e}")

if __name__ == "__main__":
    main()
