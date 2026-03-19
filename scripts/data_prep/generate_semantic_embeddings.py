import os
import pandas as pd
import numpy as np
import argparse
import time
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser(description="Generate Sentence-BERT Embeddings for Movie Metadata")
    parser.add_argument("--input", type=str, default="data/processed/ml-100k/movies_metadata.csv", help="Input metadata CSV")
    parser.add_argument("--output", type=str, default="data/processed/ml-100k/movie_embeddings.npy", help="Output NumPy array path")
    parser.add_argument("--model", type=str, default="BAAI/bge-base-en-v1.5", help="Sentence-BERT model name")
    parser.add_argument("--use-profiles", action="store_true", help="If flag is set, reads from movies_profiles.csv and uses LLM-distilled llm_profile instead of raw string concatenation")
    args = parser.parse_args()

    # 1. Load Data
    input_file = "data/processed/ml-100k/movies_profiles.csv" if args.use_profiles else args.input
    
    if not os.path.exists(input_file):
        print(f"Error: Could not find input file at {input_file}")
        return
        
    print(f"Loading metadata from {input_file}...")
    df = pd.read_csv(input_file)
    
    # 2. Format Semantic Prompts
    print("Constructing full text profiles for embedding...")
    text_profiles = []
    movie_ids = []
    
    if args.use_profiles and 'llm_profile' in df.columns:
        print(">> Triggered LLM Profile BGE Enhancement!")
        for idx, row in df.iterrows():
            # Use distilled LLM profile
            profile_text = str(row['llm_profile']) if pd.notna(row['llm_profile']) else "Unknown Profile"
            text_profiles.append(profile_text)
            movie_ids.append(row['movie_id'])
    else:
        for idx, row in df.iterrows():
            # Handle nan values safely
            title = str(row['clean_title']) if pd.notna(row['clean_title']) else "Unknown Title"
            year = str(int(row['year'])) if pd.notna(row['year']) else "Unknown Year"
            genres = str(row['genres_text']) if pd.notna(row['genres_text']) else "Unknown Genres"
            director = str(row['director']) if pd.notna(row['director']) else "Unknown"
            cast = str(row['top_cast']) if pd.notna(row['top_cast']) else "Unknown"
            overview = str(row['overview']) if pd.notna(row['overview']) else "No description available."
            
            # Crafting a dense, highly informative prompt text
            profile_text = (
                f"Title: {title} ({year}). "
                f"Genres: {genres}. "
                f"Directed by: {director}. "
                f"Starring: {cast}. "
                f"Synopsis: {overview}"
            )
            text_profiles.append(profile_text)
            movie_ids.append(row['movie_id'])
        
    # 3. Load Model
    print(f"Loading HuggingFace Sentence-BERT model: {args.model}...")
    start_time = time.time()
    try:
        model = SentenceTransformer(args.model)
    except Exception as e:
        print(f"Failed to load model. Ensure huggingface_hub is accessible. Error: {e}")
        return
        
    print(f"Model loaded in {time.time() - start_time:.2f} seconds. Computing embeddings...")
    
    # 4. Compute Embeddings
    start_time = time.time()
    embeddings = model.encode(text_profiles, show_progress_bar=True, convert_to_numpy=True)
    print(f"Generated embeddings shape: {embeddings.shape} in {time.time() - start_time:.2f} seconds.")
    
    # 5. Save output
    # Instead of just the raw matrix, we also save a parallel ID mapping so RL agent knows which row is which movie
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # We save a dictionary containing both IDs and embeddings
    # Using np.save for fast binary storage
    output_data = {
        'movie_ids': np.array(movie_ids),
        'embeddings': embeddings
    }
    np.save(args.output, output_data)
    print(f"Successfully saved ID and Embedding arrays to {args.output}")
    
    # Quick Validation Check
    # Verify that Toy Story (ID=1) and Aladdin (ID=71, animation comedy) might be relatively close
    print("\n--- Quick Cosine Similarity Validation ---")
    try:
        idx_1 = df[df['movie_id'] == 1].index[0]  # Toy Story
        idx_71 = df[df['movie_id'] == 71].index[0]  # Lion King (id 71 in ML100k)
        
        vec1 = embeddings[idx_1]
        vec71 = embeddings[idx_71]
        
        cos_sim = np.dot(vec1, vec71) / (np.linalg.norm(vec1) * np.linalg.norm(vec71))
        print(f"Similarity between '{df.iloc[idx_1]['clean_title']}' and '{df.iloc[idx_71]['clean_title']}': {cos_sim:.4f}")
    except IndexError:
        pass # Ignored if specific IDs are missing in mock tests

if __name__ == "__main__":
    main()
