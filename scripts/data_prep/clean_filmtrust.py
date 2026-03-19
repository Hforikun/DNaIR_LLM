import pandas as pd
import os

def clean_filmtrust(raw_dir, processed_dir):
    print("Cleaning FilmTrust datasets...")
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Clean Ratings
    # format: user_id item_id rating (space separated)
    r_cols = ['user_id', 'item_id', 'rating']
    ratings = pd.read_csv(os.path.join(raw_dir, 'ratings.txt'), sep='\s+', names=r_cols)
    ratings.to_csv(os.path.join(processed_dir, 'ratings.csv'), index=False)
    print(f"Saved ratings.csv with shape {ratings.shape}")
    
    # 2. Clean Trust network (if it exists)
    trust_path = os.path.join(raw_dir, 'trust.txt')
    if os.path.exists(trust_path):
        t_cols = ['truster_id', 'trustee_id', 'trust_value']
        trust = pd.read_csv(trust_path, sep='\s+', names=t_cols)
        trust.to_csv(os.path.join(processed_dir, 'trust.csv'), index=False)
        print(f"Saved trust.csv with shape {trust.shape}")
    else:
        print("Note: trust.txt not found, skipping.")

if __name__ == "__main__":
    base_dir = "/Users/kerwin/Desktop/RS/DNaIR-LLM"
    clean_filmtrust(os.path.join(base_dir, "data/raw/filmtrust"), os.path.join(base_dir, "data/processed/filmtrust"))
