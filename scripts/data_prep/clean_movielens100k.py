import pandas as pd
import os

def clean_movielens100k(raw_dir, processed_dir):
    print("Cleaning MovieLens 100K datasets...")
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Clean Users
    # format: user_id | age | gender | occupation | zip_code
    u_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    users = pd.read_csv(os.path.join(raw_dir, 'u.user'), sep='|', names=u_cols, encoding='latin-1')
    users.to_csv(os.path.join(processed_dir, 'users.csv'), index=False)
    print(f"Saved users.csv with shape {users.shape}")
    
    # 2. Clean Items
    # format: movie_id | movie_title | release_date | video_release_date | IMDb_URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western
    i_cols = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 
              'unknown', 'Action', 'Adventure', 'Animation', "Childrens", 'Comedy', 'Crime', 
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
              'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = pd.read_csv(os.path.join(raw_dir, 'u.item'), sep='|', names=i_cols, encoding='latin-1')
    
    # Optional: Fill empty titles or replace with something generic, though normally it's clean
    items['movie_title'] = items['movie_title'].fillna('')
    # Will be useful later for LLM prompt generation
    items.to_csv(os.path.join(processed_dir, 'items.csv'), index=False)
    print(f"Saved items.csv with shape {items.shape}")
    
    # 3. Clean Ratings
    # format: user_id | item_id | rating | timestamp
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(os.path.join(raw_dir, 'u.data'), sep='\t', names=r_cols, encoding='latin-1')
    ratings.to_csv(os.path.join(processed_dir, 'ratings.csv'), index=False)
    print(f"Saved ratings.csv with shape {ratings.shape}")

if __name__ == "__main__":
    base_dir = "/Users/kerwin/Desktop/RS/DNaIR-LLM"
    clean_movielens100k(os.path.join(base_dir, "data/raw/ml-100k"), os.path.join(base_dir, "data/processed/ml-100k"))
