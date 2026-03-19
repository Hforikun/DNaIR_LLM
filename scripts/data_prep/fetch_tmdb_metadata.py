import os
import pandas as pd
import requests
import time
import re
import argparse
from tqdm import tqdm
import concurrent.futures

# ----- TMDB API Helper functions -----
def search_movie_tmdb(title, year, api_key):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        'api_key': api_key,
        'query': title,
        'year': year,
        'language': 'en-US'
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 429: # Rate limit
            time.sleep(1)
            response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('results'):
            return data['results'][0]['id']
    except Exception as e:
        pass
    return None

def get_movie_details_tmdb(tmdb_id, api_key):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    params = {
        'api_key': api_key,
        'language': 'en-US',
        'append_to_response': 'credits'
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 429:
            time.sleep(1)
            response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        pass
    return None

def get_mock_movie_details(title, year):
    return {
        'overview': f"A movie titled {title} released in {year}.",
        'genres': [{'name': 'Unknown'}],
        'credits': {
            'cast': [{'name': "Unknown"}],
            'crew': [{'job': 'Director', 'name': "Unknown"}]
        }
    }

def parse_title_and_year(raw_title):
    match = re.search(r'^(.*?)\s*\((\d{4})\)\s*$', str(raw_title))
    if match:
        title = match.group(1).strip()
        year = match.group(2)
        if title.endswith(", The"): title = "The " + title[:-5]
        elif title.endswith(", A"): title = "A " + title[:-3]
        elif title.endswith(", An"): title = "An " + title[:-4]
        elif title.endswith(", L'"): title = "L'" + title[:-4]
        elif title.endswith(", Il"): title = "Il " + title[:-4]
        elif title.endswith(", La"): title = "La " + title[:-4]
        return title, year
    return raw_title, None

def process_movie(row_tuple, api_key):
    idx, row = row_tuple
    movie_id = row['movie_id']
    raw_title = row['movie_title']
    
    search_title, search_year = parse_title_and_year(raw_title)
    
    overview = ""
    genres_str = ""
    director_list = []
    cast_list = []
    
    if api_key:
        tmdb_id = search_movie_tmdb(search_title, search_year, api_key)
        if tmdb_id:
            details = get_movie_details_tmdb(tmdb_id, api_key)
            if details:
                overview = details.get('overview', '')
                genres_str = ", ".join([g['name'] for g in details.get('genres', [])])
                credits = details.get('credits', {})
                cast_list = [c['name'] for c in credits.get('cast', [])[:3]]
                director_list = [c['name'] for c in credits.get('crew', []) if c.get('job') == 'Director']
                
        # Be slightly polite; TMDB is ~40/10s => effectively 4 per second.
        # With 5 workers we might hit 429 occasionally but requests logic handles it.
    
    # Fallbacks
    if not overview.strip():
        overview = f"A movie titled {search_title} released in {search_year if search_year else 'an unknown year'}."
    if not director_list: director_list = ["Unknown Director"]
    if not cast_list: cast_list = ["Unknown Cast"]
    if not genres_str: genres_str = "Unknown"

    return {
        'movie_id': movie_id,
        'original_title': raw_title,
        'clean_title': search_title,
        'year': search_year,
        'genres_text': genres_str,
        'director': ", ".join(director_list),
        'top_cast': ", ".join(cast_list),
        'overview': overview.replace('\n', ' ').replace('\r', '')
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--input", type=str, default="data/processed/ml-100k/items.csv")
    parser.add_argument("--output", type=str, default="data/processed/ml-100k/movies_metadata.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return
        
    df_items = pd.read_csv(args.input)
    if args.limit: df_items = df_items.head(args.limit)

    print(f"Parallel processing {len(df_items)} movies...")
    
    metadata_list = []
    
    # Map with multithreading
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # submit all tasks
        futures = {executor.submit(process_movie, row_tuple, args.api_key): row_tuple for row_tuple in df_items.iterrows()}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            metadata_list.append(future.result())

    # Sort array by movie_id to preserve original order
    metadata_list = sorted(metadata_list, key=lambda x: x['movie_id'])
    
    df_meta = pd.DataFrame(metadata_list)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_meta.to_csv(args.output, index=False)
    print(f"\nSaved metadata with shape {df_meta.shape} to {args.output}")

if __name__ == "__main__":
    main()
