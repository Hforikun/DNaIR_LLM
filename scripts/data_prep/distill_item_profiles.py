import os
import pandas as pd
import requests
import argparse
import time
from tqdm import tqdm
import concurrent.futures

def distill_profile(row_data, api_key, model_name):
    idx, row = row_data
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    title = str(row.get('clean_title', 'Unknown'))
    year = str(row.get('year', 'Unknown'))
    genres = str(row.get('genres_text', 'Unknown'))
    director = str(row.get('director', 'Unknown'))
    cast = str(row.get('top_cast', 'Unknown'))
    overview = str(row.get('overview', 'No description available'))
    
    system_prompt = "You are a movie analyst. Given raw metadata, produce a concise 50-80 word item profile that captures this movie's core thematic essence, narrative style, emotional tone, and target audience. Do NOT list actors or crew. Focus on what makes this movie semantically unique."
    user_prompt = f"Title: {title} ({year}). Genres: {genres}. Director: {director}. Cast: {cast}. Synopsis: {overview}"
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3, # Low temperature for consistent profiling
        "max_tokens": 150
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            if response.status_code == 200:
                result = response.json()
                profiletext = result['choices'][0]['message']['content'].strip()
                # Clean up any potential markdown newlines
                profiletext = profiletext.replace('\n', ' ').replace('\r', '')
                return idx, profiletext
            elif response.status_code == 429: # Rate limit
                time.sleep(2 * (attempt + 1))
            else:
                time.sleep(1)
        except Exception as e:
            time.sleep(1)
            
    # Fallback to raw text if completely failed
    fallback_text = f"Title: {title}. Genres: {genres}. Synopsis: {overview}"
    return idx, fallback_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, required=True, help="DeepSeek API Key")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="DeepSeek Model Version")
    parser.add_argument("--input", type=str, default="data/processed/ml-100k/movies_metadata.csv")
    parser.add_argument("--output", type=str, default="data/processed/ml-100k/movies_profiles.csv")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for testing")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent API requests")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return
        
    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)

    print(f"Starting LLM distillation for {len(df)} movies using {args.model} via DeepSeek API...")
    
    # Store results in a dictionary maps idx -> profile to maintain order
    profiles_dict = {}
    
    # Iterate with thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(distill_profile, (idx, row), args.api_key, args.model): idx for idx, row in df.iterrows()}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            idx, profile = future.result()
            profiles_dict[idx] = profile

    # Reconstruct in original order
    ordered_profiles = [profiles_dict[idx] for idx in df.index]
    df['llm_profile'] = ordered_profiles
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved distilled profiles to {args.output}")
    
    # Print a sample
    print("\n--- Sample Distilled Profile ---")
    print(df.iloc[0]['clean_title'])
    print(df.iloc[0]['llm_profile'])

if __name__ == "__main__":
    main()
