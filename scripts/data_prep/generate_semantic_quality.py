"""Generate LLM Zero-Shot Semantic Quality Scores (ψ̂_a) via DeepSeek API.

For each of the 1682 movies in ML-100K, this script sends the LLM-generated
profile text to DeepSeek-Chat and asks it to produce a single quality score
ψ̂_a ∈ [0.0, 1.0]. The score measures intrinsic content quality independent
of popularity — this is the core innovation that breaks the Bayesian prior
deadlock for cold-start items.

Formula context (from Step 4 theory):
  R_new(s̃, ã) = r_rel(s̃, ã) + β · ψ̂_a · r_nov(a)
  ψ̂_a = P_LLM(High Quality | Profile(a))

Output: quality_scores_llm.npy → dict {'movie_ids': array, 'scores': array}
        quality_scores_llm.json → {movie_id_str: float_score} (for Env loading)

Usage:
  python scripts/data_prep/generate_semantic_quality.py
"""
import os
import csv
import json
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ── Configuration ──
DEEPSEEK_API_KEY = "sk-2da12a9bce624391bd5e78eaa1ddbaa2"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
MAX_WORKERS = 8
INPUT_CSV = "data/processed/ml-100k/movies_profiles.csv"
OUTPUT_JSON = "data/processed/ml-100k/quality_scores_llm.json"
OUTPUT_NPY = "data/processed/ml-100k/quality_scores_llm.npy"

# ── Prompt Template ──
QUALITY_PROMPT = """You are a film quality evaluator. Based ONLY on the following movie profile, 
rate the intrinsic artistic and entertainment quality of this film on a scale from 0.0 to 1.0.

Scoring guidelines:
- 0.0-0.2: Very poor quality (incoherent plot, no artistic merit)
- 0.2-0.4: Below average (generic, derivative, poorly executed)
- 0.4-0.6: Average (competent but unremarkable)
- 0.6-0.8: Good to excellent (well-crafted, engaging, memorable)
- 0.8-1.0: Masterpiece (groundbreaking, deeply resonant, timeless)

IMPORTANT: Judge quality by content merit, NOT by popularity or box office.
A niche art film can score 0.9 if it's brilliantly made.

Movie Profile:
{profile}

Respond with ONLY a single decimal number between 0.0 and 1.0. Nothing else."""


def query_deepseek(client, movie_id, title, profile):
    """Send one movie profile to DeepSeek and get a quality score."""
    if not profile or profile.strip() == '':
        return movie_id, 0.5  # Neutral prior for missing profiles

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a precise film quality scorer. Output only a number."},
                {"role": "user", "content": QUALITY_PROMPT.format(profile=profile)}
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=10
        )
        raw = response.choices[0].message.content.strip()
        # Extract the float from the response
        score = float(raw)
        score = max(0.05, min(0.95, score))  # Clamp to avoid extremes
        return movie_id, score
    except (ValueError, TypeError):
        # If LLM returned non-numeric, try to find a number in the response
        import re
        match = re.search(r'(\d+\.?\d*)', raw)
        if match:
            score = float(match.group(1))
            score = max(0.05, min(0.95, score))
            return movie_id, score
        return movie_id, 0.5  # Fallback
    except Exception as e:
        print(f"  ⚠️ Movie {movie_id} ({title}): API error — {e}")
        return movie_id, 0.5


def main():
    print("=" * 60)
    print("Generating LLM Zero-Shot Quality Scores (ψ̂_a)")
    print(f"Model: {MODEL} | Workers: {MAX_WORKERS}")
    print("=" * 60)

    # Load existing profiles
    movies = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            movies.append({
                'movie_id': row['movie_id'],
                'title': row.get('clean_title', row.get('original_title', '')),
                'profile': row.get('llm_profile', '')
            })
    print(f"\n[1] Loaded {len(movies)} movie profiles")

    # Check if partial results exist (resume support)
    partial_scores = {}
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, 'r') as f:
            partial_scores = json.load(f)
        print(f"[2] Found {len(partial_scores)} existing scores (resuming)")

    # Filter movies that still need scoring
    movies_to_score = [m for m in movies if str(m['movie_id']) not in partial_scores]
    print(f"[3] Remaining to score: {len(movies_to_score)}")

    if len(movies_to_score) == 0:
        print("    All movies already scored! Skipping API calls.")
    else:
        # Initialize DeepSeek client
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

        # Multi-threaded API calls
        scored = 0
        errors = 0
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for movie in movies_to_score:
                future = executor.submit(
                    query_deepseek, client,
                    movie['movie_id'], movie['title'], movie['profile']
                )
                futures[future] = movie['movie_id']

            for future in as_completed(futures):
                movie_id, score = future.result()
                partial_scores[str(movie_id)] = round(score, 4)
                scored += 1

                if scored % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = scored / elapsed
                    remaining = (len(movies_to_score) - scored) / max(rate, 0.01)
                    print(f"    Scored {scored}/{len(movies_to_score)} "
                          f"({rate:.1f}/sec, ~{remaining:.0f}s remaining)")

                    # Save checkpoint
                    with open(OUTPUT_JSON, 'w') as f:
                        json.dump(partial_scores, f, indent=2)

        elapsed = time.time() - start_time
        print(f"\n[4] Scoring complete: {scored} movies in {elapsed:.1f}s "
              f"({scored/elapsed:.1f}/sec)")

    # Save final results
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(partial_scores, f, indent=2)

    # Also save as .npy for fast loading
    movie_ids = np.array([int(k) for k in partial_scores.keys()])
    scores = np.array([partial_scores[str(mid)] for mid in movie_ids])
    np.save(OUTPUT_NPY, {'movie_ids': movie_ids, 'scores': scores})

    # Statistics
    vals = list(partial_scores.values())
    print(f"\n[Results]")
    print(f"  Total movies scored: {len(vals)}")
    print(f"  Score range: [{min(vals):.4f}, {max(vals):.4f}]")
    print(f"  Mean: {sum(vals)/len(vals):.4f}")
    print(f"  Median: {sorted(vals)[len(vals)//2]:.4f}")

    # Distribution
    bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    print(f"\n  Distribution:")
    for lo, hi in bins:
        count = sum(1 for s in vals if lo <= s < hi)
        pct = count / len(vals) * 100
        bar = '█' * int(pct / 2)
        print(f"    [{lo:.1f}, {hi:.1f}): {count:4d} ({pct:4.1f}%)  {bar}")

    # Show examples
    sorted_items = sorted(partial_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 5 highest quality:")
    for mid, score in sorted_items[:5]:
        title = next((m['title'] for m in movies if str(m['movie_id']) == str(mid)), '?')
        print(f"    Movie {mid} ({title}): ψ̂ = {score:.4f}")
    print(f"\n  Bottom 5 lowest quality:")
    for mid, score in sorted_items[-5:]:
        title = next((m['title'] for m in movies if str(m['movie_id']) == str(mid)), '?')
        print(f"    Movie {mid} ({title}): ψ̂ = {score:.4f}")

    print(f"\n✅ Saved to {OUTPUT_JSON} and {OUTPUT_NPY}")


if __name__ == "__main__":
    main()
