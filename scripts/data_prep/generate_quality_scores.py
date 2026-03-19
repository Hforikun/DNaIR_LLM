"""Generate LLM Zero-Shot Semantic Quality Scores (ψ̂_a).

Reads the DeepSeek-generated movie profiles and extracts a quality score
ψ̂_a ∈ [0, 1] for each movie based on sentiment analysis of the LLM's
textual assessment. This replaces the original Bayesian quality factor
that was hopelessly dependent on historical interaction counts.

Formula context (from Step 4 theory):
  R_new(s̃, ã) = sim(s̃, ã) + β · ψ̂_a · r_nov(a)
  ψ̂_a = P_LLM(High Quality | Prompt(T_a))

Output: quality_scores_llm.json  → {movie_id_str: float_score}
"""
import json
import csv
import re
import os
import sys

# ── Positive quality markers and their weights ──
# These are sentiment/quality keywords that LLMs typically use when
# describing high-quality content in their profiles.
QUALITY_MARKERS = {
    # Superlative praise (weight 3)
    'masterpiece': 3, 'masterfully': 3, 'groundbreaking': 3, 'landmark': 3,
    'pioneering': 3, 'seminal': 3, 'iconic': 3, 'timeless': 3, 'classic': 3,
    'extraordinary': 3, 'exceptional': 3, 'brilliant': 3, 'genius': 3,
    # Strong positive (weight 2)
    'compelling': 2, 'powerful': 2, 'riveting': 2, 'captivating': 2,
    'profound': 2, 'resonant': 2, 'unforgettable': 2, 'stunning': 2,
    'outstanding': 2, 'superb': 2, 'excellent': 2, 'remarkable': 2,
    'gripping': 2, 'poignant': 2, 'nuanced': 2, 'sophisticated': 2,
    'thought-provoking': 2, 'emotionally': 2, 'authentic': 2, 'innovative': 2,
    'intelligent': 2, 'bold': 2, 'ambitious': 2, 'beautifully': 2,
    'heartfelt': 2, 'visionary': 2, 'definitive': 2, 'triumphant': 2,
    # Moderate positive (weight 1)
    'entertaining': 1, 'enjoyable': 1, 'engaging': 1, 'charming': 1,
    'fun': 1, 'witty': 1, 'clever': 1, 'solid': 1, 'strong': 1,
    'effective': 1, 'appealing': 1, 'sharp': 1, 'stylish': 1, 'smart': 1,
    'creative': 1, 'fresh': 1, 'vivid': 1, 'memorable': 1, 'dynamic': 1,
    'universally': 1, 'fans': 1, 'celebrates': 1, 'rich': 1, 'layered': 1,
}

# Negative markers (subtract from score)
NEGATIVE_MARKERS = {
    'mediocre': -2, 'forgettable': -2, 'uninspired': -2, 'predictable': -1,
    'formulaic': -1, 'shallow': -1, 'derivative': -1, 'clichéd': -1,
    'lackluster': -2, 'poorly': -2, 'weak': -1, 'flat': -1, 'dull': -1,
    'generic': -1, 'uneven': -1, 'convoluted': -1, 'heavy-handed': -1,
    'problematic': -1, 'dated': -1,
}


def compute_quality_score(profile_text: str) -> float:
    """Extract a quality score ψ̂_a ∈ [0, 1] from LLM profile text.
    
    Uses weighted keyword matching on DeepSeek's language patterns.
    The LLM naturally uses more superlative language for better films.
    """
    if not profile_text or profile_text.strip() == '':
        return 0.5  # Neutral prior for missing profiles

    text_lower = profile_text.lower()
    
    raw_score = 0
    max_possible = 0  # Track theoretical max for normalization
    
    for word, weight in QUALITY_MARKERS.items():
        count = len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower))
        raw_score += count * weight
        max_possible += weight  # Each word could appear once
    
    for word, weight in NEGATIVE_MARKERS.items():
        count = len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower))
        raw_score += count * weight  # weight is already negative
    
    # Normalize to [0, 1] using sigmoid-like scaling
    # Typical profiles get 3-8 marker hits, so we center at 5
    import math
    normalized = 1.0 / (1.0 + math.exp(-(raw_score - 4) / 3.0))
    
    # Clamp to [0.05, 0.95] to avoid extreme values
    return max(0.05, min(0.95, normalized))


def main():
    input_path = 'data/processed/ml-100k/movies_profiles.csv'
    output_path = 'data/processed/ml-100k/quality_scores_llm.json'
    
    print("=" * 60)
    print("Generating LLM Zero-Shot Quality Scores (ψ̂_a)")
    print("=" * 60)
    
    quality_scores = {}
    
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            movie_id = row['movie_id']
            profile = row.get('llm_profile', '')
            score = compute_quality_score(profile)
            quality_scores[str(movie_id)] = round(score, 4)
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(quality_scores, f, indent=2)
    
    # Statistics
    scores = list(quality_scores.values())
    print(f"\n[Results]")
    print(f"  Total movies scored: {len(scores)}")
    print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")
    print(f"  Mean: {sum(scores)/len(scores):.4f}")
    print(f"  Median: {sorted(scores)[len(scores)//2]:.4f}")
    
    # Distribution
    bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    print(f"\n  Distribution:")
    for lo, hi in bins:
        count = sum(1 for s in scores if lo <= s < hi)
        bar = '█' * (count // 10)
        print(f"    [{lo:.1f}, {hi:.1f}): {count:4d}  {bar}")
    
    # Show top 5 and bottom 5
    sorted_items = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 5 highest quality movies:")
    for mid, score in sorted_items[:5]:
        print(f"    Movie {mid}: ψ̂ = {score:.4f}")
    print(f"\n  Bottom 5 lowest quality movies:")
    for mid, score in sorted_items[-5:]:
        print(f"    Movie {mid}: ψ̂ = {score:.4f}")
    
    print(f"\n✅ Saved to {output_path}")


if __name__ == "__main__":
    main()
