import numpy as np
import random

# ----- 1. Synthetic Semantic Dictionaries -----
# Map specific index ranges to semantic natural language traits

USER_SEMANTIC_DICT = [
    (range(0, 10), "demographic traits such as young urban professional"),
    (range(10, 20), "affinity for electronics and tech gadgets"),
    (range(20, 30), "interest in apparel and fashion trends"),
    (range(30, 40), "engagement with books and media content"),
    (range(40, 50), "price sensitivity and discount hunting behavior"),
    (range(50, 60), "consumption power for premium brands"),
    (range(60, 75), "activity levels during weekday evenings"),
    (range(75, 88), "temporal browsing habits during weekends")
]

ITEM_SEMANTIC_DICT = [
    (range(0, 6), "traits typical of high-end electronics"),
    (range(6, 11), "attributes of fast-fashion apparel"),
    (range(11, 16), "premium brand tier tier signals"),
    (range(16, 21), "positive popularity and trendiness signals"),
    (range(21, 27), "high historical conversion rate indicators")
]

# ----- 2. Semantic Binning and Translation -----
def get_semantic_intensity(value, trait_str, is_user=True):
    """Translate numerical values into coherent intensity adverbs and phrases."""
    val = float(value)
    intensity = ""
    
    if is_user:
        if val >= 0.8:
            intensity = f"a very strong {trait_str}"
        elif 0.5 <= val < 0.8:
            intensity = f"a moderate {trait_str}"
        else:
            intensity = f"an extremely low {trait_str}"
    else:
        if val >= 0.8:
            intensity = f"strong {trait_str}"
        elif 0.5 <= val < 0.8:
            intensity = f"moderate {trait_str}"
        else:
            intensity = f"weak {trait_str}"
            
    return f"{intensity} ({val:.2f})"

def find_trait_from_index(index, mapping_dict):
    """Find the semantic meaning of a specific dimension index."""
    for rng, meaning in mapping_dict:
        if index in rng:
            return meaning
    return "unknown latent feature"

def extract_top_k_semantics(vector, mapping_dict, is_user=True, top_k=3):
    """Extract and translate the Top-K salient features from a multi-dimensional vector."""
    top_indices = np.argsort(vector)[-top_k:][::-1]
    bottom_index = np.argmin(vector)
    
    seen_traits = set()
    semantic_parts = []
    
    for idx in top_indices:
        trait_str = find_trait_from_index(idx, mapping_dict)
        if trait_str not in seen_traits:
            semantic_parts.append(get_semantic_intensity(vector[idx], trait_str, is_user=is_user).replace("a very strong", "very strong").replace("a moderate", "moderate").replace("an extremely low", "extremely low"))
            seen_traits.add(trait_str)
            
    if is_user and len(vector) > top_k:
        bottom_trait = find_trait_from_index(bottom_index, mapping_dict)
        if bottom_trait not in seen_traits:
            semantic_parts.append(get_semantic_intensity(vector[bottom_index], bottom_trait, is_user=True).replace("a very strong", "very strong").replace("a moderate", "moderate").replace("an extremely low", "extremely low"))
            
    return semantic_parts

# ----- 3. Prompt Templating -----
def generate_vt_prompt(user_vector_88d, item_vector_27d):
    """Generate the structured natural language prompt for LLM."""
    
    user_semantics = extract_top_k_semantics(user_vector_88d, USER_SEMANTIC_DICT, is_user=True, top_k=2)
    item_semantics = extract_top_k_semantics(item_vector_27d, ITEM_SEMANTIC_DICT, is_user=False, top_k=2)
    
    # Format grammar
    user_desc = ", ".join(user_semantics[:-1]) + ", and " + user_semantics[-1] if len(user_semantics) > 1 else user_semantics[0]
    item_desc = " with ".join(item_semantics) if len(item_semantics) > 0 else "neutral attributes"
    
    prompt = f"""User Profile: The user's interaction vector indicates {user_desc}. Other latent behavioral traits are mostly inactive.

Item Profile: This candidate item exhibits {item_desc}.

Task: Based on these multi-dimensional interaction signals and your e-commerce domain knowledge, evaluate the intrinsic matching quality and novelty of this item for this user. Output a confidence score between 0 and 1."""

    return prompt

# ----- 4. Verification & Testing -----
def run_mock_verification():
    print("=== Virtual-Taobao Semantic Prompt Verification (Realistic Constraints) ===\n")
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    # Generate 10 pairs of mock Tensors with realistic e-commerce constraints
    for i in range(1, 11):
        # 1. Base initialization: set ambient noise very low to avoid unnatural feature collisions
        mock_user = np.random.uniform(0.00, 0.20, 88)
        mock_item = np.random.uniform(0.00, 0.20, 27)
        
        # 2. Mutually Exclusive Item Category Constraint (Dims 0-15)
        # An item can be electronics (0-5), apparel (6-10), OR premium brand (11-15), but rarely all.
        item_categories = [(0, 6), (6, 11), (11, 16)]
        chosen_item_cat = random.choice(item_categories)
        # Inject dominant category signal
        mock_item[np.random.randint(chosen_item_cat[0], chosen_item_cat[1])] = np.random.uniform(0.85, 0.99)
        
        # Item Popularity/Conversion signals (Dims 16-27 are independent of category)
        mock_item[np.random.randint(16, 21)] = np.random.uniform(0.85, 0.99) # popularity
        
        # 3. User Constraints (Dims 10-39 are Core Preferences)
        user_categories = [(10, 20), (20, 30), (30, 40)]
        chosen_user_cat = random.choice(user_categories)
        # Inject dominant preference 
        mock_user[np.random.randint(chosen_user_cat[0], chosen_user_cat[1])] = np.random.uniform(0.85, 0.99)
        
        # User Demographics, Sensitivity, Activity (Independent)
        mock_user[np.random.randint(0, 10)] = np.random.uniform(0.80, 0.99)  # Demographic
        mock_user[np.random.randint(40, 50)] = np.random.uniform(0.80, 0.99) # Price sensitivity
        mock_user[np.random.randint(60, 75)] = np.random.uniform(0.80, 0.99) # Activity
        
        # Force a lowest value for semantic contrast
        mock_user[np.random.randint(75, 88)] = 0.01

        prompt = generate_vt_prompt(mock_user, mock_item)
        print(f"--- Mock Pair {i} ---")
        print(prompt)
        print("\n")

if __name__ == "__main__":
    run_mock_verification()
