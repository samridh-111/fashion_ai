import random
from utils import load_features, get_similar_tops

# Load pre-extracted features
jeans_features, tops_features = load_features()

# Pick a random jeans image for testing
jeans_image = random.choice(list(jeans_features.keys()))
print(f"👖 Selected jeans: {jeans_image}")

# Recommend matching tops
recommendations = get_similar_tops(jeans_features[jeans_image], tops_features, top_n=3)

print("\n👕 Recommended Tops:")
for top_name, score in recommendations:
    print(f"- {top_name} (similarity: {score:.3f})")
