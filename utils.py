import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_features(file_path="features.npz"):
    """Load saved features from disk"""
    data = np.load(file_path, allow_pickle=True)
    jeans_features = data["jeans"].item()
    tops_features = data["tops"].item()
    return jeans_features, tops_features

def get_similar_tops(jeans_vector, tops_features, top_n=3):
    """Return top N most similar tops for a given jeans vector"""
    similarities = {}
    for top_name, top_vector in tops_features.items():
        sim = cosine_similarity([jeans_vector], [top_vector])[0][0]
        similarities[top_name] = sim
    sorted_tops = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_tops[:top_n]
