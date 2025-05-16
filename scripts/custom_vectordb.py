import os
import math

# Custom Vector Database implementation
class CustomVectorDatabase:
    def __init__(self, persistence_dir="./vector_db"):
        self.collections = {}
        self.persistence_dir = persistence_dir
        os.makedirs(persistence_dir, exist_ok=True)
    
    def get_or_create_collection(self, name):
        if name not in self.collections:
            self.collections[name] = CustomCollection(name)
        return self.collections[name]

class CustomCollection:
    def __init__(self, name):
        self.name = name
        self.documents = []
        self.embeddings = []
        self.ids = []
    
    def add(self, documents, ids, embeddings):
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.ids.extend(ids)
    
    def query(self, query_embeddings, n_results=3):
        results = []
        distances = []
        
        for query_embedding in query_embeddings:
            similarities = []
            for i, embedding in enumerate(self.embeddings):
                similarity = self._cosine_similarity(query_embedding, embedding)
                similarities.append((similarity, i))
            
            # Sort by similarity (highest first)
            similarities.sort(reverse=True)
            
            # Get top results
            top_indices = [similarities[i][1] for i in range(min(n_results, len(similarities)))]
            top_distances = [1 - similarities[i][0] for i in range(min(n_results, len(similarities)))]
            
            results.append([self.ids[i] for i in top_indices])
            distances.append(top_distances)
        
        return {
            "ids": results,
            "documents": [[self.documents[self.ids.index(id_)] for id_ in result] for result in results],
            "distances": distances
        }
    
    def _cosine_similarity(self, vec1, vec2):
        if not vec1 or not vec2:
            return 0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 * magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
