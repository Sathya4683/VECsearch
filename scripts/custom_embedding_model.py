import math
import json
from collections import Counter

# Custom embedding model based on TF-IDF principles
class CustomEmbeddingModel:
    def __init__(self, vector_size=300):
        self.vector_size = vector_size
        self.vocabulary = {}
        self.idf = {}
        self.doc_count = 0
    
    def fit(self, documents):
        """Build vocabulary and calculate IDF values from documents."""
        # Count document frequency for each term
        doc_freq = Counter()
        self.doc_count = len(documents)
        
        # Process each document
        for doc in documents:
            # Count each term only once per document
            terms = set(doc.split())
            for term in terms:
                doc_freq[term] += 1
        
        # Create vocabulary with most frequent terms
        sorted_terms = sorted(doc_freq.items(), key=lambda x: x[1], reverse=True)
        self.vocabulary = {term: idx for idx, (term, _) in 
                          enumerate(sorted_terms[:self.vector_size])}
        
        # Calculate IDF for each term in vocabulary
        for term, freq in doc_freq.items():
            if term in self.vocabulary:
                self.idf[term] = math.log((self.doc_count + 1) / (freq + 1)) + 1
    
    def encode(self, texts, batch_size=32, show_progress_bar=False):
        """Encode texts into vectors using TF-IDF weighting."""
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            if show_progress_bar and i % 10 == 0:
                print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            batch = texts[i:i+batch_size]
            batch_embeddings = []
            
            for text in batch:
                vector = self._text_to_vector(text)
                batch_embeddings.append(vector)
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def _text_to_vector(self, text):
        """Convert a text to a TF-IDF weighted vector."""
        # Initialize vector with zeros
        vector = [0.0] * self.vector_size
        
        # Count term frequencies
        words = text.split()
        term_freq = Counter(words)
        
        # Calculate TF-IDF for each term
        for term, count in term_freq.items():
            if term in self.vocabulary:
                idx = self.vocabulary[term]
                tf = count / max(len(words), 1)  # Term frequency
                tfidf = tf * self.idf.get(term, 1.0)  # TF-IDF
                vector[idx] = tfidf
        
        # Normalize the vector
        magnitude = math.sqrt(sum(v*v for v in vector))
        if magnitude > 0:
            vector = [v/magnitude for v in vector]
        
        return vector
    
    def save_model_data(self, output_file):
        """Save model vocabulary and IDF values to a file."""
        model_data = {
            "vocabulary": self.vocabulary,
            "idf": self.idf,
            "vector_size": self.vector_size,
            "doc_count": self.doc_count
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2)
