import numpy as np

# Functions to save preprocessed text and embeddings
def save_preprocessed_text(preprocessed_items, output_file):
    """Save preprocessed text to a file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, text in enumerate(preprocessed_items):
            f.write(f"Item {i+1}:\n{text}\n\n")
    print(f"Preprocessed text saved to {output_file}")

def save_embeddings_matrix(embeddings, vocabulary, output_file):
    """Save embeddings matrix to a file with vocabulary mapping."""
    if not embeddings:
        print("No embeddings to save.")
        return
        
    # Convert embeddings to numpy array for better formatting
    embedding_matrix = np.array(embeddings)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header with dimensions
        f.write(f"# Embedding Matrix: {embedding_matrix.shape[0]} items × {embedding_matrix.shape[1]} dimensions\n\n")
        
        # Write vocabulary index mapping
        f.write("# Vocabulary Index Mapping:\n")
        sorted_vocab = sorted(vocabulary.items(), key=lambda x: x[1])
        for term, idx in sorted_vocab:
            f.write(f"# {idx}: {term}\n")
        f.write("\n")
        
        # Write the actual matrix
        f.write("# Embedding Matrix (rows=items, columns=dimensions):\n")
        for i, embedding in enumerate(embedding_matrix):
            # Format the vector nicely
            vector_str = ' '.join([f"{val:.6f}" for val in embedding[:10]])
            if len(embedding) > 10:
                vector_str += " ... " + ' '.join([f"{val:.6f}" for val in embedding[-5:]])
            f.write(f"Item {i+1}: [{vector_str}]\n")
    
    print(f"Embeddings matrix saved to {output_file}")
