import os
from .text_preprocessing import sent_tokenize, preprocess_text, word_tokenize
from .save_process_embeddings import save_embeddings_matrix, save_preprocessed_text

# Functions to process text and interact with the vector database
def read_file(filepath):
    """Read content from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def process_sentences(content, collection, model, stopwords, output_dir=None):
    """Process and store sentences with their embeddings."""
    # Split content into sentences
    raw_sentences = sent_tokenize(content)
    
    # Save original sentences
    if output_dir:
        with open(f"{output_dir}/original_sentences.txt", "w", encoding="utf-8") as f:
            for i, s in enumerate(raw_sentences):
                f.write(f"Sentence {i+1}: {s}\n\n")
    
    # Preprocess sentences and get tokens
    cleaned_sentences = []
    all_tokens = []
    for s in raw_sentences:
        if s.strip():
            cleaned, tokens = preprocess_text(s, stopwords, save_tokens=True)
            cleaned_sentences.append(cleaned)
            all_tokens.append(tokens)
    
    # Save preprocessed sentences
    if output_dir:
        save_preprocessed_text(cleaned_sentences, f"{output_dir}/preprocessed_sentences.txt")
    
    # Fit model to the corpus
    model.fit(cleaned_sentences)
    
    # Save model data
    if output_dir:
        model.save_model_data(f"{output_dir}/sentence_model_data.json")
    
    # Generate embeddings
    ids = [f"id{i+1}" for i in range(len(cleaned_sentences))]
    embeddings = model.encode(cleaned_sentences, batch_size=32, show_progress_bar=True)
    
    # Save embeddings matrix
    if output_dir:
        save_embeddings_matrix(embeddings, model.vocabulary, f"{output_dir}/sentence_embeddings.txt")
    
    # Store in collection
    collection.add(documents=cleaned_sentences, ids=ids, embeddings=embeddings)
    
    # Return mapping from IDs to original sentences
    return dict(zip(ids, raw_sentences))

def process_paragraphs(content, collection, model, stopwords, output_dir=None):
    """Process and store paragraphs with their embeddings."""
    # Split content into paragraphs
    raw_paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    
    # Save original paragraphs
    if output_dir:
        with open(f"{output_dir}/original_paragraphs.txt", "w", encoding="utf-8") as f:
            for i, p in enumerate(raw_paragraphs):
                f.write(f"Paragraph {i+1}: {p}\n\n")
    
    # Preprocess paragraphs and get tokens
    cleaned_paragraphs = []
    all_tokens = []
    for p in raw_paragraphs:
        cleaned, tokens = preprocess_text(p, stopwords, save_tokens=True)
        cleaned_paragraphs.append(cleaned)
        all_tokens.append(tokens)
    
    # Save preprocessed paragraphs
    if output_dir:
        save_preprocessed_text(cleaned_paragraphs, f"{output_dir}/preprocessed_paragraphs.txt")
    
    # Fit model to the corpus
    model.fit(cleaned_paragraphs)
    
    # Save model data
    if output_dir:
        model.save_model_data(f"{output_dir}/paragraph_model_data.json")
    
    # Generate embeddings
    ids = [f"id{i+1}" for i in range(len(cleaned_paragraphs))]
    embeddings = model.encode(cleaned_paragraphs, batch_size=32, show_progress_bar=True)
    
    # Save embeddings matrix
    if output_dir:
        save_embeddings_matrix(embeddings, model.vocabulary, f"{output_dir}/paragraph_embeddings.txt")
    
    # Store in collection
    collection.add(documents=cleaned_paragraphs, ids=ids, embeddings=embeddings)
    
    # Return mapping from IDs to original paragraphs
    return dict(zip(ids, raw_paragraphs))

def run_query(collection, query_text, model, stopwords, n_results=3, output_dir=None):
    """Run a query against a collection."""
    # Preprocess query
    cleaned_query, query_tokens = preprocess_text(query_text, stopwords, save_tokens=True)
    
    # Save query processing information
    if output_dir:
        with open(f"{output_dir}/query_processing.txt", "w", encoding="utf-8") as f:
            f.write(f"Original Query: {query_text}\n")
            f.write(f"Tokens: {' '.join(word_tokenize(query_text.lower()))}\n")
            f.write(f"After Stopword Removal: {' '.join([w for w in word_tokenize(query_text.lower()) if w not in stopwords])}\n")
            f.write(f"After Lemmatization: {' '.join(query_tokens)}\n")
            f.write(f"Preprocessed Query: {cleaned_query}\n")
    
    # Generate query embedding
    query_embedding = model.encode([cleaned_query])[0]
    
    # Save query embedding
    if output_dir:
        with open(f"{output_dir}/query_embedding.txt", "w", encoding="utf-8") as f:
            f.write("Query Embedding Vector:\n")
            vector_str = ' '.join([f"{val:.6f}" for val in query_embedding[:10]])
            if len(query_embedding) > 10:
                vector_str += " ... " + ' '.join([f"{val:.6f}" for val in query_embedding[-5:]])
            f.write(f"[{vector_str}]\n")
    
    # Query the collection
    return collection.query(query_embeddings=[query_embedding], n_results=n_results)