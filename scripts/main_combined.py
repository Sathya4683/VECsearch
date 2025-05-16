import re
import os
import math
import json
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Tuple

# Keep the original prompt_toolkit imports
from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window, HSplit
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

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

# Intelligent text processing functions
def sent_tokenize(text):
    """Split text into sentences using regex patterns."""
    # Match sentence endings with punctuation followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def word_tokenize(text):
    """Split text into words."""
    # Split on whitespace and remove punctuation
    words = re.findall(r'\b\w+\b', text.lower())
    return words

# Load stopwords from file
def load_stopwords(filepath="stopwords.txt"):
    """Load stopwords from a file into a set for O(1) lookup."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return set(line.strip().lower() for line in f if line.strip())
    except FileNotFoundError:
        print(f"Warning: Stopwords file '{filepath}' not found. Using empty stopwords set.")
        return set()

def rule_based_lemmatize(word):
    """
    Apply rule-based lemmatization based on common English patterns.
    This function identifies common suffixes and transforms them to base forms.
    """
    word = word.lower()
    
    # Handle irregular plurals first
    irregular_plurals = {
        "men": "man", "women": "woman", "children": "child", "people": "person",
        "mice": "mouse", "feet": "foot", "teeth": "tooth", "geese": "goose",
        "oxen": "ox", "phenomena": "phenomenon", "criteria": "criterion",
        "data": "datum", "analyses": "analysis", "diagnoses": "diagnosis",
        "theses": "thesis", "crises": "crisis", "hypotheses": "hypothesis"
    }
    
    if word in irregular_plurals:
        return irregular_plurals[word]
    
    # Handle some common irregular verbs
    irregular_verbs = {
        "am": "be", "is": "be", "are": "be", "was": "be", "were": "be", "been": "be", "being": "be",
        "has": "have", "had": "have", "having": "have",
        "does": "do", "did": "do", "doing": "do",
        "goes": "go", "went": "go", "gone": "go", "going": "go",
        "makes": "make", "made": "make", "making": "make",
        "says": "say", "said": "say", "saying": "say",
        "sees": "see", "saw": "see", "seen": "see", "seeing": "see",
        "takes": "take", "took": "take", "taken": "take", "taking": "take"
    }
    
    if word in irregular_verbs:
        return irregular_verbs[word]
    
    # Rule 1: -ing endings (for verbs)
    if word.endswith('ing'):
        # Double letter + ing: running -> run
        if len(word) > 4 and word[-4] == word[-5]:
            return word[:-4]
        # e + ing: hiking -> hike
        elif word.endswith('eing'):
            return word[:-3]
        # ing: working -> work
        else:
            stem = word[:-3]
            if len(stem) > 1:  # Make sure we have a valid stem
                return stem
    
    # Rule 2: -ed endings (for verbs)
    if word.endswith('ed'):
        # Double letter + ed: stopped -> stop
        if len(word) > 3 and word[-3] == word[-4]:
            return word[:-3]
        # e + ed: liked -> like
        elif word.endswith('eed'):
            return word[:-1]
        # ed: worked -> work
        else:
            stem = word[:-2]
            if len(stem) > 1:  # Make sure we have a valid stem
                return stem
    
    # Rule 3: -s endings (plural nouns and 3rd person verbs)
    if word.endswith('s') and not word.endswith('ss'):
        # ies: countries -> country
        if word.endswith('ies'):
            return word[:-3] + 'y'
        # es: boxes -> box
        elif word.endswith('es'):
            if word.endswith('sses'):  # glasses -> glass
                return word[:-2]
            else:
                return word[:-2]
        # s: cats -> cat
        else:
            return word[:-1]
    
    # Rule 4: -er/-est endings (comparative/superlative adjectives)
    if word.endswith('er') and len(word) > 3:
        # Double letter + er: bigger -> big
        if word[-3] == word[-4]:
            return word[:-3]
        # e + er: nicer -> nice
        elif word.endswith('ier'):
            return word[:-3] + 'y'
        else:
            return word[:-2]
    
    if word.endswith('est') and len(word) > 4:
        # Double letter + est: biggest -> big
        if word[-4] == word[-5]:
            return word[:-4]
        # e + est: nicest -> nice
        elif word.endswith('iest'):
            return word[:-4] + 'y'
        else:
            return word[:-3]
    
    # Rule 5: -ly endings (adverbs)
    if word.endswith('ly') and len(word) > 3:
        return word[:-2]
    
    # Rule 6: -ful, -ness, -ment, -ity (noun and adjective transformations)
    if word.endswith('ful') and len(word) > 4:
        return word[:-3]
    
    if word.endswith('ness') and len(word) > 5:
        stem = word[:-4]
        if stem.endswith('i'):
            return stem[:-1] + 'y'  # happiness -> happy
        return stem
    
    if word.endswith('ment') and len(word) > 5:
        return word[:-4]
    
    if word.endswith('ity') and len(word) > 4:
        return word[:-3] + 'e'  # activity -> active
    
    # Return the word unchanged if no rules apply
    return word

def preprocess_text(text, stopwords, save_tokens=False):
    """
    Preprocess text by lowercasing, removing non-alphabetic characters,
    tokenizing, removing stopwords, and applying rule-based lemmatization.
    
    If save_tokens is True, returns both the cleaned text and the tokens
    """
    # Convert to lowercase and remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize into words
    tokens = word_tokenize(text)
    
    # Remove stopwords using the provided set
    filtered_tokens = [word for word in tokens if word not in stopwords]
    
    # Apply rule-based lemmatization
    lemmatized_tokens = [rule_based_lemmatize(token) for token in filtered_tokens]
    
    # Join tokens back into a string
    cleaned_text = ' '.join(lemmatized_tokens)
    
    if save_tokens:
        return cleaned_text, lemmatized_tokens
    return cleaned_text

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

# Interactive UI using prompt_toolkit
def interactive_highlight_view(highlight_ids, id_map):
    """Create an interactive view to navigate through highlighted text."""
    index = [0]  # Use a list for mutable state

    def get_text():
        """Generate formatted text for display."""
        display = []
        for id_, text in id_map.items():
            if id_ in highlight_ids:
                if highlight_ids[index[0]] == id_:
                    display.append(('[SetCursorPosition]', ''))
                    display.append(('class:selected', f">>> {text} <<<\n\n"))
                else:
                    display.append(('class:highlight', f"{text}\n\n"))
            else:
                display.append(('', f"{text}\n\n"))
        return display

    # Set up key bindings
    kb = KeyBindings()

    @kb.add('down')
    def next_highlight(event):
        index[0] = min(index[0] + 1, len(highlight_ids) - 1)
        event.app.invalidate()

    @kb.add('up')
    def prev_highlight(event):
        index[0] = max(index[0] - 1, 0)
        event.app.invalidate()

    @kb.add('q')
    def exit_(event):
        event.app.exit()

    # Create the UI layout
    content_control = FormattedTextControl(get_text)
    root_container = HSplit([Window(content_control, always_hide_cursor=False, wrap_lines=True)])
    layout = Layout(root_container)

    # Define styles
    style = Style.from_dict({
        "highlight": "#00ff00",
        "selected": "bold underline #ff0066",
    })

    # Create and run the application
    app = Application(layout=layout, key_bindings=kb, full_screen=True, style=style)
    app.run()

# Main function
def main():
    # Get user input
    file_path = input("Enter the path to the text file (e.g., sample.txt): ").strip()
    stopwords_path = input("Enter the path to the stopwords file (or press Enter for default 'stopwords.txt'): ").strip()
    stopwords_path = stopwords_path if stopwords_path else "stopwords.txt"
    query_text = input("Enter your query: ").strip()
    n_sentences = input("Enter how many top n sentences you want to retrieve (or type 'all'): ")
    n_paragraphs = input("Enter how many top n paragraphs you want to retrieve (or type 'all'): ")
    
    # Create output directory for preprocessed files
    output_dir = "preprocessed_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Parse input
    n_sentences = 100000 if n_sentences.lower() == "all" else int(n_sentences)
    n_paragraphs = 100000 if n_paragraphs.lower() == "all" else int(n_paragraphs)

    # Load stopwords
    print(f"Loading stopwords from {stopwords_path}...")
    stopwords = load_stopwords(stopwords_path)
    print(f"Loaded {len(stopwords)} stopwords.")

    # Read content from file
    content = read_file(file_path)

    # Initialize custom vector database and models
    chroma_client = CustomVectorDatabase()
    collection_sentences = chroma_client.get_or_create_collection(name="sentences")
    collection_paragraphs = chroma_client.get_or_create_collection(name="paragraphs")

    # Create embedding models
    sentence_model = CustomEmbeddingModel()
    paragraph_model = CustomEmbeddingModel()

    # Process text
    print("Processing sentences...")
    sentence_id_map = process_sentences(content, collection_sentences, sentence_model, stopwords, output_dir)
    
    print("Processing paragraphs...")
    paragraph_id_map = process_paragraphs(content, collection_paragraphs, paragraph_model, stopwords, output_dir)

    # Run queries
    print("Running queries...")
    sentence_result = run_query(collection_sentences, query_text, sentence_model, stopwords, n_results=n_sentences, output_dir=output_dir)
    paragraph_result = run_query(collection_paragraphs, query_text, paragraph_model, stopwords, n_results=n_paragraphs, output_dir=output_dir)

    # Get top results
    top_sentence_ids = sentence_result["ids"][0]
    top_paragraph_ids = paragraph_result["ids"][0]

    # Show results summary
    print("\nTop sentence matches:")
    for i, id_ in enumerate(top_sentence_ids[:3]):
        print(f"{i+1}. {sentence_id_map[id_][:100]}...")
    
    print("\n" + "#" * 80 + "\n")
    
    print("Top paragraph matches:")
    for i, id_ in enumerate(top_paragraph_ids[:3]):
        print(f"{i+1}. {paragraph_id_map[id_][:100]}...")

    # Interactive view
    choice = input("View sentence-wise or paragraph-wise? (s/p): ").strip().lower()
    if choice == "s":
        interactive_highlight_view(top_sentence_ids, sentence_id_map)
    elif choice == "p":
        interactive_highlight_view(top_paragraph_ids, paragraph_id_map)

if __name__ == "__main__":
    main()