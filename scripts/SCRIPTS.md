# SCRIPTS.md

This directory contains scripts used in main.py

## Scripts

- **custom_embedding.py**: Custom embedding model based on TF-IDF principles
- **custom_vectordb.py**: imitating ChromaDB client and collection functionality , also similarity search using cosine similarity
- **prompt_toolkit.py**: setting up the prompt_toolkit CLI controls
- **text_preprocessing.py**: perform the cleaning- lemmatization, tokenization, stopword removal
- **save_process_embeddings.py**: save the embeddings and the cleaned text in preprocessed_output directory
- **run_query.py**: compiles all the preprocessing logic and runs the query (after converting it to embeddings) to perform the retreival from the custom vectorDB

---
