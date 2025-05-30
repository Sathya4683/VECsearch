import os
#import custom embedding model and vector DB
from scripts.custom_embedding_model import CustomEmbeddingModel
from scripts.custom_vectordb import CustomVectorDatabase, CustomCollection
#import functions from text_preprocessing and run_query
from scripts.text_preprocessing import load_stopwords
from scripts.run_query import process_paragraphs, process_sentences, read_file, run_query
#import the functions necessary for the interactive prompt toolkit view
from scripts.prompt_toolkit import interactive_highlight_view

from rag import generate_response

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

    # Initialize custom vector database and models (similar to how one would intialize a chromdb client and create a collection)
    custom_client = CustomVectorDatabase()
    collection_sentences = custom_client.get_or_create_collection(name="sentences")
    collection_paragraphs = custom_client.get_or_create_collection(name="paragraphs")

    # Create embedding models
    sentence_model = CustomEmbeddingModel()
    paragraph_model = CustomEmbeddingModel()

    # Process text
    print("Processing sentences...")
    sentence_id_map = process_sentences(content, collection_sentences, sentence_model, stopwords, output_dir)
    
    print("Processing paragraphs...")
    paragraph_id_map = process_paragraphs(content, collection_paragraphs, paragraph_model, stopwords, output_dir)

    print("Running queries...")
    # Run queries
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
    while True:
        choice = input("View sentence-wise or paragraph-wise or RAG functionality? (s/p/RAG, type 'exit' to quit): ").strip().lower()
        if choice == "exit":
            break
        elif choice == "s":
            interactive_highlight_view(top_sentence_ids, sentence_id_map)
        elif choice == "p":
            interactive_highlight_view(top_paragraph_ids, paragraph_id_map)
        elif choice == "rag":
            # Concatenate top 2 sentences and top 2 paragraphs as context
            context_sentences = "\n".join([sentence_id_map[id_] for id_ in top_sentence_ids[:2]])
            context_paragraphs = "\n".join([paragraph_id_map[id_] for id_ in top_paragraph_ids[:2]])
            combined_context = context_sentences + "\n\n" + context_paragraphs

            user_query = input("Enter your question for RAG (with relevance to the intial query provided): ")
            answer = generate_response(user_query, combined_context)
            print("\nResponse from RAG:\n")
            print(answer)
        else:
            print("Invalid choice. Please enter 's', 'p', or 'exit'.")


if __name__ == "__main__":
    main()