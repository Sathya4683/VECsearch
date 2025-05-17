---

# VECsearch (CS2010 Data Mining CIA-2 Project for text processing)

Lightweight vector database built from scratch with NumPy and GPU parallelism, featuring custom tokenization, lemmatization, and stop-word removal—extended with RAG for context-aware document QA and an interactive CLI using prompt-toolkit.

## Features

- Lightweight vector database with sentence- and paragraph-level indexing
- Custom tokenizer, lemmatizer, and stopword remover
- GPU parallelism using CuPy for performance (optional)
- RAG-based question answering using Google Gemini
- Interactive terminal interface with arrow-key navigation

## Setup Instructions

### Step 1: Clone the Repository

Clone the repository to your local machine:

```
git clone https://github.com/Sathya4683/VECsearch.git
cd VECsearch
```

### Step 2: Install Dependencies
```
pip install -r requirements.txt
```
This would install torch, langchain, langchain-community, langchain_google_genai, Numpy, Pandas, prompt_toolkit

### Step 3: Set Up Google Gemini API Key

1. **Obtain your Google Gemini API key** from [Google Gemini](https://aistudio.google.com/prompts/new_chat).
2. Paste your API key in .env file (refer demo.env)

### Step 4: Run the Application

To run the application, use the following command:

```
python main.py
```
### Step 5: Using the Application

1. **Enter a Query:**:
   - input the source text simple (type in "assets/sample/sample.txt" to use the sample text document provided)
   - input the source stopwords dataset (type in "assets/sample/stopwords.txt" to use the sample stopwords document provided)
   - input your search query (e.g., "Plants making food?").
   - also input the value of "k" (top K results) when prompted. Can type in "all" for all matches in descending order of the similarity score. 

2. **Choose a Retrieval Mode**:
   - Select whether to view sentence-wise, paragraph-wise, or RAG-based results when prompted.

3. **View Top Matches**:
   - The application will display top matching sentences or paragraphs based on semantic similarity to your query.

4. **Navigate the Results**:
   - Use arrow keys in the interactive CLI to highlight and scroll through results for better readability. (refer CONTROLS.md)

5. **Generate a RAG Response:**:
   - If you choose the RAG option, the app will combine the top retrieved sentences and paragraphs and send them, along with your question, to  the Google Gemini API for a concise and context-aware answer.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
