# RAG-Based Factoid Question-Answering Chatbot Using Ollama

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) based factoid question-answering chatbot using Python. It retrieves relevant information from a collection of text documents using TF-IDF vectorization and cosine similarity. The retrieved passages are provided as context to the Llama 3.2 model through Ollama to generate relevant answers.

## Technologies Used

- Python
- TF-IDF Vectorization
- Cosine Similarity
- Scikit-learn
- Ollama
- Llama 3.2
- Retrieval-Augmented Generation (RAG)

## RAG Pipeline

The chatbot follows these steps:

1. **Document Preparation:** Loads, cleans and divides text documents into smaller overlapping chunks.
2. **Information Retrieval:** Uses TF-IDF to convert passages into numerical representations.
3. **Similarity Calculation:** Uses cosine similarity to identify relevant passages.
4. **Context Retrieval:** Selects the top three relevant passages.
5. **LLM Integration:** Sends the retrieved context and question to Llama 3.2 through Ollama.
6. **Answer Generation:** Generates a short answer based on the retrieved information.

## Key Features

- Multiple text document support.
- Automatic document cleaning and chunking.
- TF-IDF-based information retrieval.
- Top-three passage retrieval.
- Cosine similarity score display.
- Source document identification.
- Local LLM integration using Ollama.
- Context-based answer generation.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone Y[OUR_GITHUB_REPOSITORY_UR](https://github.com/vibhaedward17/RAG-Factoid-Question-Answering-Chatbot.git)L
cd RAG-Factoid-Question-Answering-Chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama and Download the Model

Download Ollama from https://ollama.com/

Run:

```bash
ollama pull llama3.2
```

### 5. Run the Chatbot

```bash
python rag_chatbot.py
```

Enter questions in the terminal. Type `exit` to close the chatbot.

## Testing and Demonstration

The chatbot was tested using multiple factoid questions related to the prepared knowledge base. The output displays the retrieved passages, source filenames, cosine similarity scores and generated answers.

## Limitations

- Answer quality depends on the available knowledge base.
- TF-IDF relies on word-based similarity.
- Ollama and the Llama 3.2 model must be installed locally.

## Conclusion

This project demonstrates a basic RAG pipeline combining information retrieval and a locally running Large Language Model. It provides practical experience in document preparation, TF-IDF retrieval, similarity measurement and context-based answer generation.

## Author

Developed as an academic project focused on Retrieval-Augmented Generation, Natural Language Processing and Large Language Models.
