from pathlib import Path
import re
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCUMENTS_FOLDER = Path(__file__).parent / "Documents"
MODEL_NAME = "llama3.2:latest"
TOP_K = 3


def load_documents():
    passages = []

    for file_path in DOCUMENTS_FOLDER.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8", errors="replace")
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)

        # Split long documents into manageable, overlapping chunks.
        paragraphs = re.split(r"\n\s*\n", text)

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            start = 0
            chunk_size = 1800
            overlap = 250

            while start < len(paragraph):
                chunk = paragraph[start:start + chunk_size].strip()

                if len(chunk) >= 40:
                    passages.append({
                        "text": chunk,
                        "source": file_path.name
                    })

                if start + chunk_size >= len(paragraph):
                    break

                start += chunk_size - overlap

    return passages


def retrieve_passages(question, passages):
    documents = [p["text"] for p in passages]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    document_vectors = vectorizer.fit_transform(documents)
    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(
        question_vector, document_vectors
    )[0]

    best_indices = similarities.argsort()[::-1][:TOP_K]

    return [
        (passages[i], float(similarities[i]))
        for i in best_indices
    ]


def generate_answer(question, retrieved):
    context_parts = []

    for i, (passage, score) in enumerate(retrieved, start=1):
        context_parts.append(
            f"Passage {i} (Source: {passage['source']}):\n"
            f"{passage['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an academic factoid question-answering assistant.

Answer the question using only the supplied passages.
Give a short, direct answer.
Do not invent facts or use outside knowledge.
If the passages do not contain the answer, say:
"The answer was not found in the retrieved passages."

Retrieved passages:
{context}

Question: {question}

Answer:
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )

    return response["message"]["content"], context


def main():
    print("Loading academic documents...")
    passages = load_documents()

    if not passages:
        print("No text passages found. Check the Documents folder.")
        return

    print(f"Loaded {len(passages)} passages.")
    print(f"Using Ollama model: {MODEL_NAME}")
    print("\nRAG Factoid Question-Answering Chatbot")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Chatbot closed.")
            break

        if not question:
            continue

        try:
            retrieved = retrieve_passages(question, passages)

            print("\n--- Retrieved Passages ---")
            for i, (passage, score) in enumerate(retrieved, start=1):
                print(f"\nPassage {i}")
                print(f"Source: {passage['source']}")
                print(f"Cosine similarity: {score:.4f}")
                print(passage["text"][:1200])

            answer, context = generate_answer(question, retrieved)

            print("\n--- Generated Answer ---")
            print(answer)
            print("\n" + "=" * 60 + "\n")

        except Exception as error:
            print(f"Error: {error}")
            print("Check that Ollama is running and the model is installed.")


if __name__ == "__main__":
    main()
