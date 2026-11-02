from pathlib import Path
from pypdf import PdfReader

documents_folder = Path(__file__).parent / "Documents"

pdf_files = {
    "RAG 11.pdf": "RAG11.txt",
    "RAG NLP1.pdf": "RAG1.txt"
}

for pdf_name, txt_name in pdf_files.items():
    pdf_path = documents_folder / pdf_name
    txt_path = documents_folder / txt_name

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_name}")
        continue

    reader = PdfReader(str(pdf_path))
    extracted_text = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            extracted_text.append(page_text)

    full_text = "\n\n".join(extracted_text)
    txt_path.write_text(full_text, encoding="utf-8")

    print(f"\nExtracted: {pdf_name}")
    print(f"Saved to: {txt_name}")
    print(f"Characters extracted: {len(full_text)}")

print("\nFinished!")
