from pathlib import Path
from pypdf import PdfReader

folder = Path(__file__).parent / "Documents"
pdf_path = folder / "RAG  NLP1.pdf"
txt_path = folder / "RAG1.txt"

reader = PdfReader(str(pdf_path))
pages = []

for page in reader.pages:
    text = page.extract_text()
    if text:
        pages.append(text)

full_text = "\n\n".join(pages)
txt_path.write_text(full_text, encoding="utf-8")

print("Extraction complete!")
print(f"PDF: {pdf_path.name}")
print(f"Text file: {txt_path.name}")
print(f"Characters extracted: {len(full_text)}")
