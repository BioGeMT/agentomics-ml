"""
Convert PDFs to Markdown using Docling.

Docling provides advanced PDF understanding including page layout, reading order,
table structure, and image classification — producing much cleaner output than
basic PDF-to-text converters.
"""

from docling.document_converter import DocumentConverter
from pathlib import Path

RAG_DIR = Path(__file__).parent
PDF_DIR = RAG_DIR / "raw_knowledge"
OUTPUT_DIR = RAG_DIR / "processed_knowledge"


def convert_pdf_to_markdown(pdf_path: Path, output_dir: Path) -> Path:
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    markdown = result.document.export_to_markdown()

    out_path = output_dir / f"{pdf_path.stem}.md"
    out_path.write_text(markdown, encoding="utf-8")

    print(f"  ✓ {pdf_path.name} → {out_path.name}  ({len(markdown.splitlines())} lines)")
    return out_path

def convert_all_pdfs(dataset_knowledge_dir: Path = PDF_DIR, output_dir: Path = OUTPUT_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(dataset_knowledge_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {dataset_knowledge_dir}")
        return []

    print(f"Converting {len(pdf_files)} PDF(s) with Docling → {output_dir}\n")
    converted = []
    for pdf_path in pdf_files:
        try:
            out = convert_pdf_to_markdown(pdf_path, output_dir)
            converted.append(out)
        except Exception as e:
            print(f"  ✗ {pdf_path.name}: {e}")

    print(f"\nDone. {len(converted)}/{len(pdf_files)} converted.")
    return converted


if __name__ == "__main__":
    convert_all_pdfs()
