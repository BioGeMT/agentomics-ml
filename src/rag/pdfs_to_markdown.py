from markitdown import MarkItDown
from pathlib import Path
import json

def convert_pdf_to_markdown(pdf_path, output_dir):
    """
    Convert a single PDF to Markdown using MarkItDown.
    
    Args:
        pdf_path: Path to input PDF
        output_dir: Directory to save markdown file
    
    Returns:
        Path to generated markdown file
    """
    md = MarkItDown()
    
    # Convert PDF
    result = md.convert(pdf_path)
    
    # Save markdown
    pdf_name = Path(pdf_path).stem
    md_path = Path(output_dir) / f"{pdf_name}.md"
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(result.text_content)
    
    print(f"✓ Converted: {pdf_path} → {md_path}")
    
    return md_path

def convert_all_pdfs(pdf_dir, output_dir):
    """
    Convert all PDFs in a directory to Markdown.
    
    Args:
        pdf_dir: Directory containing PDF files
        output_dir: Directory to save markdown files
    """
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠ No PDF files found in {pdf_dir}")
        return []
    
    print(f"Found {len(pdf_files)} PDF files")
    
    converted_files = []
    for pdf_path in pdf_files:
        try:
            md_path = convert_pdf_to_markdown(pdf_path, output_dir)
            converted_files.append(md_path)
        except Exception as e:
            print(f"✗ Error converting {pdf_path}: {e}")
    
    print(f"\n✓ Successfully converted {len(converted_files)}/{len(pdf_files)} files")
    
    return converted_files

if __name__ == "__main__":
    # Example usage
    PDF_DIR = "/SCRATCH/ablation/agentomics-ml/src/rag/raw_knowledge/"
    MARKDOWN_DIR = "/SCRATCH/ablation/agentomics-ml/src/rag/processed_knowledge/"
    
    convert_all_pdfs(PDF_DIR, MARKDOWN_DIR)