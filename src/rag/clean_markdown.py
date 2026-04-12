"""
Clean markdown files produced by Docling PDF parsing of academic papers.

Removes boilerplate sections (References, Acknowledgements, etc.) and
non-text placeholders (images, undecodable formulas).
"""

import re
from pathlib import Path

# Section headings to drop (along with all their content until the next heading).
# Matched case-insensitively against the heading text. The leading pattern
# tolerates decorative prefixes like "■ " and numeric prefixes like "1. ".
_HEADING_PREFIX = r"(?:■\s*)?(?:\d+\.?\s*)?"

REMOVE_SECTIONS = re.compile(
    r"^" + _HEADING_PREFIX + r"("
    r"references.*"
    r"|acknowledgements?[\s:]*"
    r"|author contributions?[\s:]*"
    r"|author information[\s:]*"
    r"|corresponding author[\s:]*"
    r"|authors?[\s:]*"
    r"|notes?[\s:]*"
    r"|access[\s:]*"
    r"|associated content[\s:]*"
    r"|one[-\s]sentence summary[\s:]*"
    r"|citation[\s:]*"
    r"|editor[\s:]*"
    r"|received[\s:]*"
    r"|accepted[\s:]*"
    r"|published[\s:]*"
    r"|copyright[\s:]*"
    r"|declaration of interests?[\s:]*"
    r"|competing interests?[\s:]*"
    r"|data availability[\s:]*"
    r"|materials and data availability[\s:]*"
    r"|funding[\s:]*"
    r"|supplementary material[\s:]*"
    r"|supplementary (figures?|tables?|files?).*"
    r"|supplemental (figures?|tables?|files?).*"
    r"|graphical abstract[\s:]*"
    r"|hhs public access[\s:]*"
    r"|#.*equal contribution.*"
    r")$",
    re.IGNORECASE,
)

# Headings that mark the real start of paper content. Everything before the
# first match is dropped by trim_preamble(). Tolerates decorative "■" and
# numeric "1." prefixes.
CONTENT_START_HEADINGS = re.compile(
    r"^" + _HEADING_PREFIX + r"("
    r"abstract"
    r"|introduction"
    r"|background"
    r"|main\s+text"
    r")\b",
    re.IGNORECASE,
)

# Lines to drop unconditionally
REMOVE_LINES = re.compile(r"^\s*<!--.*?-->\s*$")


def remove_sections(text: str) -> str:
    """
    Split the document into sections at each ## heading and drop
    any section whose title matches REMOVE_SECTIONS.
    """
    # Split on ## headings, keeping the delimiter
    parts = re.split(r"(?=^## )", text, flags=re.MULTILINE)

    kept = []
    for part in parts:
        # Extract heading text (first line, strip the ## prefix)
        first_line = part.splitlines()[0] if part.strip() else ""
        heading = re.sub(r"^#{1,6}\s*", "", first_line).strip()

        if REMOVE_SECTIONS.match(heading):
            continue
        kept.append(part)

    return "".join(kept)


def trim_preamble(text: str) -> str:
    """
    Drop everything before the first heading matching CONTENT_START_HEADINGS
    (Abstract, Introduction, Background, Main text). This removes the
    title/authors/affiliations/citation/dates block that docling emits above
    the first real section.

    If no matching heading is found, returns the text unchanged so downstream
    cleaning can still do what it can.
    """
    parts = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    for i, part in enumerate(parts):
        first_line = part.splitlines()[0] if part.strip() else ""
        heading = re.sub(r"^#{1,6}\s*", "", first_line).strip()
        if CONTENT_START_HEADINGS.match(heading):
            return "".join(parts[i:])
    return text


def remove_placeholder_lines(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not REMOVE_LINES.match(ln)]
    return "\n".join(lines)


def collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)


def clean(text: str) -> str:
    text = trim_preamble(text)
    text = remove_sections(text)
    text = remove_placeholder_lines(text)
    text = collapse_blank_lines(text)
    return text.strip()


def clean_all(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(input_dir.glob("*.md"))
    if not md_files:
        print(f"No markdown files found in {input_dir}")
        return

    print(f"Cleaning {len(md_files)} file(s)  {input_dir} → {output_dir}\n")
    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        cleaned = clean(text)

        before = text.count("\n")
        after = cleaned.count("\n")
        reduction = 100 * (1 - after / before) if before else 0

        out_path = output_dir / md_path.name
        out_path.write_text(cleaned, encoding="utf-8")
        print(f"  {md_path.name}: {before} → {after} lines  ({reduction:.0f}% reduction)")

    print("\nDone.")


if __name__ == "__main__":
    clean_all()
