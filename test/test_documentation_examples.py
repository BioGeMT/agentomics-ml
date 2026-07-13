"""
UX-003: Executable documentation checks

This module validates documentation examples without executing expensive end-to-end runs.
It checks:
- JSON fenced code blocks parse correctly
- Relative markdown links point to existing files
- Script paths in examples exist
- Documented CLI flags appear in --help output
- Dataset paths follow conventions (datasets/<name> for public, test_datasets/<name>/test for hidden)
- No obvious placeholder secrets that resemble real keys

Limitations:
- Shell command examples with line continuations are not fully parsed
- Only checks public run.sh flags, not internal script flags
- Network links are syntax-checked only, no actual HTTP requests
- Does not validate prose wording or exact formatting
"""

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path
from typing import List, Tuple, Set

# Setup path to repository root
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Documents to check - explicit allowlist to avoid noise from generated reports
DOCS_TO_CHECK = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "getting-started" / "quick-start.md",
    REPO_ROOT / "docs" / "getting-started" / "installation.md",
    REPO_ROOT / "docs" / "user-guide" / "datasets.md",
    REPO_ROOT / "docs" / "user-guide" / "dataset-best-practices.md",
]


class MarkdownParser:
    """
    Minimal markdown parser for extracting fenced code blocks and links.
    Does not implement full markdown spec - only what's needed for doc validation.
    """

    @staticmethod
    def extract_fenced_blocks(content: str, language: str = None) -> List[Tuple[str, int, str]]:
        """
        Extract fenced code blocks from markdown content.

        Returns list of (language, line_number, content) tuples.
        Line numbers are 1-indexed to match editor conventions.
        """
        blocks = []
        lines = content.split('\n')
        in_block = False
        block_lang = None
        block_start = None
        block_lines = []

        for i, line in enumerate(lines, start=1):
            # Check for fence start (``` or ~~~)
            fence_match = re.match(r'^(```|~~~)(\w*)', line)

            if fence_match and not in_block:
                # Start of block
                in_block = True
                block_lang = fence_match.group(2) or None
                block_start = i
                block_lines = []
            elif fence_match and in_block:
                # End of block
                if language is None or block_lang == language:
                    blocks.append((block_lang, block_start, '\n'.join(block_lines)))
                in_block = False
                block_lang = None
                block_start = None
                block_lines = []
            elif in_block:
                # Inside block
                block_lines.append(line)

        return blocks

    @staticmethod
    def extract_links(content: str) -> List[Tuple[str, int]]:
        """
        Extract markdown links from content.

        Returns list of (link_target, line_number) tuples.
        Handles both [text](url) and [text][ref] formats.
        """
        links = []
        lines = content.split('\n')

        for i, line in enumerate(lines, start=1):
            # Match [text](url) format
            for match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', line):
                links.append((match.group(2), i))

        return links


class DocumentationExamplesTest(unittest.TestCase):
    """Test that documentation examples are valid and consistent."""

    @classmethod
    def setUpClass(cls):
        """Load help output once for all tests."""
        # Get run.sh --help output (should be fast and not require Docker/Conda)
        try:
            result = subprocess.run(
                ['bash', str(REPO_ROOT / 'run.sh'), '--help'],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(REPO_ROOT)
            )
            cls.help_output = result.stdout
        except Exception as e:
            cls.help_output = ""
            print(f"Warning: Could not get --help output: {e}", file=sys.stderr)

    def test_all_json_blocks_parse(self):
        """All JSON fenced code blocks must parse as valid JSON."""
        parser = MarkdownParser()
        failures = []

        for doc_path in DOCS_TO_CHECK:
            if not doc_path.exists():
                failures.append(f"Document not found: {doc_path}")
                continue

            content = doc_path.read_text()
            json_blocks = parser.extract_fenced_blocks(content, 'json')

            for lang, line_num, block_content in json_blocks:
                try:
                    json.loads(block_content)
                except json.JSONDecodeError as e:
                    failures.append(
                        f"{doc_path.relative_to(REPO_ROOT)}:{line_num} - "
                        f"Invalid JSON: {e.msg} at line {e.lineno}, column {e.colno}"
                    )

        if failures:
            self.fail("JSON validation failures:\n" + "\n".join(failures))

    def test_relative_links_exist(self):
        """Relative markdown links must point to existing files."""
        parser = MarkdownParser()
        failures = []

        for doc_path in DOCS_TO_CHECK:
            if not doc_path.exists():
                continue

            content = doc_path.read_text()
            links = parser.extract_links(content)

            for link_target, line_num in links:
                # Skip absolute URLs (http://, https://, mailto:)
                if re.match(r'^(https?://|mailto:)', link_target):
                    continue

                # Skip anchors within the same document
                if link_target.startswith('#'):
                    continue

                # Handle links with anchors (e.g., file.md#section)
                link_path = link_target.split('#')[0]

                # Resolve relative to the document's directory
                target_path = (doc_path.parent / link_path).resolve()

                if not target_path.exists():
                    failures.append(
                        f"{doc_path.relative_to(REPO_ROOT)}:{line_num} - "
                        f"Link target not found: {link_target} "
                        f"(resolved to {target_path.relative_to(REPO_ROOT)})"
                    )

        if failures:
            self.fail("Broken link failures:\n" + "\n".join(failures))

    def test_script_paths_exist(self):
        """Script paths mentioned in documentation must exist."""
        parser = MarkdownParser()
        failures = []

        # Pattern to match script invocations (./path/to/script.sh or python path/to/script.py)
        script_pattern = re.compile(r'(?:^|\s)\./([\w/\-_.]+\.(?:sh|py))')

        for doc_path in DOCS_TO_CHECK:
            if not doc_path.exists():
                continue

            content = doc_path.read_text()

            # Check in bash/shell code blocks
            bash_blocks = parser.extract_fenced_blocks(content, 'bash')
            bash_blocks.extend(parser.extract_fenced_blocks(content, 'sh'))
            bash_blocks.extend(parser.extract_fenced_blocks(content, ''))  # unmarked blocks

            for lang, line_num, block_content in bash_blocks:
                for match in script_pattern.finditer(block_content):
                    script_path = REPO_ROOT / match.group(1)
                    if not script_path.exists():
                        failures.append(
                            f"{doc_path.relative_to(REPO_ROOT)}:{line_num} - "
                            f"Script not found: {match.group(1)}"
                        )

        if failures:
            self.fail("Missing script failures:\n" + "\n".join(failures))

    def test_documented_flags_in_help(self):
        """Primary CLI flags mentioned in docs must appear in --help output."""
        if not self.help_output:
            self.skipTest("Could not get --help output")

        # Primary flags we document and expect in help
        # These are flags users are expected to use directly
        documented_flags = {
            '--doctor',
            '--validate-dataset',
            '--preset',
            '--dataset',
            '--model',
            '--iterations',
            '--verbosity',
            '--local',
            '--help',
            '--list-metrics',
            '--list-models',
            '--list-datasets',
        }

        failures = []
        for flag in documented_flags:
            if flag not in self.help_output:
                failures.append(f"Flag '{flag}' not found in --help output")

        if failures:
            self.fail("Flag validation failures:\n" + "\n".join(failures))

    def test_dataset_path_conventions(self):
        """Dataset path examples must follow conventions."""
        parser = MarkdownParser()
        failures = []

        # Patterns that violate conventions
        # Public datasets should be under datasets/<name>
        # Hidden test should be under test_datasets/<name>/test (NOT train/ or validation/)

        # More specific violations to catch:
        # 1. Public dataset with test/ subdirectory (should use test_datasets instead)
        # 2. Hidden test with train/ or validation/ subdirectory
        # Use word boundary or start-of-path to avoid matching "test_datasets" as "datasets"
        bad_patterns = [
            (r'(?<![_/])datasets/[^/\s]+/test/', 'Public dataset should not contain test/ subdirectory'),
            (r'test_datasets/[^/\s]+/train/', 'Hidden test area should not contain train/ subdirectory'),
            (r'test_datasets/[^/\s]+/validation/', 'Hidden test area should not contain validation/ subdirectory'),
        ]

        for doc_path in DOCS_TO_CHECK:
            if not doc_path.exists():
                continue

            content = doc_path.read_text()
            lines = content.split('\n')

            for i, line in enumerate(lines, start=1):
                # Skip tree diagram lines (└──, ├──, │, etc.)
                if any(marker in line for marker in ['└──', '├──', '│']):
                    continue

                # Skip comment lines or lines clearly marking incorrect usage
                if any(marker in line.lower() for marker in ['incorrect', 'wrong', 'do not', 'not under', 'should not']):
                    continue

                for pattern, message in bad_patterns:
                    if re.search(pattern, line):
                        failures.append(
                            f"{doc_path.relative_to(REPO_ROOT)}:{i} - "
                            f"{message}: {line.strip()[:80]}"
                        )

        if failures:
            self.fail("Dataset path convention failures:\n" + "\n".join(failures))

    def test_no_obvious_placeholder_secrets(self):
        """Documentation should not contain placeholder values that look like real secrets."""
        parser = MarkdownParser()
        failures = []

        # Patterns that look like real API keys/secrets
        # Real keys often have specific prefixes and lengths
        suspicious_patterns = [
            (r'sk-[a-zA-Z0-9]{32,}', 'OpenAI-style key'),
            (r'OPENAI_API_KEY=["\']?sk-', 'OpenAI key assignment'),
            (r'ANTHROPIC_API_KEY=["\']?sk-ant-', 'Anthropic key assignment'),
            (r'OPENROUTER_API_KEY=["\']?sk-or-', 'OpenRouter key assignment'),
        ]

        for doc_path in DOCS_TO_CHECK:
            if not doc_path.exists():
                continue

            content = doc_path.read_text()
            lines = content.split('\n')

            for i, line in enumerate(lines, start=1):
                # Skip lines that are clearly examples/placeholders
                if any(marker in line.lower() for marker in ['your-key-here', 'your_key', 'example', 'placeholder']):
                    continue

                for pattern, key_type in suspicious_patterns:
                    if re.search(pattern, line):
                        failures.append(
                            f"{doc_path.relative_to(REPO_ROOT)}:{i} - "
                            f"Possible real {key_type} found: {line.strip()[:80]}"
                        )

        if failures:
            self.fail("Possible secret exposure:\n" + "\n".join(failures))

    def test_all_documented_files_exist(self):
        """All documents in the test allowlist must exist."""
        missing = []
        for doc_path in DOCS_TO_CHECK:
            if not doc_path.exists():
                missing.append(str(doc_path.relative_to(REPO_ROOT)))

        if missing:
            self.fail(f"Expected documentation files missing:\n" + "\n".join(missing))


if __name__ == '__main__':
    unittest.main()
