from test.utils_test import BaseAgentTest
from pathlib import Path
from src.tools.replace_tool import create_replace_tool, Edit

class TestReplaceTool(BaseAgentTest):
    """Test suite for the replace tool functionality."""

    def test_single_replacement_python(self):
        """Test replacing a single occurrence in a Python file."""
        test_file = self.config.runs_dir / self.agent_id / "test_single_replace.py"

        # Create test file using write_python_tool
        code = """def hello():
    x = 10
    return x
"""
        write_result = self.write_python_tool.function(code=code, file_path=str(test_file))
        self.assertIn("Code syntax OK", write_result)

        # Perform replacement
        result = self.replace_tool.function(
            file_path=str(test_file),
            edit=Edit(old="x = 10", new="x = 20", replace_all=False)
        )

        self.assertIn("Successfully applied 1 edit(s)", result)

        # Verify replacement using bash cat
        cat_result = self.bash_tool.function(f"cat {test_file}")
        self.assertIn("x = 20", cat_result)
        self.assertNotIn("x = 10", cat_result)

    def test_multiple_replacements_in_single_call(self):
        """Test applying multiple replacements in one tool call."""
        test_file = self.config.runs_dir / self.agent_id / "test_multiple_replace.py"

        code = """def calculate(a, b):
    result = a + b
    print(result)
    return result
"""
        self.write_python_tool.function(code=code, file_path=str(test_file))

        # Perform multiple replacements
        result = self.replace_tool.function(
            file_path=str(test_file),
            edit=[
                Edit(old="def calculate(a, b):", new="def calculate_sum(a, b):", replace_all=False),
                Edit(old="result = a + b", new="result = a + b + 1", replace_all=False)
            ]
        )

        self.assertIn("Successfully applied 2 edit(s)", result)

        # Verify replacements
        cat_result = self.bash_tool.function(f"cat {test_file}")
        self.assertIn("def calculate_sum(a, b):", cat_result)
        self.assertIn("result = a + b + 1", cat_result)

    def test_replace_all_occurrences(self):
        """Test replacing all occurrences of a string."""
        test_file = self.config.runs_dir / self.agent_id / "test_replace_all.py"

        code = """x = 5
y = x + 10
z = x * 2
print(x)
"""
        self.write_python_tool.function(code=code, file_path=str(test_file))

        # Replace all occurrences of 'x'
        result = self.replace_tool.function(
            file_path=str(test_file),
            edit=Edit(old="x", new="value", replace_all=True)
        )

        self.assertIn("Successfully applied 1 edit(s)", result)

        # Verify all replacements
        cat_result = self.bash_tool.function(f"cat {test_file}")
        self.assertIn("value", cat_result)
        # Check that standalone 'x' is replaced
        self.assertNotIn("x =", cat_result)
        self.assertNotIn("= x ", cat_result)

    def test_multiline_replacement(self):
        """Test replacing multi-line strings."""
        test_file = self.config.runs_dir / self.agent_id / "test_multiline.py"

        code = """def process():
    x = 1
    y = 2
    return x + y

def other():
    pass
"""
        self.write_python_tool.function(code=code, file_path=str(test_file))

        # Replace multi-line function body
        result = self.replace_tool.function(
            file_path=str(test_file),
            edit=Edit(
                old="""def process():
    x = 1
    y = 2
    return x + y""",
                new="""def process():
    x = 10
    y = 20
    z = x + y
    return z""",
                replace_all=False
            )
        )

        self.assertIn("Successfully applied 1 edit(s)", result)

        cat_result = self.bash_tool.function(f"cat {test_file}")
        self.assertIn("x = 10", cat_result)
        self.assertIn("z = x + y", cat_result)

    def test_text_file_replacement(self):
        """Test replacement in non-Python text files."""
        test_file = self.config.runs_dir / self.agent_id / "test_text.txt"

        # Create text file using bash
        bash_result = self.bash_tool.function(f"echo 'Hello World\nThis is a test\nHello again' > {test_file}")
        self.assertNotIn("Command failed", bash_result)

        result = self.replace_tool.function(
            file_path=str(test_file),
            edit=Edit(old="Hello", new="Goodbye", replace_all=True)
        )

        self.assertIn("Successfully applied 1 edit(s)", result)

        cat_result = self.bash_tool.function(f"cat {test_file}")
        self.assertIn("Goodbye", cat_result)
        self.assertNotIn("Hello", cat_result)

    def test_error_text_not_found(self):
        """Test error handling when text to replace is not found."""
        test_file = self.config.runs_dir / self.agent_id / "test_not_found.py"

        code = "def foo():\n    pass"
        self.write_python_tool.function(code=code, file_path=str(test_file))

        result = self.replace_tool.function(
            file_path=str(test_file),
            edit=Edit(old="DOES_NOT_EXIST", new="something", replace_all=False)
        )

        self.assertIn("Error", result)
        self.assertIn("Could not find text to replace", result)

        # Verify file is unchanged
        cat_result = self.bash_tool.function(f"cat {test_file}")
        # Remove the timing message that bash_tool appends
        file_content = '\n'.join([line for line in cat_result.split('\n') if not line.startswith('[Tool call took')])
        self.assertEqual(file_content.strip(), code.strip())

    def test_error_invalid_python_syntax(self):
        """Test that invalid Python syntax is rejected."""
        test_file = self.config.runs_dir / self.agent_id / "test_invalid_syntax.py"

        code = """def valid_function():
    return 42
"""
        self.write_python_tool.function(code=code, file_path=str(test_file))

        result = self.replace_tool.function(
            file_path=str(test_file),
            edit=Edit(old="return 42", new="return (((", replace_all=False)
        )

        self.assertIn("Error", result)
        self.assertIn("invalid Python syntax", result)

        # Verify file is unchanged
        cat_result = self.bash_tool.function(f"cat {test_file}")
        self.assertIn("return 42", cat_result)

    def test_security_path_outside_agent_directory(self):
        """Test that tool rejects paths outside agent's directory."""
        result = self.replace_tool.function(
            file_path="/etc/passwd",
            edit=Edit(old="root", new="hacked", replace_all=False)
        )

        self.assertIn("Error", result)
        self.assertIn("file_path must start with", result)

    def test_error_file_does_not_exist(self):
        """Test error when file doesn't exist."""
        nonexistent_file = self.config.runs_dir / self.agent_id / "nonexistent_file.py"

        result = self.replace_tool.function(
            file_path=str(nonexistent_file),
            edit=Edit(old="test", new="test2", replace_all=False)
        )

        self.assertIn(f"{nonexistent_file} is not valid", result)

    def test_sequential_edits_build_on_each_other(self):
        """Test that sequential edits in a list build on previous edits."""
        test_file = self.config.runs_dir / self.agent_id / "test_sequential.py"

        code = """x = 1
y = 2
"""
        self.write_python_tool.function(code=code, file_path=str(test_file))

        # First edit changes x=1 to x=5, second edit uses that result
        result = self.replace_tool.function(
            file_path=str(test_file),
            edit=[
                Edit(old="x = 1", new="x = 5", replace_all=False),
                Edit(old="x = 5", new="x = 10", replace_all=False)
            ]
        )

        self.assertIn("Successfully applied 2 edit(s)", result)

        cat_result = self.bash_tool.function(f"cat {test_file}")
        # The timing message might contain "x = 1" as part of "0.1 seconds", so check actual file lines
        file_lines = [line for line in cat_result.split('\n') if not line.startswith('[Tool call took')]
        file_content = '\n'.join(file_lines)
        self.assertIn("x = 10", file_content)
        # Check that the old values don't exist (using exact line matching to avoid "x = 1" matching "x = 10")
        self.assertNotIn("x = 1\n", file_content)
        self.assertNotIn("x = 5\n", file_content)

    def test_error_on_second_edit_doesnt_corrupt_file(self):
        """Test that if second edit fails, file is not corrupted."""
        test_file = self.config.runs_dir / self.agent_id / "test_error_no_corrupt.py"

        code = """def foo():
    x = 1
    return x
"""
        self.write_python_tool.function(code=code, file_path=str(test_file))

        # Second edit will fail because text doesn't exist
        result = self.replace_tool.function(
            file_path=str(test_file),
            edit=[
                Edit(old="x = 1", new="x = 2", replace_all=False),
                Edit(old="NONEXISTENT", new="something", replace_all=False)
            ]
        )

        self.assertIn("Error on edit 2/2", result)
        self.assertIn("Could not find text to replace", result)

        # Verify file is unchanged (not corrupted with error message)
        cat_result = self.bash_tool.function(f"cat {test_file}")
        # Remove the timing message that bash_tool appends
        file_content = '\n'.join([line for line in cat_result.split('\n') if not line.startswith('[Tool call took')])
        self.assertEqual(file_content.strip(), code.strip())

    def test_replacement_preserves_indentation(self):
        """Test that replacements preserve exact indentation."""
        test_file = self.config.runs_dir / self.agent_id / "test_indentation.py"

        code = """def outer():
    def inner():
        x = 1
        y = 2
    return inner
"""
        self.write_python_tool.function(code=code, file_path=str(test_file))

        result = self.replace_tool.function(
            file_path=str(test_file),
            edit=Edit(old="        x = 1", new="        x = 100", replace_all=False)
        )

        self.assertIn("Successfully applied 1 edit(s)", result)

        cat_result = self.bash_tool.function(f"cat {test_file}")
        self.assertIn("        x = 100", cat_result)
