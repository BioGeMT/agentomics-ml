from test.utils_test import BaseAgentTest

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
            old="x = 10",
            new="x = 20",
            replace_all=False
        )

        self.assertIn("Successfully applied 1 edit(s)", result)

        # Verify replacement using bash cat
        cat_result = self.bash_tool.function(f"cat {test_file}")
        self.assertIn("x = 20", cat_result)
        self.assertNotIn("x = 10", cat_result)

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
            old="x", 
            new="value", 
            replace_all=True
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
            old="Hello", 
            new="Goodbye", 
            replace_all=True
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
            old="DOES_NOT_EXIST",
            new="something",
            replace_all=False
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
            old="return 42",
            new="return (((", 
            replace_all=False
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
            old="root", 
            new="hacked", 
            replace_all=False
        )

        self.assertIn("Error", result)
        self.assertIn("file_path must start with", result)

    def test_error_file_does_not_exist(self):
        """Test error when file doesn't exist."""
        nonexistent_file = self.config.runs_dir / self.agent_id / "nonexistent_file.py"

        result = self.replace_tool.function(
            file_path=str(nonexistent_file),
            old="test", 
            new="test2", 
            replace_all=False
        )

        self.assertIn(f"{nonexistent_file} is not valid", result)

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
            old="        x = 1", 
            new="        x = 100", 
            replace_all=False
        )

        self.assertIn("Successfully applied 1 edit(s)", result)

        cat_result = self.bash_tool.function(f"cat {test_file}")
        self.assertIn("        x = 100", cat_result)
