from pathlib import Path

from tests.helpers import write_step_file


def test_replace_tool_updates_only_first_occurrence(
    default_agent_config,
    replace_tool,
):
    content = "x = 10\ny = x\nx = 10\n"
    test_file = write_step_file(
        default_agent_config,
        "single_replace.py",
        content,
    )

    replace_tool.function(
        file_path=str(test_file),
        old="x = 10",
        new="x = 20",
        replace_all=False,
    )

    assert test_file.read_text(encoding="utf-8") == content.replace(
        "x = 10",
        "x = 20",
        1,
    )


def test_replace_tool_updates_all_occurrences(
    default_agent_config,
    replace_tool,
):
    content = "Hello World\nHello again\n"
    test_file = write_step_file(
        default_agent_config,
        "replace_all.txt",
        content,
    )

    replace_tool.function(
        file_path=str(test_file),
        old="Hello",
        new="Goodbye",
        replace_all=True,
    )

    assert test_file.read_text(encoding="utf-8") == content.replace(
        "Hello",
        "Goodbye",
    )


def test_replace_tool_updates_multiline_text(
    default_agent_config,
    replace_tool,
):
    content = """def process():
    x = 1
    y = 2
    return x + y


def other():
    pass
"""
    test_file = write_step_file(
        default_agent_config,
        "multiline.py",
        content,
    )
    old = """def process():
    x = 1
    y = 2
    return x + y"""
    new = """def process():
    x = 10
    y = 20
    return x + y"""

    replace_tool.function(
        file_path=str(test_file),
        old=old,
        new=new,
        replace_all=False,
    )

    assert test_file.read_text(encoding="utf-8") == content.replace(old, new, 1)


def test_missing_text_leaves_file_unchanged(
    default_agent_config,
    replace_tool,
):
    content = "def foo():\n    pass\n"
    test_file = write_step_file(
        default_agent_config,
        "not_found.py",
        content,
    )

    result = replace_tool.function(
        file_path=str(test_file),
        old="DOES_NOT_EXIST",
        new="something",
        replace_all=False,
    )

    assert result.startswith("Error")
    assert test_file.read_text(encoding="utf-8") == content


def test_invalid_python_replacement_leaves_file_unchanged(
    default_agent_config,
    replace_tool,
):
    content = "def valid_function():\n    return 42\n"
    test_file = write_step_file(
        default_agent_config,
        "invalid_syntax.py",
        content,
    )

    result = replace_tool.function(
        file_path=str(test_file),
        old="return 42",
        new="return (((",
        replace_all=False,
    )

    assert result.startswith("Error")
    assert test_file.read_text(encoding="utf-8") == content


def test_file_outside_current_step_is_rejected(
    tmp_path: Path,
    replace_tool,
):
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("original", encoding="utf-8")

    result = replace_tool.function(
        file_path=str(outside_file),
        old="original",
        new="changed",
        replace_all=False,
    )

    assert result.startswith("Error")
    assert outside_file.read_text(encoding="utf-8") == "original"


def test_missing_file_is_rejected(
    default_agent_config,
    replace_tool,
):
    missing_file = default_agent_config.current_step_dir / "missing.txt"

    result = replace_tool.function(
        file_path=str(missing_file),
        old="old",
        new="new",
        replace_all=False,
    )

    assert "not valid" in result
    assert not missing_file.exists()
