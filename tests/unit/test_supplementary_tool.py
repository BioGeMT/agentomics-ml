from types import SimpleNamespace

import pytest

from agentomics.tools.supplementary_tool import create_supplementary_tools


@pytest.fixture
def supp_dir(default_agent_config):
    path = default_agent_config.dataset_dir / "supplementary"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def small_cap(monkeypatch):
    monkeypatch.setattr("agentomics.tools.supplementary_tool.MAX_OUTPUT_CHARS", 200)


@pytest.fixture
def make_tool(default_agent_config):
    def make():
        tools = {t.name: t.function for t in create_supplementary_tools(default_agent_config)}
        return SimpleNamespace(**tools)

    return make


@pytest.mark.parametrize("operation", ["ls", "read_file", "grep"])
def test_paths_outside_folder_rejected(supp_dir, make_tool, operation):
    (supp_dir.parent / "secret.txt").write_text("secret")

    out = getattr(make_tool(), operation)(path="../secret.txt", **({"pattern": "secret"} if operation == "grep" else {}))

    assert "secret" not in out.replace("secret.txt", "")
    assert "outside" in out


def test_read_of_overlong_first_line_still_advances(supp_dir, make_tool, small_cap):
    (supp_dir / "wide.txt").write_text("x" * 1000 + "\nsecond\n")
    tool = make_tool()

    out = tool.read_file(path="wide.txt")

    assert "start_line=2" in out
    assert "second" in tool.read_file(path="wide.txt", start_line=2)
