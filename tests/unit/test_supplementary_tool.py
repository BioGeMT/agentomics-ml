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


def test_list_shows_entries_with_sizes(supp_dir, make_tool):
    (supp_dir / "README.md").write_text("hello")
    (supp_dir / "sub").mkdir()
    (supp_dir / "paper.pdf").write_bytes(b"%PDF\x00\x01")

    out = make_tool().ls()

    assert "README.md" in out
    assert "paper.pdf" in out
    assert "sub/" in out


def test_read_returns_line_range(supp_dir, make_tool):
    (supp_dir / "a.txt").write_text("\n".join(f"line{i}" for i in range(1, 11)))

    out = make_tool().read_file(path="a.txt", start_line=3, max_lines=2)

    assert "line3" in out and "line4" in out
    assert "line2" not in out and "line5" not in out


def test_search_finds_matches_with_context(supp_dir, make_tool):
    (supp_dir / "a.txt").write_text("alpha\nbeta\nGamma\ndelta\n")

    out = make_tool().grep(pattern="gamma", path="a.txt", case_sensitive=False, context_lines=1)

    assert "Gamma" in out and "beta" in out and "delta" in out
    assert "alpha" not in out
    assert "No matches" in make_tool().grep(pattern="gamma", path="a.txt", case_sensitive=True)


def test_search_is_recursive(supp_dir, make_tool):
    (supp_dir / "d").mkdir()
    (supp_dir / "d" / "x.txt").write_text("hit\nhit\nhit\n")
    tool = make_tool()

    out = tool.grep(pattern="hit")
    assert "d/x.txt:1" in out and "d/x.txt:3" in out


@pytest.mark.parametrize("operation", ["ls", "read_file", "grep"])
def test_paths_outside_folder_rejected(supp_dir, make_tool, operation):
    (supp_dir.parent / "secret.txt").write_text("secret")

    out = getattr(make_tool(), operation)(path="../secret.txt", **({"pattern": "secret"} if operation == "grep" else {}))

    assert "secret" not in out.replace("secret.txt", "")
    assert "outside" in out


def test_symlink_pointing_out_rejected(supp_dir, make_tool):
    (supp_dir.parent / "secret.txt").write_text("topsecret")
    (supp_dir / "link.txt").symlink_to(supp_dir.parent / "secret.txt")
    tool = make_tool()

    assert "topsecret" not in tool.read_file(path="link.txt")
    assert "outside" in tool.read_file(path="link.txt")
    assert "topsecret" not in tool.grep(pattern="topsecret")


def test_binary_files_are_never_read_or_searched(supp_dir, make_tool):
    (supp_dir / "b.bin").write_bytes(b"needle\x00needle")
    tool = make_tool()

    assert "binary" in tool.read_file(path="b.bin")
    assert "needle" not in tool.read_file(path="b.bin")
    assert "needle" not in tool.grep(pattern="needle")


def test_read_over_cap_keeps_head_and_hints_next_line(supp_dir, make_tool, small_cap):
    (supp_dir / "big.txt").write_text("\n".join(f"row{i:04d}" for i in range(1, 1001)))

    out = make_tool().read_file(path="big.txt", max_lines=1000)

    assert "row0001" in out and "row1000" not in out
    last_shown = max(int(w[3:7]) for w in out.split() if w.startswith("row"))
    assert f"start_line={last_shown + 1}" in out


def test_search_over_cap_keeps_head_and_hints_narrowing(supp_dir, make_tool, small_cap):
    (supp_dir / "big.txt").write_text("\n".join(f"match{i:04d}" for i in range(1, 500)))

    out = make_tool().grep(pattern="match", path="big.txt")

    assert "match0001" in out and "match0499" not in out
    assert "narrow" in out


def test_tools_are_plain_described_tools_not_in_shared_registry(default_agent_config):
    from agentomics.tools.tool_registry import TOOL_FACTORY_REGISTRY

    tools = create_supplementary_tools(default_agent_config)

    assert [t.name for t in tools] == ["ls", "grep", "read_file"]
    assert all(t.tool_def.description and t.tool_def.parameters_json_schema["properties"] for t in tools)
    assert not any(t.name in TOOL_FACTORY_REGISTRY or "supplementary" in TOOL_FACTORY_REGISTRY for t in tools)


def test_read_of_overlong_first_line_still_advances(supp_dir, make_tool, small_cap):
    (supp_dir / "wide.txt").write_text("x" * 1000 + "\nsecond\n")
    tool = make_tool()

    out = tool.read_file(path="wide.txt")

    assert "start_line=2" in out
    assert "second" in tool.read_file(path="wide.txt", start_line=2)
