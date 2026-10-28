import pytest

from agentomics.tools.supplementary_tool import create_supplementary_tool


@pytest.fixture
def supp_dir(default_agent_config):
    path = default_agent_config.dataset_dir / "supplementary"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def make_tool(default_agent_config):
    def make(**overrides):
        for key, value in overrides.items():
            setattr(default_agent_config, key, value)
        return create_supplementary_tool(default_agent_config).function

    return make


def test_list_shows_entries_with_sizes(supp_dir, make_tool):
    (supp_dir / "README.md").write_text("hello")
    (supp_dir / "sub").mkdir()
    (supp_dir / "paper.pdf").write_bytes(b"%PDF\x00\x01")

    out = make_tool()(operation="list")

    assert "README.md" in out and "5 B" in out
    assert "paper.pdf" in out and "binary" in out
    assert "sub/" in out


def test_read_returns_line_range(supp_dir, make_tool):
    (supp_dir / "a.txt").write_text("\n".join(f"line{i}" for i in range(1, 11)))

    out = make_tool()(operation="read", path="a.txt", start_line=3, max_lines=2)

    assert "line3" in out and "line4" in out
    assert "line2" not in out and "line5" not in out


def test_search_finds_matches_with_context(supp_dir, make_tool):
    (supp_dir / "a.txt").write_text("alpha\nbeta\nGamma\ndelta\n")

    out = make_tool()(operation="search", pattern="gamma", path="a.txt", case_sensitive=False, context_lines=1)

    assert "Gamma" in out and "beta" in out and "delta" in out
    assert "alpha" not in out
    assert "No matches" in make_tool()(operation="search", pattern="gamma", path="a.txt", case_sensitive=True)


def test_search_recursive_and_max_matches(supp_dir, make_tool):
    (supp_dir / "d").mkdir()
    (supp_dir / "d" / "x.txt").write_text("hit\nhit\nhit\n")
    tool = make_tool()

    assert "hit" not in tool(operation="search", pattern="hit", recursive=False)
    out = tool(operation="search", pattern="hit", recursive=True, max_matches=2)
    assert "x.txt:1" in out and "x.txt:2" in out and "x.txt:3" not in out
    assert "lower the match limit" in out or "max_matches" in out


@pytest.mark.parametrize("operation", ["list", "read", "search"])
def test_paths_outside_folder_rejected(supp_dir, make_tool, operation):
    (supp_dir.parent / "secret.txt").write_text("secret")

    out = make_tool()(operation=operation, path="../secret.txt", pattern="secret")

    assert "secret" not in out.replace("secret.txt", "")
    assert "outside" in out


def test_symlink_pointing_out_rejected(supp_dir, make_tool):
    (supp_dir.parent / "secret.txt").write_text("topsecret")
    (supp_dir / "link.txt").symlink_to(supp_dir.parent / "secret.txt")
    tool = make_tool()

    assert "topsecret" not in tool(operation="read", path="link.txt")
    assert "outside" in tool(operation="read", path="link.txt")
    assert "topsecret" not in tool(operation="search", pattern="topsecret", recursive=True)


def test_binary_files_are_never_read_or_searched(supp_dir, make_tool):
    (supp_dir / "b.bin").write_bytes(b"needle\x00needle")
    tool = make_tool()

    assert "binary" in tool(operation="read", path="b.bin")
    assert "needle" not in tool(operation="read", path="b.bin")
    assert "needle" not in tool(operation="search", pattern="needle", recursive=True)


def test_read_over_cap_keeps_head_and_hints_next_line(supp_dir, make_tool):
    (supp_dir / "big.txt").write_text("\n".join(f"row{i:04d}" for i in range(1, 1001)))

    out = make_tool(supplementary_tool_max_call_chars=200)(operation="read", path="big.txt", max_lines=1000)

    assert "row0001" in out and "row1000" not in out
    last_shown = max(int(w[3:7]) for w in out.split() if w.startswith("row"))
    assert f"start_line={last_shown + 1}" in out


def test_search_over_cap_keeps_head_and_hints_narrowing(supp_dir, make_tool):
    (supp_dir / "big.txt").write_text("\n".join(f"match{i:04d}" for i in range(1, 500)))

    out = make_tool(supplementary_tool_max_call_chars=200)(
        operation="search", pattern="match", path="big.txt", max_matches=500
    )

    assert "match0001" in out and "match0499" not in out
    assert "narrow" in out and "match limit" in out


def test_cumulative_cap_returns_budget_message(supp_dir, make_tool):
    (supp_dir / "a.txt").write_text("x" * 300)
    tool = make_tool(supplementary_tool_max_call_chars=200, supplementary_tool_max_total_chars=300)

    tool(operation="read", path="a.txt")
    tool(operation="read", path="a.txt")
    out = tool(operation="read", path="a.txt")

    assert "budget" in out and "notes" in out and "plan" in out
    assert "x" * 10 not in out


def test_call_budget_returns_message_not_error(supp_dir, make_tool):
    tool = make_tool(supplementary_tool_max_calls=2)

    tool(operation="list")
    tool(operation="list")
    out = tool(operation="list")

    assert "budget" in out and "notes" in out and "plan" in out


def test_budget_is_per_tool_instance(supp_dir, make_tool):
    first = make_tool(supplementary_tool_max_calls=1)
    first(operation="list")
    assert "budget" in first(operation="list")

    assert "budget" not in make_tool(supplementary_tool_max_calls=1)(operation="list")


def test_tool_is_not_in_shared_registry():
    from agentomics.tools.tool_registry import TOOL_FACTORY_REGISTRY

    assert "supplementary" not in TOOL_FACTORY_REGISTRY
