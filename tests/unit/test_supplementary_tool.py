from types import SimpleNamespace

import pytest

from agentomics.tools.supplementary_tool import create_supplementary_tools


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
        tools = {t.name: t.function for t in create_supplementary_tools(default_agent_config)}
        return SimpleNamespace(**tools)

    return make


def test_list_shows_entries_with_sizes(supp_dir, make_tool):
    (supp_dir / "README.md").write_text("hello")
    (supp_dir / "sub").mkdir()
    (supp_dir / "paper.pdf").write_bytes(b"%PDF\x00\x01")

    out = make_tool().ls()

    assert "README.md" in out and "5 B" in out
    assert "paper.pdf" in out and "binary" in out
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


def test_search_recursive_and_max_matches(supp_dir, make_tool):
    (supp_dir / "d").mkdir()
    (supp_dir / "d" / "x.txt").write_text("hit\nhit\nhit\n")
    tool = make_tool()

    assert "hit" not in tool.grep(pattern="hit", recursive=False)
    out = tool.grep(pattern="hit", recursive=True, max_matches=2)
    assert "x.txt:1" in out and "x.txt:2" in out and "x.txt:3" not in out
    assert "lower the match limit" in out or "max_matches" in out


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
    assert "topsecret" not in tool.grep(pattern="topsecret", recursive=True)


def test_binary_files_are_never_read_or_searched(supp_dir, make_tool):
    (supp_dir / "b.bin").write_bytes(b"needle\x00needle")
    tool = make_tool()

    assert "binary" in tool.read_file(path="b.bin")
    assert "needle" not in tool.read_file(path="b.bin")
    assert "needle" not in tool.grep(pattern="needle", recursive=True)


def test_read_over_cap_keeps_head_and_hints_next_line(supp_dir, make_tool):
    (supp_dir / "big.txt").write_text("\n".join(f"row{i:04d}" for i in range(1, 1001)))

    out = make_tool(supplementary_tool_max_call_chars=200).read_file(path="big.txt", max_lines=1000)

    assert "row0001" in out and "row1000" not in out
    last_shown = max(int(w[3:7]) for w in out.split() if w.startswith("row"))
    assert f"start_line={last_shown + 1}" in out


def test_search_over_cap_keeps_head_and_hints_narrowing(supp_dir, make_tool):
    (supp_dir / "big.txt").write_text("\n".join(f"match{i:04d}" for i in range(1, 500)))

    out = make_tool(supplementary_tool_max_call_chars=200).grep(
        pattern="match", path="big.txt", max_matches=500
    )

    assert "match0001" in out and "match0499" not in out
    assert "narrow" in out and "match limit" in out


def test_cumulative_cap_returns_budget_message(supp_dir, make_tool):
    (supp_dir / "a.txt").write_text("x" * 300)
    tool = make_tool(supplementary_tool_max_call_chars=200, supplementary_tool_max_total_chars=300)

    outputs = [tool.read_file(path="a.txt") for _ in range(5)]
    out = outputs[-1]

    assert "budget" in out and "notes" in out and "plan" in out
    assert "x" * 10 not in out


def test_call_budget_returns_message_not_error(supp_dir, make_tool):
    tool = make_tool(supplementary_tool_max_calls=2)

    tool.ls()
    tool.ls()
    out = tool.ls()

    assert "budget" in out and "notes" in out and "plan" in out


def test_budget_is_per_tool_instance(supp_dir, make_tool):
    first = make_tool(supplementary_tool_max_calls=1)
    first.ls()
    assert "budget" in first.ls()

    assert "budget" not in make_tool(supplementary_tool_max_calls=1).ls()


def test_budget_is_shared_across_the_three_tools(supp_dir, make_tool):
    (supp_dir / "a.txt").write_text("a")
    tool = make_tool(supplementary_tool_max_calls=2)

    tool.ls()
    tool.read_file(path="a.txt")

    assert "budget" in tool.grep(pattern="a")


def test_tools_are_plain_described_tools_not_in_shared_registry(default_agent_config):
    from agentomics.tools.tool_registry import TOOL_FACTORY_REGISTRY

    tools = create_supplementary_tools(default_agent_config)

    assert [t.name for t in tools] == ["ls", "grep", "read_file"]
    assert all(t.tool_def.description and t.tool_def.parameters_json_schema["properties"] for t in tools)
    assert not any(t.name in TOOL_FACTORY_REGISTRY or "supplementary" in TOOL_FACTORY_REGISTRY for t in tools)


def test_read_of_overlong_first_line_still_advances(supp_dir, make_tool):
    (supp_dir / "wide.txt").write_text("x" * 1000 + "\nsecond\n")
    tool = make_tool(supplementary_tool_max_call_chars=200)

    out = tool.read_file(path="wide.txt")

    assert "start_line=2" in out
    assert "second" in tool.read_file(path="wide.txt", start_line=2)


@pytest.mark.parametrize("cap", [150, 80, 40])
def test_output_including_hint_never_exceeds_per_call_cap(supp_dir, make_tool, cap):
    (supp_dir / "big.txt").write_text("\n".join(f"row{i:04d}" for i in range(1, 300)))
    for i in range(300):
        (supp_dir / f"f{i:03d}.txt").write_text("x")
    tool = make_tool(supplementary_tool_max_call_chars=cap, supplementary_tool_max_calls=10)

    assert len(tool.read_file(path="big.txt", max_lines=500)) <= cap
    assert len(tool.grep(pattern="row", path="big.txt", max_matches=500)) <= cap
    assert len(tool.ls()) <= cap


def test_continuation_note_counts_toward_per_call_cap(supp_dir, make_tool):
    (supp_dir / "a.txt").write_text("\n".join("y" * 18 for _ in range(10)))

    out = make_tool(supplementary_tool_max_call_chars=200).read_file(path="a.txt", max_lines=5)

    assert len(out) <= 200 and "start_line=" in out


def test_total_cap_includes_appended_hints(supp_dir, make_tool):
    (supp_dir / "a.txt").write_text("\n".join("z" * 20 for _ in range(100)))
    tool = make_tool(supplementary_tool_max_call_chars=120, supplementary_tool_max_total_chars=250)

    total = len(tool.read_file(path="a.txt")) + len(tool.read_file(path="a.txt"))

    assert total <= 250


def test_non_positive_arguments_are_clamped(supp_dir, make_tool):
    (supp_dir / "a.txt").write_text("one\ntwo\nthree\n")
    tool = make_tool()

    out = tool.read_file(path="a.txt", start_line=-5, max_lines=0)
    assert "1: one" in out and "2: two" not in out
    out = tool.grep(pattern="two", path="a.txt", context_lines=-3, max_matches=0)
    assert "two" in out and "one" not in out


def test_list_marks_outward_symlink_as_rejected(supp_dir, make_tool):
    (supp_dir.parent / "secret.txt").write_text("topsecret")
    (supp_dir / "link.txt").symlink_to(supp_dir.parent / "secret.txt")

    out = make_tool().ls()

    assert "link.txt" in out and "rejected" in out and "topsecret" not in out
