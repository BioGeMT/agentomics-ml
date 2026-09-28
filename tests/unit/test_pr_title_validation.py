import pytest

from tests.helpers import run_pr_title_validator


@pytest.mark.parametrize(
    ("title", "expected_impact"),
    [
        pytest.param(
            "fix: preserve uploaded filenames",
            "patch",
            id="fix",
        ),
        pytest.param("perf: reduce memory use", "patch", id="performance"),
        pytest.param("feat: add CSV exports", "minor", id="feature"),
        pytest.param(
            "feat!: remove the legacy workspace format",
            "major",
            id="breaking-feature",
        ),
        pytest.param(
            "fix!: reject the removed argument",
            "major",
            id="breaking-fix",
        ),
    ],
)
def test_releasing_types_declare_the_expected_impact(
    title: str,
    expected_impact: str,
):
    result = run_pr_title_validator(title)

    assert result.returncode == 0, result.stderr
    assert f"Release Impact: {expected_impact}" in result.stdout


@pytest.mark.parametrize(
    "title",
    [
        pytest.param("docs: explain workspace layout", id="documentation"),
        pytest.param("test: cover image selection", id="test"),
        pytest.param("ci: cache the container build", id="ci"),
        pytest.param("chore: update maintainers", id="chore"),
        pytest.param("refactor: share argument parsing", id="refactor"),
    ],
)
def test_documented_non_releasing_types_declare_no_release_impact(
    title: str,
):
    result = run_pr_title_validator(title)

    assert result.returncode == 0, result.stderr
    assert "Release Impact: none" in result.stdout


@pytest.mark.parametrize(
    "title",
    [
        pytest.param("feature: add CSV exports", id="unknown-type"),
        pytest.param("docs!: remove obsolete instructions", id="breaking-docs"),
        pytest.param("feat(cli): add CSV exports", id="scope"),
        pytest.param("feat : add CSV exports", id="space-before-colon"),
        pytest.param("feat:", id="missing-description"),
        pytest.param("Add CSV exports", id="missing-type"),
    ],
)
def test_invalid_or_ambiguous_titles_are_rejected(title: str):
    result = run_pr_title_validator(title)

    assert result.returncode != 0
