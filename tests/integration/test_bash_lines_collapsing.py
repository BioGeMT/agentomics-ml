import pytest


@pytest.mark.parametrize(
    ("output_lines", "expected_lines"),
    [
        pytest.param(
            ["Start", "Middle", "End"],
            ["Start", "Middle", "End"],
            id="wholly-unique",
        ),
        pytest.param(
            ["Repeat"] * 6 + ["Between"] + ["Repeat"] * 5,
            ["Repeat", "Between", "Repeat"],
            id="separated-repeated-groups",
        ),
    ],
)
def test_bash_output_preserves_unique_lines_and_separate_groups(
    bash_tool, output_lines, expected_lines,
):
    result = bash_tool.function("printf '%s\\n' " + " ".join(output_lines))

    retained_lines = [line for line in result.splitlines() if line in output_lines]
    assert retained_lines == expected_lines


def test_bash_output_collapses_repeated_lines_without_losing_surrounding_output(
    bash_tool,
):
    result = bash_tool.function(
        "echo Start; "
        "for i in {1..6}; do echo Repeat; done; "
        "echo End"
    )

    assert result.index("Start") < result.index("Repeat") < result.index("End")
    assert result.count("Repeat") == 1
    assert "[line repeated" in result


def test_failed_bash_output_reports_failure_and_collapses_repeated_errors(
    bash_tool,
):
    result = bash_tool.function(
        "for i in {1..20}; do echo 'ERROR: failed' >&2; done; exit 7"
    )

    assert "Command failed with error code 7" in result
    assert sum(line == "ERROR: failed" for line in result.splitlines()) == 1
    assert "[line repeated" in result


def test_repeated_empty_lines_are_collapsed_without_losing_surrounding_output(
    bash_tool,
):
    result = bash_tool.function(
        "echo Before; for i in {1..10}; do echo; done; echo After"
    )

    assert result.index("Before") < result.index("[line repeated") < result.index("After")


def test_long_bash_output_preserves_its_beginning_and_end(bash_tool):
    result = bash_tool.function(
        "echo BEGIN; "
        "for i in {1..600}; do printf 'unique-%04d-abcdefghij\\n' \"$i\"; done; "
        "echo END"
    )

    assert "output truncated, too long" in result
    assert "BEGIN" in result
    assert "END" in result
