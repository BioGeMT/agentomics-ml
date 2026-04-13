from test.test_utils import BaseAgentTest

class BashLinesCollapsing(BaseAgentTest):
    def test_error_output_collapsing(self):
        """Test that error output also collapses repeated lines."""
        # Run a command that fails with repeated error messages
        result = self.bash_tool.function(
            "for i in {1..20}; do echo 'ERROR: failed' >&2; done; exit 1"
        )

        # Should indicate command failed
        self.assertIn("Command failed with error code 1", result)

        # Should collapse the repeated errors
        self.assertIn("... [line repeated", result)

        error_count = result.count("ERROR: failed")
        self.assertLess(error_count, 10)

    def test_repeated_warnings(self):
        """Test collapsing of repeated warning messages."""
        result = self.bash_tool.function(
            "for i in {1..50}; do echo 'WARNING: deprecated API'; done"
        )

        # Should see collapse message
        self.assertIn("... [line repeated", result)
        self.assertIn("more times] ...", result)

        # Should not see 50 individual warning lines
        warning_count = result.count("WARNING: deprecated API")
        self.assertLess(warning_count, 10,
                       f"Expected collapsed output, but found {warning_count} warning lines")

    def test_mixed_repeated_and_unique_lines(self):
        """Test mixed repeated and unique lines are handled correctly."""
        result = self.bash_tool.function(
            "echo 'Start'; "
            "for i in {1..5}; do echo 'Repeat'; done; "
            "echo 'Middle'; "
            "for i in {1..4}; do echo 'Again'; done; "
            "echo 'End'"
        )

        # Check structure is preserved
        self.assertIn("Start", result)
        self.assertIn("Middle", result)
        self.assertIn("End", result)
        self.assertIn("... [line repeated", result)

        # Should have 2 collapse messages (one for "Repeat", one for "Again")
        collapse_count = result.count("... [line repeated")
        self.assertEqual(collapse_count, 2)

    def test_no_repeated_lines(self):
        """Test that output without repeated lines is unchanged."""
        result = self.bash_tool.function(
            "echo 'line1'; echo 'line2'; echo 'line3'; echo 'line4'"
        )

        # Should not see any collapse messages
        self.assertNotIn("... [line repeated", result)

        # All unique lines should be present
        self.assertIn("line1", result)
        self.assertIn("line2", result)
        self.assertIn("line3", result)
        self.assertIn("line4", result)

    def test_below_threshold_not_collapsed(self):
        """Test that lines below threshold (only 1 occurrence) are not collapsed."""
        result = self.bash_tool.function(
            "echo 'Single line'; echo 'Another single line'"
        )

        # Should not see collapse message for single occurrences
        self.assertNotIn("... [line repeated", result)
        self.assertIn("Single line", result)
        self.assertIn("Another single line", result)

    def test_empty_lines_collapsing(self):
        """Test that repeated empty lines are collapsed."""
        result = self.bash_tool.function(
            "echo 'line1'; for i in {1..10}; do echo ''; done; echo 'line2'"
        )

        # Should collapse empty lines
        self.assertIn("... [line repeated", result)
        self.assertIn("line1", result)
        self.assertIn("line2", result)

    def test_large_output_with_collapsing(self):
        """Test that collapsing helps keep large output under truncation limit."""
        # Generate 1000 repeated warnings
        result = self.bash_tool.function(
            "for i in {1..1000}; do echo 'DEPRECATION: old API'; done; echo 'FINAL LINE'"
        )

        # Should see collapse message
        self.assertIn("... [line repeated", result)

        # Should see the final line (not truncated away)
        self.assertIn("FINAL LINE", result)

        # The collapse should reduce 1000 lines to just a few
        deprecation_count = result.count("DEPRECATION: old API")
        self.assertLess(deprecation_count, 10)

    def test_multiple_collapse_groups(self):
        """Test multiple groups of different repeated lines."""
        result = self.bash_tool.function(
            "for i in {1..10}; do echo 'Error A'; done; "
            "echo 'separator'; "
            "for i in {1..15}; do echo 'Error B'; done; "
            "echo 'separator'; "
            "for i in {1..5}; do echo 'Error C'; done"
        )

        # Should have 3 collapse groups
        collapse_count = result.count("... [line repeated")
        self.assertEqual(collapse_count, 3)

        # Separators should be preserved
        self.assertEqual(result.count("separator"), 2)
