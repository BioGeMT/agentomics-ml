import unittest


class TestCiFailureProbe(unittest.TestCase):
    def test_ci_reports_failure(self):
        self.fail("intentional CI failure probe")
