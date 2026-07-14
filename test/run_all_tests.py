import argparse
import unittest
import sys

import test.test_utils as test_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Agentomics test suite.")
    parser.add_argument("--workspace-dir", help="Workspace directory the tests run against.")
    args = parser.parse_args()

    if args.workspace_dir:
        test_utils.WORKSPACE_DIR = args.workspace_dir

    print("="*20, "Running all tests", "="*20)

    loader = unittest.TestLoader()
    suite = loader.discover('test', pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
