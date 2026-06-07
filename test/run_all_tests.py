import argparse
import unittest
import sys

import test.test_utils as test_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Agentomics test suite.")
    parser.add_argument("--workspace-dir", help="Workspace directory the tests run against.")
    parser.add_argument("--prepared-datasets-dir", help="Directory containing the prepared datasets.")
    args = parser.parse_args()

    if args.workspace_dir:
        test_utils.WORKSPACE_DIR = args.workspace_dir
    if args.prepared_datasets_dir:
        test_utils.PREPARED_DATASETS_DIR = args.prepared_datasets_dir

    print("="*20, "Running all tests", "="*20)

    loader = unittest.TestLoader()
    suite = loader.discover('test', pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
