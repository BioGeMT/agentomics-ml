import unittest
import sys

if __name__ == "__main__":
    print("="*20, "Running all tests", "="*20)

    loader = unittest.TestLoader()
    suite = loader.discover('test', pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)