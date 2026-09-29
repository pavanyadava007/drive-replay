"""Entry point for Bazel py_test: runs pytest on the file given as the first argument."""
import sys

import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", *sys.argv[1:]]))
