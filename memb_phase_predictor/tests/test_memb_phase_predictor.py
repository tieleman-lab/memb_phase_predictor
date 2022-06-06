"""
Unit and regression test for the memb_phase_predictor package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import memb_phase_predictor


def test_memb_phase_predictor_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "memb_phase_predictor" in sys.modules
