"""Test data module."""
import pytest
import numpy as np
from spectrum_analysis.data import GetData


@pytest.fixture(name='empty_file')
def e_file():
    """Return an empty file for testing."""
    return ''


@pytest.fixture(name='raman_file')
def r_file():
    """Return a raman data file for testing."""
    return 'examples/testdata/1/0001.txt'


@pytest.fixture(name='tribo_file')
def t_file():
    """Return a tribo data file for testing."""
    return 'examples/testdata/tribo/1.txt'


def test_GetData_with_raman_data(raman_file):
    """Tests get_data with standard Raman data."""
    xdata, ydata = GetData(raman_file)
    assert xdata.size == 1989
    assert ydata.size == 1989


def test_GetData_with_tribo_data(tribo_file):
    """Tests get_data with standard Tribo data."""
    xdata, ydata = GetData(tribo_file, measurement='tribo')
    assert xdata.size == 1558
    assert ydata.size == 1558


def test_GetData_without_file_specified(empty_file):
    """Tests get_data without data."""
    xdata, ydata = GetData(empty_file)
    zeros = np.zeros([1])
    np.testing.assert_equal(xdata, zeros)
    np.testing.assert_equal(ydata, zeros)
