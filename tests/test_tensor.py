"""Tests for the Tensor Encoding (paper 112)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quilt_substrate.substrate import Cell, Substrate


def test_tensor_stored():
    c = Cell(address="t", value=None, tensor=[[[1.0, 2.0], [3.0, 4.0]]], axes=("d", "x", "y"))
    assert c.tensor == [[[1.0, 2.0], [3.0, 4.0]]]
    assert c.axes == ("d", "x", "y")


def test_slice_1d():
    c = Cell(address="t", value=None, tensor=[1.0, 2.0, 3.0], axes=("x",))
    out = c.slice(x=0)
    assert out == [1.0]


def test_slice_2d():
    c = Cell(address="t", value=None, tensor=[[1.0, 2.0], [3.0, 4.0]], axes=("x", "y"))
    out = c.slice(x=0)
    assert 1.0 in out
    assert 2.0 in out


def test_slice_3d():
    c = Cell(address="t", value=None, tensor=[[[1.0, 2.0], [3.0, 4.0]]], axes=("d", "x", "y"))
    out = c.slice(d=0)
    # d=0 layer is a 2x2 matrix: [[1.0, 2.0], [3.0, 4.0]]
    assert 1.0 in out
    assert 4.0 in out


def test_slice_unknown_axis_returns_none():
    c = Cell(address="t", value=None, tensor=[1.0], axes=("x",))
    out = c.slice(z=0)
    assert out is None


def test_bay_cross_section():
    """The bathy cross-section (scenario 03): depth × horizontal position."""
    s = Substrate()
    # Two cells with different depth profiles
    c1 = Cell(address="bay/A17", value=12.5,
              tensor=[[[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]],
              axes=("depth", "x", "y"))
    s.add(c1)
    # Slice the depth=0 layer (a 2x3 matrix → flat list of 6 elements)
    surface = c1.slice(depth=0)
    assert surface is not None
    assert 10.0 in surface
    assert 15.0 in surface


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
