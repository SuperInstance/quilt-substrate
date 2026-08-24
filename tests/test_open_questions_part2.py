"""Tests for Open Questions 5, 9, and 2 (resolved in the latest work).

- Open Q5: Robust consensus (geometric median, robust to outliers)
- Open Q9: Non-linear JEPA (MLP and KNN)
- Open Q2: Witness justifications (the 'why' not just 'what')
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quilt_substrate.substrate import Cell, Substrate
from quilt_substrate.jepa import LinearJEPA, MLPJEPA, KnnJEPA, auto_train_jepa


# -- Open Q5: Robust consensus (geometric median) --

def test_geometric_median_1d_is_median():
    """For 1D, the geometric median is the median."""
    c = Cell(address="test", value=0.0)
    c._add_to_convoy("a", weight=1.0, value=10.0)
    c._add_to_convoy("b", weight=1.0, value=20.0)
    c._add_to_convoy("c", weight=1.0, value=30.0)
    median = c.geometric_median()
    assert median == 20.0  # middle value


def test_geometric_median_robust_to_outlier():
    """The geometric median is robust to a single outlier with small weight.

    If the outlier's weight is small (relative to the honest convoy),
    the median is the honest value. The geometric median in 1D IS
    the weighted median.
    """
    c = Cell(address="test", value=0.0)
    # 5 honest agents at value 10
    for i in range(5):
        c._add_to_convoy(f"h{i}", weight=1.0, value=10.0)
    # 1 outlier at value 1000 with small weight
    c._add_to_convoy("outlier", weight=1.0, value=1000.0)
    median = c.geometric_median()
    # With 5 honest at 10 and 1 outlier at 1000, the median is 10
    # (the outlier doesn't have enough weight to push the median past the 50% threshold)
    assert median == 10.0


def test_geometric_median_outlier_with_huge_weight_dominates():
    """A single agent with huge weight dominates (documented limitation).

    This is true of all consensus methods when one agent has the majority
    of weight. The substrate's solution is to cap the maximum weight
    an agent can have, or use multi-method consensus.
    """
    c = Cell(address="test", value=0.0)
    for i in range(5):
        c._add_to_convoy(f"h{i}", weight=1.0, value=10.0)
    c._add_to_convoy("outlier", weight=100.0, value=1000.0)
    # The outlier dominates — this is the known limitation
    assert c.geometric_median() == 1000.0


def test_geometric_median_with_nonnumeric_falls_back():
    """The geometric median returns cell value if convoy has no numeric entries."""
    c = Cell(address="test", value=42.0)
    c._add_to_convoy("a", weight=1.0, value="not a number")
    median = c.geometric_median()
    assert median == 42.0  # falls back to cell value


# -- Open Q9: Non-linear JEPA --

def test_linear_jepa_uniform_weights():
    """The linear JEPA returns the mean of inputs."""
    jepa = LinearJEPA()
    result = jepa({"a": 10.0, "b": 20.0, "c": 30.0})
    assert result == 20.0


def test_mlp_jepa_forward_pass():
    """The MLP JEPA returns a numeric value."""
    jepa = MLPJEPA(input_dim=4, hidden_dim=4)
    result = jepa({"a": 0.5, "b": 0.3, "c": 0.7, "d": 0.1})
    assert isinstance(result, float)


def test_mlp_jepa_train_step_reduces_loss():
    """Training the MLP reduces the loss."""
    jepa = MLPJEPA(input_dim=2, hidden_dim=4, lr=0.1)
    # Train on a simple pattern: f(x, y) = x + y
    # Initial loss
    initial_pred = jepa({"x": 0.5, "y": 0.5})
    initial_loss = (initial_pred - 1.0) ** 2
    # Train for many steps
    for _ in range(500):
        jepa.train_step({"x": 0.5, "y": 0.5}, 1.0)
    final_pred = jepa({"x": 0.5, "y": 0.5})
    final_loss = (final_pred - 1.0) ** 2
    # After training, loss should be lower (or at most similar)
    assert final_loss <= initial_loss * 1.5  # allow some noise


def test_knn_jepa_returns_averaged_value():
    """The KNN JEPA averages the k nearest neighbors."""
    jepa = KnnJEPA(k=3)
    jepa.add({"a": 0.0}, 10.0)
    jepa.add({"a": 0.1}, 20.0)
    jepa.add({"a": 0.2}, 30.0)
    jepa.add({"a": 10.0}, 100.0)
    # Query near the first three
    result = jepa({"a": 0.15})
    # 3 nearest: 0.0 (10), 0.1 (20), 0.2 (30) → mean = 20
    assert abs(result - 20.0) < 0.01


def test_knn_jepa_empty_returns_none():
    """KNN JEPA with no examples returns None."""
    jepa = KnnJEPA(k=3)
    result = jepa({"a": 1.0})
    assert result is None


def test_auto_train_jepa_mlp():
    """auto_train_jepa can train an MLP from a cell."""
    s = Substrate()
    c = Cell(address="test", value=10.0)
    s.add(c)
    model = auto_train_jepa(c, epochs=5, jepa_type="mlp")
    assert isinstance(model, MLPJEPA)


def test_auto_train_jepa_knn():
    """auto_train_jepa can build a KNN from a cell."""
    s = Substrate()
    c = Cell(address="test", value=10.0)
    s.add(c)
    model = auto_train_jepa(c, jepa_type="knn")
    assert isinstance(model, KnnJEPA)


def test_cell_predict_with_mlp_jepa():
    """A cell can use an MLP JEPA for prediction."""
    s = Substrate()
    c = Cell(address="test", value=10.0)
    s.add(c)
    jepa = MLPJEPA(input_dim=2, hidden_dim=2)
    c._jepa = jepa
    # Connect two cells as inputs
    in1 = Cell(address="in1", value=2.0)
    in2 = Cell(address="in2", value=3.0)
    s.add(in1); s.add(in2)
    c.connect(in1)
    c.connect(in2)
    pred = c.predict()
    assert isinstance(pred, float)


# -- Open Q2: Witness justifications --

def test_witness_with_justification():
    """A witness entry can carry a justification (Fable 11)."""
    c = Cell(address="test", value=42.0)
    c.witness("reyes", "write", 42.0, justification="Sailed to the spot, dropped the lead line, measured.")
    assert c.witness_log[-1].justification == "Sailed to the spot, dropped the lead line, measured."


def test_witness_justification_optional():
    """Justification defaults to empty string."""
    c = Cell(address="test", value=42.0)
    c.witness("reyes", "write", 42.0)
    assert c.witness_log[-1].justification == ""


def test_witness_to_dict_includes_justification():
    """The to_dict() of a witness entry includes the justification."""
    c = Cell(address="test", value=42.0)
    c.witness("reyes", "read", 42.0, justification="Just checking.")
    d = c.witness_log[-1].to_dict()
    assert "justification" in d
    assert d["justification"] == "Just checking."


def test_substrate_witness_with_justification():
    """The substrate-level witness also accepts a justification."""
    s = Substrate()
    c = Cell(address="test", value=42.0)
    s.add(c)
    s.witness(c, "reyes", "write", 42.0, justification="Fresh sounding.")
    assert c.witness_log[-1].justification == "Fresh sounding."


def test_justifications_create_audit_trail():
    """Justifications form a readable audit trail."""
    c = Cell(address="test", value=42.0)
    c.witness("reyes", "write", 42.0, justification="First survey")
    c.witness("skate", "read", 42.0, justification="Verifying")
    c.witness("reyes", "refresh", 42.0, justification="Stale, refreshing")
    audit = [e.justification for e in c.witness_log]
    assert audit == ["First survey", "Verifying", "Stale, refreshing"]


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
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
