"""Tests for Fable 19 (Oracle) + Fable 22 (Sundial) integration in the substrate.

These tests validate that the substrate now properly handles:
- Fable 19: Inference confidence (the oracle's confidence in her prophecy)
- Fable 22: Time-based decay (the sundial measures passage)
- Fable 19+22: Inference confidence decays over time
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from substrate import Cell


def test_inference_has_confidence():
    """An inference has a confidence (Fable 19)."""
    c = Cell(address="test/001", value=42.0)
    c.infer(50.0, confidence=0.9)
    assert c.inference == 50.0
    assert c.inference_confidence > 0.5  # freshly set


def test_inference_confidence_decays():
    """Inference confidence decays over time (Fable 22)."""
    c = Cell(address="test/001", value=42.0)
    c.decay.lam = 0.1  # fast decay for the test
    c.infer(50.0, confidence=1.0)
    c0 = c.inference_confidence
    time.sleep(0.5)  # let some time pass
    c1 = c.inference_confidence
    assert c1 < c0, f"Inference confidence should decay: {c0} → {c1}"


def test_inference_confidence_threshold():
    """Fable 19: The oracle must threshold her confidence."""
    c = Cell(address="test/001", value=42.0)
    c.infer(50.0, confidence=0.9)
    # High threshold: should return the inference
    assert c.confident_inference(threshold=0.5) == 50.0
    # Even higher threshold: should also return
    assert c.confident_inference(threshold=0.8) == 50.0
    # Very high threshold: should refuse
    assert c.confident_inference(threshold=0.95) is None


def test_inference_decays_below_threshold():
    """Fable 19+22: Inference that decays below threshold becomes unreliable."""
    c = Cell(address="test/001", value=42.0)
    c.decay.lam = 1.0  # very fast decay
    c.infer(50.0, confidence=1.0)
    assert c.confident_inference(threshold=0.5) == 50.0  # fresh
    time.sleep(2.0)  # let it decay significantly
    # After 2 seconds with lam=1.0, exp(-5*1.0*2.0) = exp(-10) ≈ 4.5e-5
    assert c.confident_inference(threshold=0.5) is None


def test_inference_confidence_clamped():
    """Confidence is clamped to [0, 1]."""
    c = Cell(address="test/001", value=42.0)
    c.infer(50.0, confidence=2.0)  # over
    assert c.inference_confidence <= 1.0
    c.infer(50.0, confidence=-0.5)  # under
    # The second infer overwrites the first; the second is clamped to 0.0
    # But we need to test the clamped value at inference time
    # Actually, the stored _inference_confidence is 0.0, but inference_confidence property
    # multiplies by exp(-5*lam*elapsed). For lam=0.0001, elapsed≈0, so it's ~0
    assert c._inference_confidence >= 0.0


def test_inference_with_fast_decay_dominates_canonical():
    """A confidently inferred value (when fresh) is the right value to read."""
    c = Cell(address="test/001", value=42.0)
    c.decay.lam = 0.0001
    c.infer(50.0, confidence=0.95)  # high confidence, just inferred
    # The canonical value is still 42.0
    assert c.value == 42.0
    # The inference is 50.0 with high confidence
    assert c.confident_inference() == 50.0


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
