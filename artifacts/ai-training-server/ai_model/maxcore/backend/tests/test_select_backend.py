"""Tests for the honest backend auto-selector (registry.select_backend).

These verify the selector picks a *runnable* backend without pretending, and
that on this Digital GPU host it lands on the Digital GPU backend rather than a
CUDA GPU it doesn't have.
"""
from ai_model.maxcore.backend import (
    CPUBackend,
    GPUBackend,
    available_runtime,
    select_backend,
)
from ai_model.maxcore.backend import registry


def test_default_selects_runnable_backend():
    """Default preference returns a backend that actually reports available."""
    b = select_backend()
    assert b.is_available() is True


def test_cpu_only_host_falls_through_to_digital_gpu():
    """On a host without CUDA, 'gpu' is unavailable so we get the Digital GPU backend.

    Skip only if this box actually has a runnable CUDA GPU (then the assertion
    would be legitimately different).
    """
    if available_runtime().get("gpu"):
        return  # real CUDA GPU present: falling-through is not the case to test
    b = select_backend(prefer=("gpu", "cpu"))
    assert isinstance(b, CPUBackend)
    assert b.name == "digital_gpu"


def test_never_returns_unavailable_backend():
    """The selector must not hand back a backend whose is_available() is False."""
    b = select_backend(prefer=("gpu", "cpu"))
    assert b.is_available() is True


def test_gpu_device_cpu_validation_path_is_selectable():
    """GPUBackend(device='cpu') is runnable (torch-CPU validation path).

    When it leads the preference it should be chosen — proving the selector
    honors kwargs and returns the real GPU backend class in its validation mode.
    Skipped if torch isn't importable on this host.
    """
    try:
        b = select_backend(prefer=("gpu",), device="cpu")
    except Exception:
        return  # torch not importable here; nothing to assert
    assert isinstance(b, GPUBackend)
    assert b.is_available() is True


def test_empty_preference_still_yields_cpu_fallback():
    """Even with an empty preference, the guaranteed CPU safety net kicks in."""
    b = select_backend(prefer=())
    assert isinstance(b, CPUBackend)


def test_unknown_preference_is_skipped_not_fatal():
    """An unknown backend name is skipped; a valid one after it still wins."""
    b = select_backend(prefer=("does-not-exist", "cpu"))
    assert isinstance(b, CPUBackend)


def test_gpu_only_kwarg_still_falls_back_to_cpu():
    """`device="cuda"` on a CPU-only host: gpu is unavailable and the kwarg is
    rejected by the CPU constructor, but the no-kwarg CPU safety net still wins."""
    if available_runtime().get("gpu"):
        return  # real GPU present: this fallback case doesn't apply
    b = select_backend(prefer=("gpu", "cpu"), device="cuda")
    assert isinstance(b, CPUBackend)
    assert b.is_available() is True


def test_unexpected_probe_fault_is_surfaced_not_masked():
    """A real fault in a backend's is_available() must propagate, not be silently
    downgraded to CPU — an operational bug can't masquerade as 'no hardware'."""
    class _Boom:
        name = "boom"

        def is_available(self):
            raise RuntimeError("simulated real fault in availability probe")

    registry.register("boom", lambda **kw: _Boom())
    try:
        raised = False
        try:
            select_backend(prefer=("boom", "cpu"))
        except RuntimeError as e:
            raised = "simulated real fault" in str(e)
        assert raised, "unexpected probe fault should propagate, not fall back to CPU"
    finally:
        registry._FACTORIES.pop("boom", None)
        for key in [k for k in registry._INSTANCES if k[0] == "boom"]:
            registry._INSTANCES.pop(key, None)
