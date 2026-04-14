"""Minimal torch + pyro mocks for pgmpy compatibility.

pgmpy unconditionally does ``import torch`` in ``global_vars.py`` and
``import pyro`` in ``FunctionalBayesianNetwork.py``.  When these heavy
libraries are not installed (we skip them because they're 100+ MB and only
needed for GPU acceleration / probabilistic programming), these mocks
satisfy the imports and let pgmpy fall back to its numpy backend.

Usage — call ``install()`` **before** any pgmpy import::

    from car.torch_mock import install
    install()
"""

from __future__ import annotations

import sys
import types
from typing import Any


# ---------------------------------------------------------------------------
# Lightweight stand-ins
# ---------------------------------------------------------------------------

class _FakeDevice:
    """Mimics ``torch.device``."""

    def __init__(self, name: str = "cpu", index: int | None = None):
        self._name = str(name)
        self.type = self._name.split(":")[0]
        self.index = index

    def __repr__(self) -> str:
        return f"device(type='{self.type}')"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _FakeDevice):
            return self.type == other.type
        if isinstance(other, str):
            return self.type == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.type)


class _FakeDtype:
    """Mimics ``torch.float32`` / ``torch.float64`` etc."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"torch.{self.name}"


class _FakeTensor:
    """Bare-minimum Tensor class so ``isinstance`` checks pass."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._data = args[0] if args else []

    # Methods pgmpy might probe
    def type(self, *args: Any, **kwargs: Any) -> "_FakeTensor":
        return self

    def to(self, *args: Any, **kwargs: Any) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def detach(self) -> "_FakeTensor":
        return self

    def numpy(self, **kwargs: Any) -> Any:
        import numpy as np
        return np.asarray(self._data)

    def nelement(self) -> int:
        return 0

    def item(self) -> float:
        return 0.0

    @property
    def device(self) -> _FakeDevice:
        return _FakeDevice("cpu")


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------

def _build_torch_module() -> types.ModuleType:
    """Construct a fake ``torch`` top-level module."""

    mod = types.ModuleType("torch")
    mod.__path__ = []  # type: ignore[attr-defined]
    mod.__package__ = "torch"
    mod.__file__ = __file__

    # Dtypes
    mod.float16 = _FakeDtype("float16")  # type: ignore[attr-defined]
    mod.float32 = _FakeDtype("float32")  # type: ignore[attr-defined]
    mod.float64 = _FakeDtype("float64")  # type: ignore[attr-defined]
    mod.int32 = _FakeDtype("int32")  # type: ignore[attr-defined]
    mod.int64 = _FakeDtype("int64")  # type: ignore[attr-defined]
    mod.bool = _FakeDtype("bool")  # type: ignore[attr-defined]

    # Device / Tensor
    mod.device = _FakeDevice  # type: ignore[attr-defined]
    mod.Tensor = _FakeTensor  # type: ignore[attr-defined]

    def _tensor(*args: Any, **kwargs: Any) -> _FakeTensor:
        return _FakeTensor(*args, **kwargs)

    mod.tensor = _tensor  # type: ignore[attr-defined]

    # ---- cuda sub-module ----
    cuda = types.ModuleType("torch.cuda")
    cuda.__package__ = "torch.cuda"
    cuda.is_available = lambda: False  # type: ignore[attr-defined]
    cuda.device_count = lambda: 0  # type: ignore[attr-defined]
    mod.cuda = cuda  # type: ignore[attr-defined]

    # ---- optim sub-module (pgmpy optimizer.py checks for it) ----
    optim = types.ModuleType("torch.optim")
    optim.__package__ = "torch.optim"
    mod.optim = optim  # type: ignore[attr-defined]

    # ---- distributions sub-module (sometimes probed) ----
    distributions = types.ModuleType("torch.distributions")
    distributions.__package__ = "torch.distributions"
    constraints = types.ModuleType("torch.distributions.constraints")
    constraints.__package__ = "torch.distributions.constraints"
    distributions.constraints = constraints  # type: ignore[attr-defined]
    mod.distributions = distributions  # type: ignore[attr-defined]

    # Math / array ops (return numpy results where possible)
    import numpy as np

    mod.amax = lambda arr, **kw: np.amax(arr, **kw)  # type: ignore[attr-defined]
    mod.argmax = lambda arr, **kw: np.argmax(arr, **kw)  # type: ignore[attr-defined]
    mod.sum = lambda arr, **kw: np.sum(arr, **kw)  # type: ignore[attr-defined]
    mod.ones = lambda *s, **kw: np.ones(s if len(s) != 1 or not isinstance(s[0], tuple) else s[0])  # type: ignore[attr-defined]
    mod.eye = lambda n, **kw: np.eye(n)  # type: ignore[attr-defined]
    mod.diag = lambda t, **kw: np.diag(t)  # type: ignore[attr-defined]
    mod.stack = lambda ts, **kw: np.stack(ts, **kw)  # type: ignore[attr-defined]
    mod.unique = lambda t, **kw: np.unique(t, **kw)  # type: ignore[attr-defined]
    mod.flip = lambda t, dims=None, **kw: np.flip(t, axis=dims)  # type: ignore[attr-defined]
    mod.where = lambda c, x, y: np.where(c, x, y)  # type: ignore[attr-defined]
    mod.allclose = lambda a, b, **kw: np.allclose(a, b, **kw)  # type: ignore[attr-defined]
    mod.einsum = np.einsum  # type: ignore[attr-defined]
    mod.mul = lambda a, b: np.multiply(a, b)  # type: ignore[attr-defined]
    mod.permute = lambda t, dims=None: np.transpose(t, axes=dims)  # type: ignore[attr-defined]

    return mod


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class _Stub:
    """Infinitely-nestable attribute stub.

    Returns itself for any attribute access or call, so expressions like
    ``pyro.optim.Adam({"lr": 1e-2})`` evaluate without error at class-
    definition time.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass  # accept any arguments silently

    def __getattr__(self, name: str) -> "_Stub":
        return _Stub()

    def __call__(self, *args: Any, **kwargs: Any) -> "_Stub":
        return _Stub()

    def __repr__(self) -> str:
        return "<Stub>"

    def __bool__(self) -> bool:
        return False

    def __iter__(self):
        return iter([])

    def items(self):
        return []


def _build_pyro_module() -> types.ModuleType:
    """Construct a fake ``pyro`` top-level module.

    pgmpy's ``FunctionalBayesianNetwork`` does ``import pyro`` at module
    level and uses ``pyro.optim.Adam(...)`` as a default argument in a
    method signature.  We only need enough for the class to *define* — we
    never actually instantiate FunctionalBayesianNetwork in the CAR pipeline.
    """
    mod = types.ModuleType("pyro")
    mod.__path__ = []  # type: ignore[attr-defined]
    mod.__package__ = "pyro"
    mod.__file__ = __file__

    # pyro.distributions — used in FunctionalBayesianNetwork / FunctionalCPD
    distributions = types.ModuleType("pyro.distributions")
    distributions.__package__ = "pyro.distributions"
    mod.distributions = distributions  # type: ignore[attr-defined]

    # pyro.optim — used as default arg: pyro.optim.Adam({"lr": 1e-2})
    optim = types.ModuleType("pyro.optim")
    optim.__package__ = "pyro.optim"
    optim.Adam = _Stub  # type: ignore[attr-defined]
    optim.SGD = _Stub  # type: ignore[attr-defined]
    mod.optim = optim  # type: ignore[attr-defined]

    # pyro.infer — used in fit() method body
    infer = types.ModuleType("pyro.infer")
    infer.__package__ = "pyro.infer"
    infer.SVI = _Stub  # type: ignore[attr-defined]
    infer.Trace_ELBO = _Stub  # type: ignore[attr-defined]
    infer.NUTS = _Stub  # type: ignore[attr-defined]
    infer.MCMC = _Stub  # type: ignore[attr-defined]
    mod.infer = infer  # type: ignore[attr-defined]

    # pyro.nn — sometimes probed
    nn = types.ModuleType("pyro.nn")
    nn.__package__ = "pyro.nn"
    mod.nn = nn  # type: ignore[attr-defined]

    # pyro.poutine
    poutine = types.ModuleType("pyro.poutine")
    poutine.__package__ = "pyro.poutine"
    mod.poutine = poutine  # type: ignore[attr-defined]

    # Commonly-used top-level functions (stubs)
    mod.sample = lambda *a, **kw: None  # type: ignore[attr-defined]
    mod.plate = lambda *a, **kw: None  # type: ignore[attr-defined]
    mod.param = lambda *a, **kw: None  # type: ignore[attr-defined]
    mod.set_rng_seed = lambda *a, **kw: None  # type: ignore[attr-defined]
    mod.clear_param_store = lambda: None  # type: ignore[attr-defined]
    mod.get_param_store = lambda: _Stub()  # type: ignore[attr-defined]

    return mod


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def install() -> None:
    """Register fake ``torch`` and ``pyro`` in ``sys.modules``.

    Safe to call multiple times — subsequent calls are no-ops if the
    modules are already registered.
    """
    # ---- torch ----
    if "torch" not in sys.modules:
        mod = _build_torch_module()
        sys.modules["torch"] = mod
        sys.modules["torch.cuda"] = mod.cuda  # type: ignore[attr-defined]
        sys.modules["torch.optim"] = mod.optim  # type: ignore[attr-defined]
        sys.modules["torch.distributions"] = mod.distributions  # type: ignore[attr-defined]
        sys.modules["torch.distributions.constraints"] = mod.distributions.constraints  # type: ignore[attr-defined]

    # ---- pyro ----
    if "pyro" not in sys.modules:
        pyro = _build_pyro_module()
        sys.modules["pyro"] = pyro
        sys.modules["pyro.distributions"] = pyro.distributions  # type: ignore[attr-defined]
        sys.modules["pyro.optim"] = pyro.optim  # type: ignore[attr-defined]
        sys.modules["pyro.infer"] = pyro.infer  # type: ignore[attr-defined]
        sys.modules["pyro.nn"] = pyro.nn  # type: ignore[attr-defined]
        sys.modules["pyro.poutine"] = pyro.poutine  # type: ignore[attr-defined]
