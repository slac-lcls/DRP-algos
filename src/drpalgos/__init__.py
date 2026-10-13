"""
Python interface to DrpAlgos.

The compiled bindings (`_drpalgos`) need the CUDA runtime.

The pure-Python subpackages do not require CUDA:
  - `drpalgos.geometry`
  - `drpalgos.adapters`
  - `drpalgos.presets`

However, accessing a binding there raises the original ImportError.
"""

try:
    from ._drpalgos import *
except ImportError as _err:
    _bindings_error: ImportError = _err

    def __getattr__(name: str):
        raise ImportError(
            f"drpalgos compiled bindings are unavailable ({_bindings_error}); "
            f"cannot access '{name}'"
        ) from _bindings_error
