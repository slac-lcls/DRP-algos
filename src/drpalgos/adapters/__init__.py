"""
Adapters convert between detector views to per-segment parameter views.

GUI plugins operate in detector-space (i.e. all segments of a detector together)
returning `intents`. I.e. a plugin provides the user intent, e.g., of drawing an ROI.
The `adapter` is responsible for converting an intent into something actionable, which
requires turning it back into per-segment parameter sets. This is what is used by
the rest of the DRP and DAQ infrastructure.

Plugins and Adapters are matched based on class variables that describe the type
that they are operating on (e.g. `roi`). Adapters should be written per-algorithm,
but may be operating on the same type (i.e., multiple algorithms might be able to
use an `roi`).

Classes:
    ParamAdapter: Base class for adapters.

Functions:
    load_adapter: Resolve an algorithm's gui_plugin metadata to an adapter instance.
    verify_adapter_against_schema: Check an adapter's parameters exist in a schema.
"""

import importlib
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from drpalgos.geometry import DetectorGeometry


class ParamAdapter:
    """Base class mapping a detector-level intent onto per-segment parameters."""

    plugin: ClassVar[str] = ""  # E.g. roi
    """The plugin is the accepted intent type for the adapter. It MUST match a plugins `intent_type.`"""
    produces: ClassVar[Tuple[str, ...]] = ()
    """The schema properties per segment."""
    detector_scope: ClassVar[
        Tuple[str, ...]
    ] = ()
    """The detector-scope parameters (which will then apply to all segments)"""

    def check(
        self,
        intent: Dict[str, Any],
        geom: DetectorGeometry,
        segments: List[int],
        shared: Dict[str, Any],
    ) -> List[str]:
        """Return human-readable problems with applying intent (empty if none).

        Args:
            intent (Dict[str, Any]): The plugin's detector-level intent.

            geom (DetectorGeometry): The detector geometry.

            segments (List[int]): Segments being configured.

            shared (Dict[str, Any]): Values of the detector-scope parameters.

        Returns:
            problems (List[str]): Problems found. Callers should block on any entry
                starting with "Error:" and show the rest as warnings.
        """
        return []

    def partition(
        self,
        intent: Dict[str, Any],
        geom: DetectorGeometry,
        segments: List[int],
        shared: Dict[str, Any],
    ) -> Dict[int, Dict[str, Any]]:
        """Return the `produces` parameters for each segment the intent applies to.

        Segments the intent does not touch are omitted.

        Raises:
            ValueError: If the intent cannot be represented (see `check`).
        """
        raise NotImplementedError

    def assemble(
        self, seg_params: Dict[int, Dict[str, Any]], geom: DetectorGeometry
    ) -> Dict[str, Any]:
        """Rebuild a detector-level intent from per-segment parameters."""
        raise NotImplementedError


_ALLOWED_PREFIX: str = __name__  # "drpalgos.adapters"


def load_adapter(gui_plugin: Dict[str, Any]) -> Optional[ParamAdapter]:
    """Resolve gui_plugin["adapter"] ("module:Class") to an adapter instance.

    Only modules inside this package are allowed.

    Args:
        gui_plugin (Dict[str, Any]): The algorithm's gui_plugin metadata.

    Returns:
        adapter (Optional[ParamAdapter]): The adapter, or None if absent/invalid.
    """
    spec: str = gui_plugin.get("adapter", "")
    module, _, entry_point = spec.partition(":")

    if not (module == _ALLOWED_PREFIX or module.startswith(_ALLOWED_PREFIX + ".")):
        return None

    try:
        cls = getattr(importlib.import_module(module), entry_point)
    except (ImportError, AttributeError, ValueError):
        return None

    if not (isinstance(cls, type) and issubclass(cls, ParamAdapter)):
        return None

    return cls()


def verify_adapter_against_schema(
    adapter: ParamAdapter, schema: Dict[str, Any]
) -> List[str]:
    """Check every parameter the adapter writes or shares exists in the schema.

    Args:
        adapter (ParamAdapter): The adapter.

        schema (Dict[str, Any]): The algorithm's normalized JSON schema.

    Returns:
        missing (List[str]): Problems found (empty if the adapter matches the schema).
    """
    props: Dict[str, Any] = schema.get("properties", {})
    return [
        f"Adapter parameter '{name}' is not in the algorithm schema"
        for name in adapter.produces + adapter.detector_scope
        if name not in props
    ]
