"""Tools for handling DRP algorithm parameter presets in detector-wide fashion.

The DRP works in segments/panels/DRP processes. In general, though, we think in
"detectors". A preset naming schema is employed to allow encoding/decoding presets
into the per-segment format required by all backend infrastructure and back to the
detector-wide format for UI presentation.

Encoding schema (v. 1) is currently:
    `__PLG<VER>.<NAME>.<DETTYPE>.<GRP>.<SEG>.<CNT>.<TS>__<user name>`

- <VER> is the schema version
- <NAME> is the name of a GUI plugin
- <DETTYPE> is the detector type handled. (e.g. epixuhrhw)
- <GRP> is random group ID (currently 8 character hex)
- <SEG> is the detector segment number.
- <CNT> is the total number of segments in the detector group.
- <TS> is a timestamp. (UTC)

User name comes after - its whatever you would normally name the preset.

Classes:
    GroupedPresetName: The decoded fields of one grouped segment preset name.
    PresetGroup: The segment presets forming one detector preset.

Functions:
    new_preset_group: New random group identifier.
    encode_preset_name: Encode a segment preset name as a member of a group.
    decode_preset_name: Decode a grouped segment preset name.
    group_presets: Split a preset listing into plain presets and detector presets.
"""

import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

_DET_PRESET_VER: int = 1
"""Version of the encoding schema."""
RESERVED_PREFIX: str = "__PLG"
"""Plain preset names may not start with this."""
_GROUP_PREFIX: str = f"__PLG{_DET_PRESET_VER}."
_GROUP_END: str = "__"
_FIELD_RE: re.Pattern = re.compile(r"^[A-Za-z0-9]+$")
_GROUP_RE: re.Pattern = re.compile(
    r"^__PLG1\.(?P<plugin>[A-Za-z0-9]+)\.(?P<det_type>[A-Za-z0-9]+)\.(?P<group>[0-9a-f]+)"
    r"\.(?P<seg>\d+)\.(?P<count>\d+)\.(?P<ts>\d{8}T\d{6})__(?P<name>.*)$",
    re.DOTALL,
)


@dataclass(frozen=True)
class GroupedPresetName:
    plugin: str
    det_type: str
    group: str
    segment: int
    count: int
    timestamp: str
    name: str


def new_preset_group() -> str:
    """Return a new group identifier for a set of segment presets."""
    return secrets.token_hex(4)


def encode_preset_name(
    name: str,
    plugin: str,
    det_type: str,
    group: str,
    segment: int,
    count: int,
    timestamp: Optional[str] = None,
) -> str:
    """Encode a segment preset name as a member of a detector-level group."""
    if name.startswith(RESERVED_PREFIX):
        raise ValueError(
            f"Preset names may not start with the reserved prefix '{RESERVED_PREFIX}'"
        )
    for label, value in (("plugin", plugin), ("det_type", det_type), ("group", group)):
        if not _FIELD_RE.match(value):
            raise ValueError(f"{label} must be alphanumeric: {value!r}")

    if not (0 <= segment and 0 < count):
        raise ValueError("Invalid segment/count")

    ts: str = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    return f"{_GROUP_PREFIX}{plugin}.{det_type}.{group}.{segment}.{count}.{ts}{_GROUP_END}{name}"


def decode_preset_name(preset_name: str) -> Optional[GroupedPresetName]:
    """Decode a grouped segment preset name, or return None for an ordinary preset."""
    m: Optional[re.Match] = _GROUP_RE.match(preset_name)
    if m is None:
        return None

    return GroupedPresetName(
        m["plugin"],
        m["det_type"],
        m["group"],
        int(m["seg"]),
        int(m["count"]),
        m["ts"],
        m["name"],
    )


GroupKey = Tuple[str, str, str]  # (plugin, det_type, group)


@dataclass
class PresetGroup:
    """The segment presets forming one detector preset."""

    plugin: str
    det_type: str
    group: str
    count: int  # Number of segments the group was saved with
    timestamp: str
    name: str
    members: Dict[int, Dict[str, Any]] = field(default_factory=dict)  # seg -> preset

    @property
    def complete(self) -> bool:
        return len(self.members) == self.count

    def label(self) -> str:
        """Display name, e.g. "Center [epixuhrhw, 2 segs, 2026-10-08 15:30Z]"."""
        ts: str = self.timestamp
        when: str = f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}Z"
        segs: str = f"{len(self.members)} segs"
        if not self.complete:
            segs = f"{len(self.members)}/{self.count} segs, incomplete"
        return f"{self.name} [{self.det_type}, {segs}, {when}]"


def group_presets(
    presets: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[GroupKey, PresetGroup]]:
    """Split a preset listing into plain presets and detector presets.

    Args:
        presets (List[Dict[str, Any]]): Preset documents (with "preset_name").

    Returns:
        plain (List[Dict[str, Any]]): Presets whose names do not decode, in order.

        groups (Dict[GroupKey, PresetGroup]): Detector presets by group key. The
            first preset seen for a segment wins if a group repeats a segment.
    """
    plain: List[Dict[str, Any]] = []
    groups: Dict[GroupKey, PresetGroup] = {}
    for preset in presets:
        decoded: Optional[GroupedPresetName] = decode_preset_name(
            preset.get("preset_name", "")
        )
        if decoded is None:
            plain.append(preset)
            continue

        key: GroupKey = (decoded.plugin, decoded.det_type, decoded.group)
        if key not in groups:
            groups[key] = PresetGroup(
                decoded.plugin,
                decoded.det_type,
                decoded.group,
                decoded.count,
                decoded.timestamp,
                decoded.name,
            )
        groups[key].members.setdefault(decoded.segment, preset)

    return plain, groups
