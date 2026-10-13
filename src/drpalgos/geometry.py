"""
A minimal detector geometry for use in algorithm configuration.

Currently the geometry is pixel-accurate with a single detector segment. It makes
no attempt to place and orient various detector segments with respect to each other.
As a result, the geometry is light and carries no dependencies on psana, Qt etc.

Each detector segement/panel is described as a single contiguous 2D image using
"panel coordinates". Each panel sites in a "detector frame" to allow for different
plugin intents that may require working across multiple panels (e.g. drawing ROIs
spanning multiple panels).

Classes:
    AsicPlacement: Where one ASIC sits within a panel's ASIC grid.
    PanelLayout: Shape of a panel and how panel pixels map to the ASIC-major buffer.
    SegmentGeom: Placement of one segment's panel in the detector frame.
    DetectorGeometry: All segments of one detector.

Functions:
    side_by_side: Default arrangement, panels left to right in segment order.
    get_geometry: Geometry for a detector type and set of segments.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

Rect = Tuple[
    int, int, int, int
]  # (r0, c0, r1, c1), half-open: rows [r0, r1), cols [c0, c1)


@dataclass(frozen=True)
class AsicPlacement:
    """ASIC Geometry in a multi-ASIC detector panel.

    This is of main importance when data returned from a FPGA is not natively
    panel-shaped. The AsicPlacement is used to map between panel indices and
    ASIC indices.

    NOTE: This will likely (hopefully?) change at some point soonish when the
          firmware is updated to return data in module shape (row-major) instead
          of its current ASIC-major outpu.
    """

    asic: int
    """Index of the ASIC in an ASIC-major buffer."""
    flip_rows: bool
    """Rows in the panel run opposite to those in the ASIC."""
    flip_cols: bool
    """Columns in the panel run opposite to those in the ASIC."""


@dataclass(frozen=True)
class PanelLayout:
    """Layout a series of ASICs in a grid on a single detector panel."""

    asic_shape: Tuple[int, int]
    """Shape of one asic as (rows, columns)"""
    grid: Tuple[int, int]
    """Grid of ASICs in the panel as (rows, columns)."""
    placements: Tuple[AsicPlacement, ...]
    """Placement information, with one entry per grid position (row-major)."""

    def __post_init__(self) -> None:
        if len(self.placements) != self.grid[0] * self.grid[1]:
            raise ValueError("PanelLayout: need one placement per grid position")

    @property
    def shape(self) -> Tuple[int, int]:
        """Panel shape in panel coordinates (rows, cols)."""
        return (self.grid[0] * self.asic_shape[0], self.grid[1] * self.asic_shape[1])

    @property
    def n_asics(self) -> int:
        return self.grid[0] * self.grid[1]

    def to_buffer(self, row: int, col: int) -> Tuple[int, int, int]:
        """Map a panel pixel (row, col) to its (asic, y, x) index in the ASIC-major buffer.

        Args:
            row (int): Panel row, 0 <= row < shape[0].

            col (int): Panel column, 0 <= col < shape[1].

        Returns:
            index (Tuple[int, int, int]): The (asic, y, x) index of the pixel.
        """
        asic_rows, asic_cols = self.asic_shape
        div_r, mod_r = divmod(row, asic_rows)
        div_c, mod_c = divmod(col, asic_cols)

        p: AsicPlacement = self.placements[div_r * self.grid[1] + div_c]
        y: int = asic_rows - 1 - mod_r if p.flip_rows else mod_r
        x: int = asic_cols - 1 - mod_c if p.flip_cols else mod_c

        return (p.asic, y, x)


@dataclass(frozen=True)
class SegmentGeom:
    """Geometry holder for an individual detector segment.

    This should correspond to a PanelLayout. In the future, the ability to orient
    the segments with respect to each other may be supported - so fields are resevered
    for this purpose but do nothing at the moment.
    """

    segment: int
    """Segment number (e.g. the 1 in epixuhr3x2_1)"""
    origin: Tuple[int, int]
    """(Row, column) index of the panel's top-left in the detector frame."""
    rotation: int = 0
    """Reserved for future use. Only 0 is supported for now."""

    def __post_init__(self) -> None:
        if self.rotation != 0:
            raise NotImplementedError("SegmentGeom: only rotation 0 is supported")


@dataclass(frozen=True)
class DetectorGeometry:
    det_type: str
    panel: PanelLayout
    segments: Tuple[SegmentGeom, ...]

    def segment_ids(self) -> List[int]:
        return [s.segment for s in self.segments]

    def _seg(self, segment: int) -> SegmentGeom:
        for s in self.segments:
            if s.segment == segment:
                return s
        raise KeyError(f"Segment {segment} not in {self.det_type} geometry")

    def panel_rect(self, segment: int) -> Rect:
        """A segment's panel in detector-frame coordinates."""
        row0: int
        col0: int
        row0, col0 = self._seg(segment).origin

        rows: int
        cols: int
        rows, cols = self.panel.shape

        return (row0, col0, row0 + rows, col0 + cols)

    def to_panel(self, segment: int, rect: Rect) -> Rect:
        """Convert a detector-frame rectangle to the segment's panel coordinates."""
        row0: int
        col0: int
        row0, col0 = self._seg(segment).origin

        return (rect[0] - row0, rect[1] - col0, rect[2] - row0, rect[3] - col0)

    def to_detector(self, segment: int, rect: Rect) -> Rect:
        """Convert a rectangle in a segment's panel coordinates to the detector frame."""
        row0: int
        col0: int
        row0, col0 = self._seg(segment).origin

        return (rect[0] + row0, rect[1] + col0, rect[2] + row0, rect[3] + col0)

    def bounds(self) -> Rect:
        """Bounding rectangle of all panels in the detector frame."""
        rects: List[Rect] = [self.panel_rect(s.segment) for s in self.segments]
        return (
            min(r[0] for r in rects),
            min(r[1] for r in rects),
            max(r[2] for r in rects),
            max(r[3] for r in rects),
        )


def intersect(rect_a: Rect, rect_b: Rect) -> Optional[Rect]:
    """Intersection of two half-open rectangles, or None if they do not overlap."""
    row0: int
    col0: int
    row0, col0 = max(rect_a[0], rect_b[0]), max(rect_a[1], rect_b[1])

    row1: int
    col1: int
    row1, col1 = min(rect_a[2], rect_b[2]), min(rect_a[3], rect_b[3])

    return (row0, col0, row1, col1) if row0 < row1 and col0 < col1 else None


def side_by_side(
    det_type: str, panel: PanelLayout, segments: Iterable[int], gap: int = 16
) -> DetectorGeometry:
    """Arrange detector segment panels left to right in segment order with small gaps.

    This is the default arrangement for now. In the future, the real relative
    placement of the panels (incorporating geometry) may be supported. But is not
    included for the moment.

    The ROIs have to be decomposed by panel anyway (what the DRP receives), so this
    is a purely visualization related issue.
    """
    cols: int = panel.shape[1]
    segs: Tuple[SegmentGeom, ...] = tuple(
        SegmentGeom(seg, (0, i * (cols + gap)))
        for i, seg in enumerate(sorted(set(segments)))
    )
    return DetectorGeometry(det_type, panel, segs)


# Taken from Mikhail's stack_3x2_asics in psana
# - Top row (ASICs A0-A2) is flipped left-right
# - Bottom row (ASICs A3-A5) is flipped up-down.
EPIXUHR_3X2_PANEL: PanelLayout = PanelLayout(
    asic_shape=(168, 192),
    grid=(2, 3),
    placements=(
        AsicPlacement(0, False, True),
        AsicPlacement(1, False, True),
        AsicPlacement(2, False, True),
        AsicPlacement(3, True, False),
        AsicPlacement(4, True, False),
        AsicPlacement(5, True, False),
    ),
)


PANEL_LAYOUTS: Dict[str, PanelLayout] = {
    "epixuhrhw": EPIXUHR_3X2_PANEL,
    "epixuhr": EPIXUHR_3X2_PANEL,  # Used with epixuhremu and fakesim tdet
}
"""Mapping of detType to the geometry description."""


def get_geometry(
    det_type: str, segments: Iterable[int], det_config: Optional[Dict[str, Any]] = None
) -> DetectorGeometry:
    """Geometry for a detector type and the segments being configured.

    Args:
        det_type (str): The detector type (detType:RO).

        segments (Iterable[int]): The segment numbers to include.

        det_config (Optional[Dict[str, Any]]): Reserved for configuration-dependent
            shapes. Currently unused.

    Returns:
        geom (DetectorGeometry): The geometry.

    Raises:
        KeyError: If no panel layout is known for det_type.
    """
    if det_type not in PANEL_LAYOUTS:
        raise KeyError(f"No panel layout known for detector type '{det_type}'")
    return side_by_side(det_type, PANEL_LAYOUTS[det_type], segments)
