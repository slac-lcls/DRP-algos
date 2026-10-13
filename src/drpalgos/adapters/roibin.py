"""
Adapter for ROIBin: detector-level ROIs to per-segment roi_start/roi_shape.

The GUI plugin intent (of intent_type `roi`) returns a ROIs, with half-open
detector frame coordinates.

The following rules are used:
- The bounding rectangles described by each ROI are clipped to a panel, which then
  becomes that segment'sROI.
- An ROI that moves between segments will restart after the clipping at the corner
  of the intersection.
- ASIC boundaries are ignored.

NOTE: Every segment that gets configured will get a set of parameters returned. This
      includes segments that have empty ROIs (ROIBin handles this - it returns the
      header-only data for empty ROIs when time binning is employed. That's required
      to ensure that the syncrhonziation is maintained).

Intent: (From the plugin)
    {"rois": [[r0, c0, r1, c1], ...]}
    - Return a list of half-open ROIs

The adapter retunrs the following output:
  - `in_shape`    = [1, panel_rows, panel_cols]
  - `roi_start`   = [0, row, col] (or [0,0,0] for an empty ROI)
  - `roi_shape`   = [1, height, width] (or [0,0,0] for an empty ROI)
  - `asic_grid`   = [grid_rows, grid_cols]
  - `asic_layout` = [asic, flip_rows, flip_cols, ...]

asic_grid and asic_layout are used for repositioning and rempaping the pixels from
ASIC major to normal row-major configuration. When FPGA output is changed to the module
format this will no longer be needed.

Classes:
    RoiBinV010: Adapter for ROIBin 0.1.0.
"""

from typing import Any, Dict, List, Optional, Tuple

from drpalgos.adapters import ParamAdapter
from drpalgos.geometry import (
    DetectorGeometry,
    PanelLayout,
    Rect,
    intersect,
)

_EMPTY_ROI: Dict[str, List[int]] = {"roi_start": [0, 0, 0], "roi_shape": [0, 0, 0]}


def _parse_roi(roi: Any) -> Optional[Rect]:
    """Return roi as a Rect if it is 4 integers with positive extent, else None."""
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        return None
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in roi):
        return None
    r0, c0, r1, c1 = roi
    return (r0, c0, r1, c1) if r1 > r0 and c1 > c0 else None


def _parse_bin(shared: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Return the (rows, cols) pixel bin from shared["bin"], or None if invalid.

    A missing bin means no binning.
    """
    b: Any = shared.get("bin", [1, 1, 1])
    if not isinstance(b, (list, tuple)) or len(b) != 3:
        return None
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in b):
        return None
    if b[0] != 1:
        return None
    return (b[1], b[2])


class RoiBinV010(ParamAdapter):
    """ROIBin version 0.1.0 adapter

    Current restrictions are:
    - 1 ROI per panel.
    - ROIs are snapped to pixel bin boundaries (no partial pixels)
    """

    plugin = "roi"
    """The plugin MUST match any `intent_type` variable in a GUI plugin"""

    produces = ("in_shape", "roi_start", "roi_shape", "asic_grid", "asic_layout")
    detector_scope = ("frames", "bin", "threshold", "use_threshold", "scale")
    """Exposed parameters."""

    def _pieces(
        self, rois: List[Rect], geom: DetectorGeometry, segments: List[int]
    ) -> Dict[int, List[Rect]]:
        """Clipped pieces of each ROI per segment, in panel coordinates."""
        pieces: Dict[int, List[Rect]] = {}
        for seg in sorted(segments):
            panel: Rect = geom.panel_rect(seg)
            for roi in rois:
                hit: Optional[Rect] = intersect(roi, panel)
                if hit is not None:
                    pieces.setdefault(seg, []).append(geom.to_panel(seg, hit))
        return pieces

    @staticmethod
    def _snap(piece: Rect, bin_rc: Tuple[int, int]) -> Rect:
        """Shrink a piece's height and width down to multiples of the bin."""
        row0, col0, row1, col1 = piece
        height: int = (row1 - row0) // bin_rc[0] * bin_rc[0]
        width: int = (col1 - col0) // bin_rc[1] * bin_rc[1]
        return (row0, col0, row0 + height, col0 + width)

    def check(
        self,
        intent: Dict[str, Any],
        geom: DetectorGeometry,
        segments: List[int],
        shared: Dict[str, Any],
    ) -> List[str]:
        """Verify that an intent can be converted to per-segment parameters"""
        problems: List[str] = []

        known: List[int] = geom.segment_ids()
        unknown: List[int] = [s for s in segments if s not in known]
        if unknown:
            problems.append(
                f"Error: segments {unknown} are not in the {geom.det_type} geometry"
            )
        segs: List[int] = [s for s in segments if s in known]

        bin_rc: Optional[Tuple[int, int]] = _parse_bin(shared)
        if bin_rc is None:
            problems.append(
                f"Error: bin must be 3 positive integers with bin[0] == 1, "
                f"got {shared.get('bin')}"
            )

        raw: Any = intent.get("rois", [])
        if not isinstance(raw, (list, tuple)):
            return problems + ["Error: intent 'rois' must be a list"]
        rois: List[Rect] = []
        for i, roi in enumerate(raw):
            rect: Optional[Rect] = _parse_roi(roi)
            if rect is None:
                problems.append(
                    f"Error: ROI {i} {roi} is not [r0, c0, r1, c1] "
                    f"with r1 > r0 and c1 > c0"
                )
            else:
                rois.append(rect)

        for i, roi in enumerate(rois):
            if all(intersect(roi, geom.panel_rect(s)) is None for s in segs):
                problems.append(
                    f"ROI {list(roi)} does not overlap any configured panel "
                    f"and is ignored"
                )

        pieces: Dict[int, List[Rect]] = self._pieces(rois, geom, segs)
        for seg, seg_pieces in pieces.items():
            if len(seg_pieces) > 1:
                problems.append(
                    f"Error: segment {seg} is overlapped by {len(seg_pieces)} ROIs; "
                    f"ROIBin 0.1.0 supports one ROI per panel"
                )
            elif bin_rc is not None:
                piece: Rect = seg_pieces[0]
                snapped: Rect = self._snap(piece, bin_rc)
                if snapped[2] == snapped[0] or snapped[3] == snapped[1]:
                    problems.append(
                        f"Segment {seg}: ROI {list(piece)} is smaller than the bin "
                        f"{list(bin_rc)}; segment gets an empty ROI (header only)"
                    )
                elif snapped != piece:
                    problems.append(
                        f"Segment {seg}: ROI {list(piece)} shrunk to "
                        f"{list(snapped)} to fit the bin {list(bin_rc)}"
                    )

        empty: List[int] = [s for s in segs if s not in pieces]
        if empty:
            problems.append(
                f"Segments {empty} have no ROI and will output headers only "
                f"(uncheck them to leave their configuration unchanged)"
            )

        return problems

    def partition(
        self,
        intent: Dict[str, Any],
        geom: DetectorGeometry,
        segments: List[int],
        shared: Dict[str, Any],
    ) -> Dict[int, Dict[str, Any]]:
        """Construct the per-segment parameter sets from the intent"""
        errors: List[str] = [
            prob
            for prob in self.check(intent, geom, segments, shared)
            if prob.startswith("Error:")
        ]
        if errors:
            raise ValueError("; ".join(errors))

        bin_rc: Tuple[int, int] = _parse_bin(shared)  # type: ignore[assignment]
        rois: List[Rect] = [_parse_roi(r) for r in intent.get("rois", [])]  # type: ignore[misc]
        pieces: Dict[int, List[Rect]] = self._pieces(rois, geom, segments)
        panel: Dict[str, List[int]] = self._panel_params(geom.panel)

        out: Dict[int, Dict[str, Any]] = {}
        for seg in sorted(segments):
            roi: Dict[str, List[int]] = dict(_EMPTY_ROI)
            if seg in pieces:
                r0, c0, r1, c1 = self._snap(pieces[seg][0], bin_rc)
                if r1 > r0 and c1 > c0:
                    roi = {"roi_start": [0, r0, c0], "roi_shape": [1, r1 - r0, c1 - c0]}
            out[seg] = {**panel, **roi}

        return out

    @staticmethod
    def _panel_params(panel: PanelLayout) -> Dict[str, List[int]]:
        """Panel shape and ASIC tiling parameters, identical for every segment."""
        rows, cols = panel.shape
        layout: List[int] = []
        for p in panel.placements:
            layout += [p.asic, int(p.flip_rows), int(p.flip_cols)]
        return {
            "in_shape": [1, rows, cols],
            "asic_grid": list(panel.grid),
            "asic_layout": layout,
        }

    def assemble(
        self, seg_params: Dict[int, Dict[str, Any]], geom: DetectorGeometry
    ) -> Dict[str, Any]:
        rois: List[List[int]] = []
        for seg in sorted(seg_params):
            params: Dict[str, Any] = seg_params[seg]
            _, row0, col0 = params["roi_start"]
            _, height, width = params["roi_shape"]
            if height == 0 or width == 0:
                continue  # Empty ROI
            rois.append(
                list(geom.to_detector(seg, (row0, col0, row0 + height, col0 + width)))
            )

        return {"rois": rois}
