"""
Decode ROIBin 0.1.0 reduced output.

Output layout per event and segment (little-endian, as written by the GPU):
  - bytes[0,   16) -- header: uint64 bin, uint32 frames, uint32 flags

  - bytes[16, ...) -- float32 binned ROI, C-order (planes, rows, cols) with
                      shape roi_shape[i] // bin[i]

With time binning (frames > 1) only the last event of each time bin carries
output, and the others are empty.

An empty ROI will output just a header, if it is participating in time-binning.

Classes:
    RoiBinHeader: Decoded header.
    RoiBinOutput: Decoded header and binned ROI of one segment.

Functions:
    output_shape: Shape of the binned ROI for a parameter set.
    decode: Decode one segment's reduced bytes.
    panel_layout: The PanelLayout described by a parameter set.
    detector_geometry: Side-by-side geometry built from per-segment parameters.
    detector_image: Place binned ROIs in the detector frame at full resolution.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np

from drpalgos.geometry import (
    AsicPlacement,
    DetectorGeometry,
    PanelLayout,
    side_by_side,
)

HEADER_DTYPE: np.dtype = np.dtype([("bin", "<u8"), ("frames", "<u4"), ("flags", "<u4")])
"""Header as written by ROIBin 0.1.0 (16 bytes)."""
NO_BIN: int = 2**64 - 1
"""Header `bin` value when time binning is off (frames == 1)."""
FLAG_ORDER_SUSPECT: int = 1
"""Header flag: the frames summed into this bin were not staged in order."""


@dataclass(frozen=True)
class RoiBinHeader:
    bin: Optional[int]
    """The time-bin index as events since CONFIGURE, or None when frames == 1 (no time binning)"""
    frames: int
    """The number of frames summed into the bin"""
    flags: int
    """FLAG_* bits"""

    @property
    def order_suspect(self) -> bool:
        return bool(self.flags & FLAG_ORDER_SUSPECT)


@dataclass(frozen=True)
class RoiBinOutput:
    header: RoiBinHeader
    data: np.ndarray  # float32 (planes, rows, cols); a view of the input bytes


def output_shape(params: Mapping[str, Any]) -> Tuple[int, int, int]:
    """Shape of the binned ROI (zero-sized for an empty ROI)."""
    shape: Tuple[int, ...] = tuple(
        int(s) // int(b) for s, b in zip(params["roi_shape"], params["bin"])
    )

    return shape  # type: ignore[return-value]


def decode(reduced: Any, params: Mapping[str, Any]) -> Optional[RoiBinOutput]:
    """Decode one segment's reduced output for one event.

    Args:
        reduced (Any): The reduced bytes (1-D uint8 array, bytes or buffer).

        params (Mapping[str, Any]): The segment's ROIBin parameters.

    Returns:
        output (Optional[RoiBinOutput]): The decoded output, or None if the event
            carries no output (time binning, not the last event of a bin).

    Raises:
        ValueError: If the size does not match the parameters.
    """
    buf: np.ndarray = np.frombuffer(reduced, dtype=np.uint8)
    if buf.size == 0:
        return None

    shape: Tuple[int, int, int] = output_shape(params)
    expected: int = HEADER_DTYPE.itemsize + 4 * int(np.prod(shape))
    if buf.size != expected:
        raise ValueError(
            f"ROIBin 0.1.0 output is {buf.size} B; expected {expected} B for "
            f"roi_shape {list(params['roi_shape'])} and bin {list(params['bin'])}"
        )

    raw = buf[: HEADER_DTYPE.itemsize].view(HEADER_DTYPE)[0]
    header = RoiBinHeader(
        bin=None if int(raw["bin"]) == NO_BIN else int(raw["bin"]),
        frames=int(raw["frames"]),
        flags=int(raw["flags"]),
    )
    data: np.ndarray = buf[HEADER_DTYPE.itemsize :].view("<f4").reshape(shape)

    return RoiBinOutput(header, data)


def panel_layout(params: Mapping[str, Any]) -> PanelLayout:
    """The panel layout described by in_shape, asic_grid and asic_layout."""
    grid_rows: int
    grid_cols: int
    grid_rows, grid_cols = (int(v) for v in params["asic_grid"])

    rows: int = int(params["in_shape"][1])
    cols: int = int(params["in_shape"][2])
    flat: List[int] = [int(v) for v in params["asic_layout"]]
    placements: Tuple[AsicPlacement, ...] = tuple(
        AsicPlacement(flat[i], bool(flat[i + 1]), bool(flat[i + 2]))
        for i in range(0, len(flat), 3)
    )
    return PanelLayout(
        (rows // grid_rows, cols // grid_cols), (grid_rows, grid_cols), placements
    )


def detector_geometry(
    det_type: str, seg_params: Mapping[int, Mapping[str, Any]]
) -> DetectorGeometry:
    """Side-by-side geometry (as in the config GUI) from per-segment parameters.

    Raises:
        ValueError: If segments describe different panels.
    """
    layouts: Set[PanelLayout] = {panel_layout(p) for p in seg_params.values()}
    if len(layouts) != 1:
        raise ValueError("ROIBin segments describe different panel layouts")

    return side_by_side(det_type, layouts.pop(), seg_params)


def detector_image(
    outputs: Mapping[int, RoiBinOutput],
    seg_params: Mapping[int, Mapping[str, Any]],
    geom: DetectorGeometry,
    fill: float = np.nan,
) -> np.ndarray:
    """Place each segment's binned ROI in the detector frame at full resolution.

    Each binned value is repeated over the bin x bin pixels it covers. Pixels
    outside every ROI are `fill`. Only the first plane is used.

    Args:
        outputs (Mapping[int, RoiBinOutput]): Decoded output per segment.

        seg_params (Mapping[int, Mapping[str, Any]]): Parameters per segment.

        geom (DetectorGeometry): The detector geometry (see detector_geometry).

        fill (float): Value for pixels without data.

    Returns:
        image (np.ndarray): float32 image covering geom.bounds().
    """
    row0: int
    col0: int
    row1: int
    col1: int
    row0, col0, row1, col1 = geom.bounds()
    image: np.ndarray = np.full((row1 - row0, col1 - col0), fill, dtype=np.float32)
    for seg, out in outputs.items():
        params: Mapping[str, Any] = seg_params[seg]
        if out.data.size == 0:
            continue

        bin_row: int = int(params["bin"][1])
        bin_col: int = int(params["bin"][2])
        tile: np.ndarray = np.repeat(
            np.repeat(out.data[0], bin_row, axis=0), bin_col, axis=1
        )

        row_start: int = int(params["roi_start"][1])
        col_start: int = int(params["roi_start"][2])

        det_row0: int
        det_col0: int
        det_row1: int
        det_col1: int
        det_row0, det_col0, det_row1, det_col1 = geom.to_detector(
            seg,
            (
                row_start,
                col_start,
                row_start + tile.shape[0],
                col_start + tile.shape[1],
            ),
        )
        image[det_row0 - row0 : det_row1 - row0, det_col0 - col0 : det_col1 - col0] = (
            tile
        )
    return image


def seg_params_from_config(drp_alg: Any) -> Dict[str, Any]:
    """Plain dict of parameters from a config object (dict or attribute container)."""
    params: Any = (
        drp_alg["parameters"] if isinstance(drp_alg, Mapping) else drp_alg.parameters
    )
    items = params.items() if isinstance(params, Mapping) else vars(params).items()
    return {
        k: (v.tolist() if hasattr(v, "tolist") else v)
        for k, v in items
        if not k.startswith("_")
    }
