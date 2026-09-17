"""Panel texture synthesis and pinhole rendering of the mounting panel.

Texture convention: the texture is the panel as seen from the arm (board
frame: X right, Y down), one BGR pixel per ``1 / px_per_m`` metres, so
texture column ``i`` / row ``j`` covers board
``[i, i+1) / px_per_m  x  [j, j+1) / px_per_m``. Board ``(x, y)`` in metres
and continuous texture ``(u, v)`` therefore differ by the single scalar
``px_per_m`` (``board_to_texture`` / ``texture_to_board``), and drawing a
region ``[u0, u1)`` is the plain slice ``tex[:, u0:u1]``.

OpenCV's warp, ArUco and projection conventions put pixel *centres* at
integer coordinates (``CameraConfig.cx == (width - 1) / 2``), so the same
physical point has OpenCV pixel coordinate ``u - 0.5``. ``render`` applies
that half-pixel shift when it builds the homography (``texture_corners_cv``);
that is what makes a marker rasterised into pixels 24..103 land exactly on
the projection of its metric corners, so texture and truth agree by
construction (a fronto-parallel render at 1 texture px per image px is a
bit-exact copy of the texture).

Markers are upright: marker x along +X_B, marker y along -Y_B (ArUco has y
up), i.e. the generated marker image is pasted as-is into the y-down
texture. Consequently ArUco's detected corner order (TL, TR, BR, BL of the
printed marker) matches ``board.marker_corners_world``.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from autotype_sim.core.board import BoardGeometry, BoardPose
from autotype_sim.core.config import CameraConfig
from autotype_sim.core.keymap import TKL_HEIGHT_U, TKL_WIDTH_U, KeyMap
from autotype_sim.core.kinematics import project_points

DEFAULT_PX_PER_M = 4000.0
DEFAULT_MARKER_DICT = "DICT_4X4_50"
MARKER_BORDER_BITS = 1  # cv2.aruco default: one black module around the bits
MARKER_QUIET_MODULES = 1.0  # white quiet zone around each marker, in modules

# BGR colours of the synthetic panel (uint8).
BACKGROUND_BGR = (0, 0, 0)
QUIET_ZONE_BGR = (255, 255, 255)
PLATE_BGR = (38, 38, 38)
CAP_BGR = (78, 78, 78)
LABEL_BGR = (40, 40, 215)  # red-ish legends, like the real board

CAP_CORNER_RADIUS_M = 0.0015
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_WIDTH_FRAC = 0.80  # legend may use this fraction of the cap width ...
LABEL_HEIGHT_FRAC = 0.40  # ... and this fraction of the cap height
LABEL_MIN_PX = 3  # skip legends that would be shorter than this

# detect_markers corner-refinement choices (cv2.aruco.CORNER_REFINE_*).
REFINE_METHODS = {
    "none": cv2.aruco.CORNER_REFINE_NONE,
    "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
    "contour": cv2.aruco.CORNER_REFINE_CONTOUR,
    "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG,
}
DEFAULT_REFINE = "contour"


# --------------------------------------------------------------------------- #
# Board <-> texture coordinates
# --------------------------------------------------------------------------- #


def board_to_texture(xy_m: np.ndarray, px_per_m: float) -> np.ndarray:
    """Board metres -> continuous texture pixels (pure scale).

    Texture pixel ``i`` spans ``[i, i+1)``; subtract 0.5 for OpenCV's
    pixel-centre convention.
    """
    return np.asarray(xy_m, dtype=float) * float(px_per_m)


def texture_to_board(uv_px: np.ndarray, px_per_m: float) -> np.ndarray:
    """Continuous texture pixels (pixel ``i`` spans ``[i, i+1)``) -> board metres."""
    return np.asarray(uv_px, dtype=float) / float(px_per_m)


def texture_size(geometry: BoardGeometry, px_per_m: float) -> tuple[int, int]:
    """``(width_px, height_px)`` of the panel texture: ``round(panel_* * px_per_m)``."""
    return (
        int(round(geometry.panel_w * float(px_per_m))),
        int(round(geometry.panel_h * float(px_per_m))),
    )


def texture_corners_cv(texture: np.ndarray) -> np.ndarray:
    """Panel corners TL, TR, BR, BL in OpenCV pixel-centre coordinates, float32 (4, 2).

    The texture's physical edges are the outer edges of its border pixels,
    i.e. -0.5 and ``W - 0.5`` / ``H - 0.5`` once pixel centres sit on integers.
    """
    h, w = texture.shape[:2]
    return np.array(
        [[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]],
        dtype=np.float32,
    )


# --------------------------------------------------------------------------- #
# ArUco helpers
# --------------------------------------------------------------------------- #


def aruco_dictionary(name: str = DEFAULT_MARKER_DICT):
    """``cv2.aruco`` predefined dictionary from its name, e.g. ``DICT_4X4_50``."""
    code = getattr(cv2.aruco, str(name), None)
    if not isinstance(code, int):
        raise ValueError(f"unknown ArUco dictionary {name!r}")
    getter = getattr(cv2.aruco, "getPredefinedDictionary", None)
    if getter is None:  # pre-4.7 API
        getter = cv2.aruco.Dictionary_get
    return getter(code)


def _marker_image(dictionary, marker_id: int, side_px: int) -> np.ndarray:
    """Upright ``side_px x side_px`` uint8 marker image (0 = black, 255 = white)."""
    generate = getattr(cv2.aruco, "generateImageMarker", None)
    if generate is None:  # pre-4.7 API
        generate = cv2.aruco.drawMarker
    return np.asarray(generate(dictionary, int(marker_id), int(side_px)), dtype=np.uint8)


def _modules_per_side(dictionary) -> int:
    """Modules across a marker including its border: bits + 2 * border (6 for 4x4)."""
    return int(getattr(dictionary, "markerSize", 4)) + 2 * MARKER_BORDER_BITS


# --------------------------------------------------------------------------- #
# Raster primitives (integer texture pixels, half-open [x0, x1) x [y0, y1))
# --------------------------------------------------------------------------- #


def _fill_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color) -> None:
    """Fill the half-open pixel box, clipped to the image."""
    h, w = img.shape[:2]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x0 < x1 and y0 < y1:
        img[y0:y1, x0:x1] = color


def _paste(img: np.ndarray, patch: np.ndarray, x0: int, y0: int) -> None:
    """Copy a 2-D or BGR patch with its top-left at (x0, y0), clipped to the image."""
    h, w = img.shape[:2]
    ph, pw = patch.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = min(pw, w - x0), min(ph, h - y0)
    if sx0 >= sx1 or sy0 >= sy1:
        return
    src = patch[sy0:sy1, sx0:sx1]
    if src.ndim == 2:
        src = src[:, :, np.newaxis]
    img[y0 + sy0 : y0 + sy1, x0 + sx0 : x0 + sx1] = src


def _rounded_rect(
    img: np.ndarray, x0: int, y0: int, x1: int, y1: int, radius: int, color
) -> None:
    """Filled rounded rectangle over the half-open pixel box."""
    if x1 - x0 < 1 or y1 - y0 < 1:
        return
    r = int(max(0, min(radius, (x1 - x0 - 1) // 2, (y1 - y0 - 1) // 2)))
    # cv2.rectangle is inclusive of both corners.
    cv2.rectangle(img, (x0 + r, y0), (x1 - 1 - r, y1 - 1), color, cv2.FILLED)
    cv2.rectangle(img, (x0, y0 + r), (x1 - 1, y1 - 1 - r), color, cv2.FILLED)
    if r > 0:
        for cx in (x0 + r, x1 - 1 - r):
            for cy in (y0 + r, y1 - 1 - r):
                cv2.circle(img, (cx, cy), r, color, cv2.FILLED)


def _draw_label(img: np.ndarray, text: str, x0: int, y0: int, x1: int, y1: int) -> None:
    """Centre ``text`` in the cap box, scaled to LABEL_*_FRAC of the cap."""
    w, h = x1 - x0, y1 - y0
    (tw, th), _ = cv2.getTextSize(text, LABEL_FONT, 1.0, 1)
    if tw <= 0 or th <= 0:
        return
    scale = min(LABEL_WIDTH_FRAC * w / tw, LABEL_HEIGHT_FRAC * h / th)
    if scale * th < LABEL_MIN_PX:
        return
    thickness = max(1, int(round(1.6 * scale)))
    (tw, th), _ = cv2.getTextSize(text, LABEL_FONT, scale, thickness)
    org = (x0 + (w - tw) // 2, y0 + (h + th) // 2)
    cv2.putText(img, text, org, LABEL_FONT, scale, LABEL_BGR, thickness, cv2.LINE_AA)


def _as_bgr(photo: np.ndarray) -> np.ndarray:
    """Coerce a grey / BGR / BGRA array to (h, w, 3) uint8 BGR.

    Accepts a non-empty ``(h, w)``, ``(h, w, 3)`` or ``(h, w, 4)`` array of
    any integer or floating dtype with finite values; non-uint8 values are
    clipped to 0..255. Everything else (empty, bool, object, NaN / inf, other
    shapes) is a ``ValueError`` -- never a raw ``cv2.error`` and never a
    silently black plate.
    """
    photo = np.asarray(photo)
    if photo.ndim not in (2, 3) or (photo.ndim == 3 and photo.shape[2] not in (3, 4)):
        raise ValueError(
            f"photo must be an (h, w) grey, (h, w, 3) BGR or (h, w, 4) BGRA image, "
            f"got shape {photo.shape}"
        )
    if photo.shape[0] < 1 or photo.shape[1] < 1:
        raise ValueError(f"photo must not be empty, got shape {photo.shape}")
    if photo.dtype != np.uint8:
        is_float = np.issubdtype(photo.dtype, np.floating)
        if not (is_float or np.issubdtype(photo.dtype, np.integer)):
            raise ValueError(f"photo must have an integer or floating dtype, got {photo.dtype}")
        if is_float and not bool(np.all(np.isfinite(photo))):
            raise ValueError("photo contains NaN or inf values")
        photo = np.clip(photo, 0, 255).astype(np.uint8)
    if photo.ndim == 2:
        photo = cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR)
    elif photo.shape[2] == 4:
        photo = cv2.cvtColor(photo, cv2.COLOR_BGRA2BGR)
    return photo


# --------------------------------------------------------------------------- #
# Texture
# --------------------------------------------------------------------------- #


def _draw_marker(
    tex: np.ndarray, geometry: BoardGeometry, dictionary, marker_id: int, px_per_m: float
) -> None:
    """Paste marker ``marker_id`` upright, centred at its board centre, inside a
    white quiet zone >= MARKER_QUIET_MODULES modules (module = marker_size / 6).

    Raises ``ValueError`` when that quiet zone would run off the texture edge
    (``BoardGeometry`` only requires the marker itself to fit): a clipped quiet
    zone merges the black marker border with the black panel and silently
    breaks detection, which the design spec s.9 forbids.
    """
    modules = _modules_per_side(dictionary)
    side_px = int(round(geometry.marker_size * px_per_m))
    if side_px < modules:
        raise ValueError(
            f"px_per_m={px_per_m} gives a {side_px} px marker; need >= {modules} px "
            f"(one pixel per module)"
        )
    quiet = int(math.ceil(MARKER_QUIET_MODULES * side_px / modules))
    cx, cy = board_to_texture(geometry.marker_centers[marker_id], px_per_m)
    x0 = int(round(cx - side_px / 2.0))
    y0 = int(round(cy - side_px / 2.0))
    qx0, qy0 = x0 - quiet, y0 - quiet
    qx1, qy1 = x0 + side_px + quiet, y0 + side_px + quiet
    tex_h, tex_w = tex.shape[:2]
    if qx0 < 0 or qy0 < 0 or qx1 > tex_w or qy1 > tex_h:
        need_m = (side_px / 2.0 + quiet) / px_per_m
        raise ValueError(
            f"marker {marker_id} centred at {tuple(geometry.marker_centers[marker_id])} m "
            f"sits too close to a panel edge for its {MARKER_QUIET_MODULES:g}-module white "
            f"quiet zone: at {px_per_m:g} px/m the centre must be >= {need_m:.4f} m "
            f"(marker_size / 2 + one module) from every edge"
        )
    _fill_rect(tex, qx0, qy0, qx1, qy1, QUIET_ZONE_BGR)
    _paste(tex, _marker_image(dictionary, marker_id, side_px), x0, y0)


def key_area_box_px(geometry: BoardGeometry, px_per_m: float) -> tuple[int, int, int, int]:
    """``(x0, y0, x1, y1)`` texture pixels of the TKL_WIDTH_U x TKL_HEIGHT_U key
    grid at ``geometry.key_area_origin`` -- the rectangle the photo fills."""
    ox, oy = geometry.key_area_origin
    gx0, gy0 = (int(round(v)) for v in board_to_texture((ox, oy), px_per_m))
    gx1, gy1 = (
        int(round(v))
        for v in board_to_texture(
            (ox + TKL_WIDTH_U * geometry.key_pitch, oy + TKL_HEIGHT_U * geometry.key_pitch),
            px_per_m,
        )
    )
    return gx0, gy0, gx1, gy1


def _draw_keyboard(
    tex: np.ndarray,
    geometry: BoardGeometry,
    keymap: KeyMap,
    px_per_m: float,
    photo: np.ndarray | None,
) -> None:
    """Keyboard plate rectangle, then the key grid on top of it: either the
    resized ``photo`` or synthetic caps drawn from the KeyMap's registration
    rectangles (so texture == truth either way).

    ``photo`` is the key grid alone -- TKL_WIDTH_U x TKL_HEIGHT_U key units,
    no bezel -- and is stretched onto exactly that rectangle, so a keycap in
    the image lands on its analytic key cell whatever the plate around it
    measures. The bezel is flat ``PLATE_BGR``. Keeping the photo bezel-free
    is deliberate: a plate-sized crop would encode the plate size and bezel
    (which are site configuration) in its own aspect ratio.
    """
    ox, oy = geometry.kb_origin
    px0, py0 = (int(round(v)) for v in board_to_texture((ox, oy), px_per_m))
    px1, py1 = (
        int(round(v))
        for v in board_to_texture((ox + geometry.kb_w, oy + geometry.kb_h), px_per_m)
    )
    if px1 - px0 < 1 or py1 - py0 < 1:
        return
    _fill_rect(tex, px0, py0, px1, py1, PLATE_BGR)

    gx0, gy0, gx1, gy1 = key_area_box_px(geometry, px_per_m)
    gw, gh = gx1 - gx0, gy1 - gy0

    if photo is not None:
        photo = _as_bgr(photo)  # validates before the degenerate-grid bail-out
        if gw < 1 or gh < 1:
            return
        shrinking = photo.shape[1] >= gw and photo.shape[0] >= gh
        interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        _paste(tex, cv2.resize(photo, (gw, gh), interpolation=interp), gx0, gy0)
        return

    radius = int(round(CAP_CORNER_RADIUS_M * px_per_m))
    for key in keymap:
        kx0, ky0 = (int(round(v)) for v in board_to_texture((key.x0, key.y0), px_per_m))
        kx1, ky1 = (int(round(v)) for v in board_to_texture((key.x1, key.y1), px_per_m))
        _rounded_rect(tex, kx0, ky0, kx1, ky1, radius, CAP_BGR)
        _draw_label(tex, key.name, kx0, ky0, kx1, ky1)


def build_texture(
    geometry: BoardGeometry,
    keymap: KeyMap,
    px_per_m: float = DEFAULT_PX_PER_M,
    photo: np.ndarray | None = None,
) -> np.ndarray:
    """BGR uint8 image of the whole panel, ``round(panel_h*ppm) x round(panel_w*ppm)``.

    Black background; keyboard plate, carrying either synthetic caps from
    ``keymap`` or the resized ``photo`` of the bare key grid; then each
    ArUco marker of ``geometry.marker_dict``
    upright at its centre inside a white quiet zone >= 1 module. Markers are
    drawn last so the quiet zone is never encroached on by the plate.

    Raises ``ValueError`` for a non-finite / non-positive ``px_per_m``, a
    ``px_per_m`` too small for one pixel per marker module, a marker whose
    quiet zone would be clipped by the panel edge, or a degenerate ``photo``.
    """
    px_per_m = float(px_per_m)
    if not (math.isfinite(px_per_m) and px_per_m > 0.0):
        raise ValueError(f"px_per_m must be a positive finite number, got {px_per_m}")
    w, h = texture_size(geometry, px_per_m)
    if w < 1 or h < 1:
        raise ValueError(f"px_per_m={px_per_m} gives an empty {w}x{h} texture")

    tex = np.zeros((h, w, 3), dtype=np.uint8)
    if BACKGROUND_BGR != (0, 0, 0):
        tex[:] = BACKGROUND_BGR
    _draw_keyboard(tex, geometry, keymap, px_per_m, photo)
    dictionary = aruco_dictionary(geometry.marker_dict)
    for marker_id in sorted(geometry.marker_centers):
        _draw_marker(tex, geometry, dictionary, marker_id, px_per_m)
    return tex


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #


def render(
    cam: CameraConfig,
    R_cam: np.ndarray,
    t_cam: np.ndarray,
    board_pose: BoardPose,
    texture: np.ndarray,
    geometry: BoardGeometry,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Camera image ``(cam.height, cam.width, 3)`` uint8 of the textured panel.

    Projects the panel corners TL, TR, BR, BL (``BoardPose.corners_world``);
    if any has ``P_C.z <= 0`` the frame is all black. Otherwise the
    homography from the texture's OpenCV-convention corners to those image
    points is applied with ``cv2.warpPerspective`` (INTER_LINEAR, black
    border). ``out`` is an optional ``(cam.height, cam.width, 3)`` uint8
    buffer to write into (and return) so a streaming loop allocates nothing
    per frame; without it the only allocation is the output image. Anything
    else passed as ``out`` (wrong shape or dtype, or pixels not packed within
    rows, which OpenCV cannot write into) raises ``ValueError`` on every
    frame -- the returned shape never depends on the pose. Row-strided views
    and sub-windows of a larger packed buffer are fine.
    """
    if out is not None and (
        not isinstance(out, np.ndarray)
        or out.shape != (cam.height, cam.width, 3)
        or out.dtype != np.uint8
        or tuple(out.strides[1:]) != (3, 1)
    ):
        raise ValueError(
            f"out must be a ({cam.height}, {cam.width}, 3) uint8 array with packed pixels, "
            f"got {type(out).__name__} shape={getattr(out, 'shape', None)} "
            f"dtype={getattr(out, 'dtype', None)} strides={getattr(out, 'strides', None)}"
        )
    uv, z = project_points(cam, R_cam, t_cam, board_pose.corners_world(geometry))
    if not bool(np.all(z > 0.0)):
        if out is None:
            return np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
        out[...] = 0
        return out
    H = cv2.getPerspectiveTransform(texture_corners_cv(texture), uv.astype(np.float32))
    return cv2.warpPerspective(
        texture,
        H,
        (cam.width, cam.height),
        dst=out,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


# --------------------------------------------------------------------------- #
# Detection (test / dashboard utility)
# --------------------------------------------------------------------------- #


def detect_markers(
    image: np.ndarray,
    marker_dict: str = DEFAULT_MARKER_DICT,
    *,
    refine: str = DEFAULT_REFINE,
) -> dict[int, np.ndarray]:
    """ArUco detection for tests and the dashboard -- NOT something the
    simulator publishes; members bring their own perception.

    Uses ``cv2.aruco.ArucoDetector`` (4.7+), falling back to
    ``cv2.aruco.detectMarkers``. Returns ``{id: corners[4, 2]}`` (float64,
    OpenCV pixel-centre coordinates) in the printed marker's TL, TR, BR, BL
    order; the first detection wins if an id repeats.

    ``refine`` is one of ``REFINE_METHODS`` (``"none"`` is OpenCV's stock
    default). The default ``"contour"`` was chosen on the rendered rig, where
    a marker is only ~20 px (3.4 px/module): over 200 sampled poses it had
    the smallest worst-case corner error (1.42 px) and never dropped a
    marker, whereas ``"subpix"`` had the best mean but a 2.3 px tail and
    ``"apriltag"`` dropped markers.
    """
    try:
        method = REFINE_METHODS[refine]
    except KeyError:
        raise ValueError(f"refine must be one of {sorted(REFINE_METHODS)}, got {refine!r}")

    image = np.asarray(image)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary = aruco_dictionary(marker_dict)

    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = method
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(gray)
    else:  # pre-4.7 API
        params = cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = method
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

    found: dict[int, np.ndarray] = {}
    if ids is None:
        return found
    for c, marker_id in zip(corners, np.asarray(ids).reshape(-1)):
        found.setdefault(int(marker_id), np.asarray(c, dtype=float).reshape(4, 2))
    return found
