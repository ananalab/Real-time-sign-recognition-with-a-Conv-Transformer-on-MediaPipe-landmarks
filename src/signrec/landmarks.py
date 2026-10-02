"""Unified landmark subset shared by every dataset, the trainer and the demo.

Order is frozen (tested): left hand (21) | right hand (21) | upper body (7) | lips (40).
Indices refer to the MediaPipe Holistic numbering of each part
(hands: 21-point hand model, pose: 33-point BlazePose, face: 468/478-point face mesh).
"""

from __future__ import annotations

from typing import Final

# MediaPipe hand model: wrist, then 4 joints per finger (thumb to pinky).
HAND_IDX: Final[tuple[int, ...]] = tuple(range(21))

# BlazePose upper body: nose, shoulders, elbows, wrists. Hips are excluded: they are
# usually out of frame in webcam and LSFB recordings (y > 1).
POSE_IDX: Final[tuple[int, ...]] = (0, 11, 12, 13, 14, 15, 16)
POSE_NAMES: Final[tuple[str, ...]] = (
    "nose",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
)

# Face-mesh lip contour (outer then inner ring), 40 points.
LIPS_IDX: Final[tuple[int, ...]] = (
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
)  # fmt: skip

# (part name, source part in MediaPipe output, indices within that part)
GROUPS: Final[tuple[tuple[str, str, tuple[int, ...]], ...]] = (
    ("left_hand", "left_hand", HAND_IDX),
    ("right_hand", "right_hand", HAND_IDX),
    ("pose", "pose", POSE_IDX),
    ("lips", "face", LIPS_IDX),
)

N_LANDMARKS: Final[int] = sum(len(idx) for _, _, idx in GROUPS)  # 89


def part_slices() -> dict[str, slice]:
    """Return the slice of each part in the unified landmark axis."""
    out: dict[str, slice] = {}
    start = 0
    for name, _, idx in GROUPS:
        out[name] = slice(start, start + len(idx))
        start += len(idx)
    return out


SLICES: Final[dict[str, slice]] = part_slices()

# Positions (in the unified axis) of a few body anchors used by preprocessing.
POSE_POS: Final[dict[str, int]] = {
    name: SLICES["pose"].start + i for i, name in enumerate(POSE_NAMES)
}

_POSE_SWAP: Final[dict[str, str]] = {
    "l_shoulder": "r_shoulder",
    "l_elbow": "r_elbow",
    "l_wrist": "r_wrist",
}

# Symmetric face-mesh indices of the lip contour, used for horizontal flips.
_LIPS_MIRROR: Final[dict[int, int]] = {
    61: 291, 185: 409, 40: 270, 39: 269, 37: 267, 0: 0,
    146: 375, 91: 321, 181: 405, 84: 314, 17: 17,
    78: 308, 191: 415, 80: 310, 81: 311, 82: 312, 13: 13,
    95: 324, 88: 318, 178: 402, 87: 317, 14: 14,
}  # fmt: skip


def mirror_permutation() -> list[int]:
    """Index permutation implementing a left/right swap of the unified subset."""
    perm = list(range(N_LANDMARKS))
    lh, rh = SLICES["left_hand"], SLICES["right_hand"]
    for i in range(21):
        perm[lh.start + i] = rh.start + i
        perm[rh.start + i] = lh.start + i
    swap = {**_POSE_SWAP, **{v: k for k, v in _POSE_SWAP.items()}}
    for name, other in swap.items():
        perm[POSE_POS[name]] = POSE_POS[other]
    lips_mirror = {**_LIPS_MIRROR, **{v: k for k, v in _LIPS_MIRROR.items()}}
    pos_in_lips = {mp_idx: SLICES["lips"].start + i for i, mp_idx in enumerate(LIPS_IDX)}
    for mp_idx, pos in pos_in_lips.items():
        perm[pos] = pos_in_lips[lips_mirror[mp_idx]]
    return perm


# Skeleton edges (unified indices) used for drawing.
_HAND_EDGES: Final[tuple[tuple[int, int], ...]] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)  # fmt: skip


def skeleton_edges() -> list[tuple[int, int]]:
    """Edges between unified landmarks, for visualisation."""
    edges: list[tuple[int, int]] = []
    for part in ("left_hand", "right_hand"):
        s = SLICES[part].start
        edges += [(s + a, s + b) for a, b in _HAND_EDGES]
    p = POSE_POS
    edges += [
        (p["l_shoulder"], p["r_shoulder"]),
        (p["l_shoulder"], p["l_elbow"]),
        (p["l_elbow"], p["l_wrist"]),
        (p["r_shoulder"], p["r_elbow"]),
        (p["r_elbow"], p["r_wrist"]),
    ]
    # LIPS_IDX lists the upper arc corner-to-corner then the lower arc in the same direction,
    # so a closed ring is: upper arc + reversed lower arc.
    s = SLICES["lips"].start
    outer = [s + i for i in [*range(0, 11), *range(19, 10, -1)]]
    inner = [s + i for i in [*range(20, 30), 39, *range(38, 29, -1)]]
    for ring in (outer, inner):
        edges += [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]
    return edges
