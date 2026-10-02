import numpy as np

from signrec import landmarks as lm


def test_subset_size_and_order() -> None:
    assert lm.N_LANDMARKS == 21 + 21 + 7 + 40
    s = lm.SLICES
    assert [k for k in s] == ["left_hand", "right_hand", "pose", "lips"]
    assert s["left_hand"] == slice(0, 21)
    assert s["right_hand"] == slice(21, 42)
    assert s["pose"] == slice(42, 49)
    assert s["lips"] == slice(49, 89)
    assert len(set(lm.LIPS_IDX)) == 40


def test_mirror_permutation_is_involution() -> None:
    perm = np.array(lm.mirror_permutation())
    assert sorted(perm) == list(range(lm.N_LANDMARKS))
    assert (perm[perm] == np.arange(lm.N_LANDMARKS)).all()
    assert perm[0] == 21 and perm[21] == 0
    assert perm[lm.POSE_POS["l_wrist"]] == lm.POSE_POS["r_wrist"]
    assert perm[lm.POSE_POS["nose"]] == lm.POSE_POS["nose"]


def test_skeleton_edges_valid() -> None:
    edges = lm.skeleton_edges()
    assert all(0 <= a < lm.N_LANDMARKS and 0 <= b < lm.N_LANDMARKS for a, b in edges)
    lips = [e for e in edges if e[0] >= 49]
    assert len(lips) == 40  # two closed rings of 20
