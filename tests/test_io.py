import numpy as np

from conftest import make_kaggle_long
from signrec.io.kaggle import load_kaggle_sequence, long_to_frames, unified_from_kaggle_index
from signrec.io.lsfb import parts_to_unified
from signrec.landmarks import LIPS_IDX, N_LANDMARKS, POSE_IDX, SLICES
from signrec.store import PackedDataset, PackedWriter


def test_kaggle_dense_shape_and_nan() -> None:
    dense = long_to_frames(make_kaggle_long(n_frames=4))
    assert dense.shape == (4, 543, 3)
    assert np.isnan(dense[:, 468:489, 0]).all()  # missing left hand
    assert not np.isnan(dense[:, 522:, 0]).any()


def test_kaggle_unified_order(kaggle_parquet) -> None:
    seq = load_kaggle_sequence(kaggle_parquet)
    assert seq.shape == (5, N_LANDMARKS, 3) and seq.dtype == np.float32
    x = seq[0, :, 0]
    assert np.isnan(x[SLICES["left_hand"]]).all()
    np.testing.assert_array_equal(x[SLICES["right_hand"]], 522 + np.arange(21))
    np.testing.assert_array_equal(x[SLICES["pose"]], 489 + np.array(POSE_IDX))
    np.testing.assert_array_equal(x[SLICES["lips"]], np.array(LIPS_IDX))
    assert len(unified_from_kaggle_index()) == N_LANDMARKS


def test_lsfb_parts_to_unified() -> None:
    T = 3
    parts = {
        "pose": np.tile(np.arange(33)[None, :, None], (T, 1, 3)).astype(np.float16),
        "left_hand": np.full((T, 21, 3), 1.0),
        "right_hand": np.full((T, 21, 3), 2.0),
        "face": np.tile(np.arange(478)[None, :, None], (T, 1, 3)).astype(np.float32),
    }
    seq = parts_to_unified(parts)
    assert seq.shape == (T, N_LANDMARKS, 3)
    assert (seq[:, SLICES["left_hand"]] == 1).all() and (seq[:, SLICES["right_hand"]] == 2).all()
    np.testing.assert_array_equal(seq[0, SLICES["pose"], 0], POSE_IDX)
    np.testing.assert_array_equal(seq[0, SLICES["lips"], 0], LIPS_IDX)


def test_store_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(0)
    seqs = [rng.random((t, N_LANDMARKS, 3)).astype(np.float32) for t in (3, 7, 1)]
    seqs[1][2, 5] = np.nan
    w = PackedWriter("toy")
    for i, s in enumerate(seqs):
        w.add(s, f"id{i}", ["b", "a", "b"][i], f"S{i}", 30.0)
    w.write(tmp_path)
    ds = PackedDataset(tmp_path)
    assert len(ds) == 3 and ds.n_classes == 2
    for i, s in enumerate(seqs):
        np.testing.assert_allclose(ds.sequence(i), s, atol=1e-3, equal_nan=True)
    assert list(ds.targets()) == [1, 0, 1]
