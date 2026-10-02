import numpy as np
import pytest

from conftest import random_sequence
from signrec.augment import AugmentConfig, augment_raw, make_augmenter, spatial, time_warp
from signrec.landmarks import N_LANDMARKS, SLICES
from signrec.preprocess import (
    PreprocessConfig,
    canonicalize_hand,
    fix_length,
    mirror,
    preprocess,
    resample,
    trim_handless,
)

CFG = PreprocessConfig()


def test_config_from_yaml_matches_defaults() -> None:
    assert PreprocessConfig.from_yaml("configs/preprocess.yaml") == CFG


@pytest.mark.parametrize("shift,scale", [((0.1, -0.05), 1.0), ((0.0, 0.0), 1.7), ((0.2, 0.1), 0.6)])
def test_invariant_to_translation_and_scale(shift, scale) -> None:
    seq = random_sequence()
    moved = seq.copy()
    moved[..., :2] = (seq[..., :2] - 0.5) * scale + 0.5 + np.array(shift)
    a, _ = preprocess(seq, CFG)
    b, _ = preprocess(moved, CFG)
    np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4)


def test_mirror_twice_is_identity() -> None:
    seq = random_sequence(nan_left=True)
    np.testing.assert_allclose(mirror(mirror(seq)), seq, equal_nan=True, atol=1e-6)


def test_canonical_hand_moves_dominant_to_right_slot() -> None:
    seq = random_sequence()
    seq[:, SLICES["right_hand"]] = np.nan  # only the left hand is signing
    out = canonicalize_hand(seq)
    assert not np.isnan(out[:, SLICES["right_hand"]]).any()
    assert np.isnan(out[:, SLICES["left_hand"]]).all()


def test_mirrored_input_gives_same_features() -> None:
    seq = random_sequence()
    seq[:, SLICES["right_hand"]] = np.nan
    a, _ = preprocess(seq, CFG)
    b, _ = preprocess(mirror(seq), CFG)
    np.testing.assert_allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("T", [1, 5, 64, 1000])
@pytest.mark.parametrize("mode", ["resample", "pad"])
def test_shapes_and_no_nan(T, mode) -> None:
    cfg = PreprocessConfig(length_mode=mode)
    x, m = preprocess(random_sequence(T=T, nan_left=True), cfg)
    assert x.shape == (cfg.T, cfg.n_features) and m.shape == (cfg.T,)
    assert np.isfinite(x).all()
    if mode == "pad" and T < cfg.T:
        assert m.sum() == T and (x[~m] == 0).all()


def test_no_hand_at_all() -> None:
    seq = random_sequence()
    seq[:, :42] = np.nan
    x, _ = preprocess(seq, CFG)
    assert np.isfinite(x).all()


def test_all_nan_sequence() -> None:
    x, _ = preprocess(np.full((10, N_LANDMARKS, 3), np.nan, np.float32), CFG)
    assert np.isfinite(x).all() and (x == 0).all()


def test_trim_removes_handless_borders() -> None:
    seq = random_sequence(T=10)
    seq[:3, :42] = np.nan
    seq[8:, :42] = np.nan
    assert trim_handless(seq).shape[0] == 5


def test_resample_endpoints_and_nan_fallback() -> None:
    seq = random_sequence(T=7)
    seq[3, 0, 0] = np.nan
    out = resample(seq, 13)
    np.testing.assert_allclose(out[0], seq[0])
    np.testing.assert_allclose(out[-1], seq[-1])
    # a missing value is filled from the nearest source frame only when that frame exists
    nearest = np.floor(np.linspace(0, 6, 13) + 0.5).astype(int)  # ties go to the later frame
    assert (np.isnan(out[:, 0, 0]) == (nearest == 3)).all()
    assert fix_length(seq, 4, "pad")[1].all()


def test_feature_count() -> None:
    assert CFG.n_features == (89 * 2 + 21 * 2) * 2
    assert (
        PreprocessConfig(parts=("left_hand", "right_hand", "pose"), use_z=True).n_features
        == (49 * 3 + 21 * 3) * 2
    )


def test_augment_is_pure_and_valid() -> None:
    rng = np.random.default_rng(0)
    seq = random_sequence(T=40)
    ref = seq.copy()
    raw, spa = make_augmenter(AugmentConfig(), rng)
    out = raw(seq)
    np.testing.assert_array_equal(seq, ref)
    assert out.ndim == 3 and out.shape[1:] == seq.shape[1:] and out.shape[0] >= 2
    x, _ = preprocess(out, CFG, spa)
    assert np.isfinite(x).all()
    assert augment_raw(seq[:1], rng, AugmentConfig()).shape[0] == 1


def test_spatial_augmentation_survives_normalisation() -> None:
    """Spatial augmentation is applied after normalisation, so it must change the features."""
    seq = random_sequence(T=30)
    rng = np.random.default_rng(1)
    a, _ = preprocess(seq, CFG)
    b, _ = preprocess(seq, CFG, lambda s: spatial(s, rng, AugmentConfig(p_spatial=1.0)))
    assert np.abs(a - b).max() > 1e-2


def test_time_warp_is_monotonic_and_keeps_endpoints() -> None:
    seq = random_sequence(T=50)
    seq[:, 0, 0] = np.arange(50)  # a clock
    out = time_warp(seq, np.random.default_rng(3), AugmentConfig(speed=(1.0, 1.0)))
    clock = out[:, 0, 0]
    assert out.shape[0] == 50 and (np.diff(clock) >= 0).all()
    assert clock[0] == 0 and abs(clock[-1] - 49) < 1e-4


def test_y_scale_equals_scaling_the_input() -> None:
    seq = random_sequence(T=20)
    scaled = seq.copy()
    scaled[..., 1] *= 0.5625
    np.testing.assert_allclose(
        preprocess(seq, CFG, y_scale=0.5625)[0], preprocess(scaled, CFG)[0], atol=1e-5
    )
