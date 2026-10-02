"""The script run on Kaggle must behave exactly like the package code."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

from signrec.io.kaggle import load_kaggle_sequence

ROOT = Path(__file__).resolve().parents[1]
ENTRY = "# " + "-" * 64 + " kernel entry point"


def _load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_inlined_kernel_matches_package(kaggle_parquet) -> None:
    src = _load("build_kaggle_kernel", ROOT / "scripts/build_kaggle_kernel.py").build()
    assert ENTRY in src
    kernel = types.ModuleType("kernel_under_test")
    sys.modules[kernel.__name__] = kernel  # dataclasses need a registered module
    exec(compile(src.split(ENTRY)[0], "kernel", "exec"), kernel.__dict__)
    a = kernel.load_kaggle_sequence(kaggle_parquet)
    np.testing.assert_array_equal(a, load_kaggle_sequence(kaggle_parquet))
    assert a.shape[1] == kernel.N_LANDMARKS
