#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.testing import assert_true
from dipy.testing.decorators import set_random_number_generator
from dipy.tracking.streamline import Streamlines
from dipy.utils.optpkg import optional_package
from dipy.workflows.stats import (
    BundleAnalysisTractometryFlow,
    BundleShapeAnalysis,
    LinearMixedModelsFlow,
    SNRinCCFlow,
    _match_peak_dirs_prefix,
    buan_bundle_profiles,
)

pd, have_pandas, _ = optional_package("pandas")
_, have_statsmodels, _ = optional_package("statsmodels", min_version="0.14.0")
_, have_tables, _ = optional_package("tables")
_, have_matplotlib, _ = optional_package("matplotlib")


def test_stats(tmp_path):
    data_path, bval_path, bvec_path = get_fnames(name="small_101D")
    volume, affine = load_nifti(data_path)
    mask = np.ones_like(volume[:, :, :, 0], dtype=np.uint8)
    mask_path = tmp_path / "tmp_mask.nii.gz"
    save_nifti(mask_path, mask, affine)

    snr_flow = SNRinCCFlow(force=True)
    args = [data_path, bval_path, bvec_path, mask_path]
    snr_flow.run(*args, out_dir=tmp_path)

    assert_true((tmp_path / "product.json").exists())
    assert_true(os.stat(tmp_path / "product.json").st_size != 0)
    assert_true((tmp_path / "cc.nii.gz").exists())
    assert_true(os.stat(tmp_path / "cc.nii.gz").st_size != 0)
    assert_true((tmp_path / "mask_noise.nii.gz").exists())
    assert_true(os.stat(tmp_path / "mask_noise.nii.gz").st_size != 0)

    snr_flow._force_overwrite = True
    snr_flow.run(*args, out_dir=tmp_path)
    assert_true((tmp_path / "product.json").exists())
    assert_true(os.stat(tmp_path / "product.json").st_size != 0)
    assert_true((tmp_path / "cc.nii.gz").exists())
    assert_true(os.stat(tmp_path / "cc.nii.gz").st_size != 0)
    assert_true((tmp_path / "mask_noise.nii.gz").exists())
    assert_true(os.stat(tmp_path / "mask_noise.nii.gz").st_size != 0)

    snr_flow._force_overwrite = True
    snr_flow.run(*args, bbox_threshold=(0.5, 1, 0, 0.15, 0, 0.2), out_dir=tmp_path)
    assert_true((tmp_path / "product.json").exists())
    assert_true(os.stat(tmp_path / "product.json").st_size != 0)
    assert_true((tmp_path / "cc.nii.gz").exists())
    assert_true(os.stat(tmp_path / "cc.nii.gz").st_size != 0)
    assert_true((tmp_path / "mask_noise.nii.gz").exists())
    assert_true(os.stat(tmp_path / "mask_noise.nii.gz").st_size != 0)


@pytest.mark.skipif(
    not have_pandas or not have_statsmodels or not have_tables or not have_matplotlib,
    reason="Requires Pandas, StatsModels and PyTables",
)
@set_random_number_generator()
def test_buan_bundle_profiles(tmp_path, rng=None):
    data_path = get_fnames(name="fornix")
    fornix = load_tractogram(data_path, "same", bbox_valid_check=False).streamlines

    f = Streamlines(fornix)

    mb = tmp_path / "model_bundles"

    os.mkdir(mb)

    sft = StatefulTractogram(f, data_path, Space.RASMM)
    save_tractogram(sft, Path(mb) / "temp.trk", bbox_valid_check=False)

    rb = tmp_path / "rec_bundles"
    os.mkdir(rb)

    sft = StatefulTractogram(f, data_path, Space.RASMM)
    save_tractogram(sft, Path(rb) / "temp.trk", bbox_valid_check=False)

    ob = tmp_path / "org_bundles"
    os.mkdir(ob)

    sft = StatefulTractogram(f, data_path, Space.RASMM)
    save_tractogram(sft, Path(ob) / "temp.trk", bbox_valid_check=False)

    dt = tmp_path / "anatomical_measures"
    os.mkdir(dt)

    fa = rng.random((255, 255, 255))

    save_nifti(Path(dt) / "fa.nii.gz", fa, affine=np.eye(4))

    out_dir = tmp_path / "output"
    os.mkdir(out_dir)

    buan_bundle_profiles(
        mb,
        rb,
        ob,
        dt,
        group_id=1,
        subject="10001",
        no_disks=100,
        out_dir=out_dir,
    )

    assert_true((out_dir / "temp_fa.h5").exists())


@pytest.mark.skipif(
    not have_pandas or not have_statsmodels or not have_tables or not have_matplotlib,
    reason="Requires Pandas, StatsModels, PyTables, and matplotlib",
)
@set_random_number_generator()
def test_bundle_analysis_tractometry_flow(tmp_path, rng=None):
    data_path = get_fnames(name="fornix")
    fornix = load_tractogram(data_path, "same", bbox_valid_check=False).streamlines

    f = Streamlines(fornix)

    mb = tmp_path / "model_bundles"
    sub = tmp_path / "subjects"

    os.mkdir(mb)
    sft = StatefulTractogram(f, data_path, Space.RASMM)
    save_tractogram(sft, Path(mb) / "temp.trk", bbox_valid_check=False)

    os.mkdir(sub)

    os.mkdir(Path(sub) / "patient")

    os.mkdir(Path(sub) / "control")

    p = sub / "patient" / "10001"
    os.mkdir(p)

    c = sub / "control" / "20002"
    os.mkdir(c)

    for pre in [p, c]:
        os.mkdir(Path(pre) / "rec_bundles")

        sft = StatefulTractogram(f, data_path, Space.RASMM)
        save_tractogram(
            sft,
            Path(pre) / "rec_bundles" / "temp.trk",
            bbox_valid_check=False,
        )
        os.mkdir(Path(pre) / "org_bundles")

        sft = StatefulTractogram(f, data_path, Space.RASMM)
        save_tractogram(
            sft,
            Path(pre) / "org_bundles" / "temp.trk",
            bbox_valid_check=False,
        )
        os.mkdir(Path(pre) / "anatomical_measures")

        fa = rng.random((255, 255, 255))

        save_nifti(
            Path(pre) / "anatomical_measures" / "fa.nii.gz",
            fa,
            affine=np.eye(4),
        )

    out_dir = tmp_path / "output"
    os.mkdir(out_dir)

    ba_flow = BundleAnalysisTractometryFlow()

    ba_flow.run(mb, sub, out_dir=out_dir)

    assert_true(Path(Path(out_dir) / "temp_fa.h5").exists())

    dft = pd.read_hdf(Path(out_dir) / "temp_fa.h5")

    # assert_true(dft.bundle.unique() == "temp")

    assert_true(set(dft.subject.unique()) == {"10001", "20002"})


@pytest.mark.skipif(
    not have_pandas or not have_statsmodels or not have_tables or not have_matplotlib,
    reason="Requires Pandas, StatsModels, PyTables, and matplotlib",
)
def test_linear_mixed_models_flow(tmp_path):
    out_dir = tmp_path / "output"
    os.mkdir(out_dir)

    d = {
        "disk": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5] * 10,
        "fa": [0.21, 0.234, 0.44, 0.44, 0.5, 0.23, 0.55, 0.34, 0.76, 0.34] * 10,
        "subject": [
            "10001",
            "10001",
            "10001",
            "10001",
            "10001",
            "20002",
            "20002",
            "20002",
            "20002",
            "20002",
        ]
        * 10,
        "group": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 10,
    }

    df = pd.DataFrame(data=d)
    store = pd.HDFStore(Path(out_dir) / "temp_fa.h5")
    store.append("fa", df, data_columns=True)
    store.close()

    lmm_flow = LinearMixedModelsFlow()

    out_dir2 = tmp_path / "output2"
    os.mkdir(out_dir2)

    input_path = Path(out_dir) / "*"

    lmm_flow.run(input_path, no_disks=5, out_dir=out_dir2)

    assert_true(Path(Path(out_dir2) / "temp_fa_pvalues.npy").exists())
    assert_true(Path(Path(out_dir2) / "temp_fa.png").exists())

    # test error
    d2 = {
        "disk": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5] * 1,
        "fa": [0.21, 0.234, 0.44, 0.44, 0.5, 0.23, 0.55, 0.34, 0.76, 0.34] * 1,
        "subject": [
            "10001",
            "10001",
            "10001",
            "10001",
            "10001",
            "20002",
            "20002",
            "20002",
            "20002",
            "20002",
        ]
        * 1,
        "group": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 1,
    }

    df = pd.DataFrame(data=d2)

    out_dir3 = tmp_path / "output3"
    os.mkdir(out_dir3)

    store = pd.HDFStore(Path(out_dir3) / "temp_fa.h5")
    store.append("fa", df, data_columns=True)
    store.close()

    out_dir4 = tmp_path / "output4"
    os.mkdir(out_dir4)

    input_path = Path(out_dir3) / "f*"
    # OS error raised if path is wrong or file does not exist
    npt.assert_raises(OSError, lmm_flow.run, input_path, no_disks=5, out_dir=out_dir4)

    input_path = Path(out_dir3) / "*"
    # value error raised if length of data frame is less than 100
    npt.assert_raises(
        ValueError, lmm_flow.run, input_path, no_disks=5, out_dir=out_dir4
    )


@set_random_number_generator()
def test_bundle_analysis_tractometry_flow_single_subject(tmp_path, rng=None):
    data_path = get_fnames(name="fornix")
    fornix = load_tractogram(data_path, "same", bbox_valid_check=False).streamlines

    f = Streamlines(fornix)

    mb = tmp_path / "model_bundles"
    os.mkdir(mb)
    sft = StatefulTractogram(f, data_path, Space.RASMM)
    save_tractogram(sft, Path(mb) / "temp.trk", bbox_valid_check=False)

    # Single-subject directory directly contains rec_bundles/, org_bundles/,
    # and anatomical_measures/ (no patient/control subdirs)
    single_sub = tmp_path / "10001"
    os.mkdir(single_sub)

    os.mkdir(single_sub / "rec_bundles")
    sft = StatefulTractogram(f, data_path, Space.RASMM)
    save_tractogram(
        sft, single_sub / "rec_bundles" / "temp.trk", bbox_valid_check=False
    )

    os.mkdir(single_sub / "org_bundles")
    sft = StatefulTractogram(f, data_path, Space.RASMM)
    save_tractogram(
        sft, single_sub / "org_bundles" / "temp.trk", bbox_valid_check=False
    )

    os.mkdir(single_sub / "anatomical_measures")
    fa = rng.random((40, 40, 40))
    save_nifti(
        single_sub / "anatomical_measures" / "fa.nii.gz",
        fa,
        affine=np.eye(4),
    )

    out_dir = tmp_path / "output"
    os.mkdir(out_dir)

    ba_flow = BundleAnalysisTractometryFlow()
    ba_flow.run(mb, single_sub, out_dir=out_dir)

    assert_true(Path(out_dir / "temp_fa_profile.npy").exists())


def _write_per_peak_subject(root):
    """Write bundles along x and a metric folder mixing 3D and per-peak maps.

    ``peak_v`` uses ``peak_dirs`` (slot 0 along x carries 2, slot 1 along y
    carries 5). ``peak_v_w`` matches both ``peak_dirs`` and ``peak_v_w_dirs``;
    the latter has the slots swapped so that the x-peak carries 7.
    """
    shape = (40, 40, 40)
    affine = np.eye(4)
    for name in ("model_bundles", "rec_bundles", "org_bundles", "anatomical_measures"):
        os.mkdir(root / name)
    measures = root / "anatomical_measures"

    save_nifti(measures / "fa.nii.gz", np.full(shape, 0.5), affine)

    dirs = np.zeros(shape + (2, 3))
    dirs[..., 0, 0] = 1.0
    dirs[..., 1, 1] = 1.0
    save_nifti(measures / "peak_dirs.nii.gz", dirs, affine)
    save_nifti(measures / "peak_v_w_dirs.nii.gz", dirs[..., ::-1, :], affine)

    values = np.empty(shape + (2,))
    values[..., 0], values[..., 1] = 2.0, 5.0
    save_nifti(measures / "peak_v.nii.gz", values, affine)
    values[..., 0], values[..., 1] = 3.0, 7.0
    save_nifti(measures / "peak_v_w.nii.gz", values, affine)
    save_nifti(measures / "other_v.nii.gz", values, affine)

    t = np.linspace(5, 34, 20)
    lines = []
    for i in range(10):
        pts = np.full((20, 3), 20.0)
        pts[:, 0] = t
        pts[:, 2] += 0.1 * i
        lines.append(pts)
    for name in ("model_bundles", "rec_bundles", "org_bundles"):
        sft = StatefulTractogram(
            Streamlines(lines), measures / "fa.nii.gz", Space.RASMM
        )
        save_tractogram(sft, root / name / "temp.trk", bbox_valid_check=False)


def test_match_peak_dirs_prefix():
    prefixes = ["force", "force_peak", "peak", "afd"]
    npt.assert_equal(_match_peak_dirs_prefix("force_peak_odi", prefixes), "force_peak")
    npt.assert_equal(_match_peak_dirs_prefix("force_peakodi", prefixes), "force")
    npt.assert_equal(_match_peak_dirs_prefix("afd", prefixes), "afd")
    npt.assert_equal(_match_peak_dirs_prefix("peak_odi", prefixes), "peak")
    assert _match_peak_dirs_prefix("afd2", prefixes) is None
    assert _match_peak_dirs_prefix("csd_afd", prefixes) is None


def test_bundle_analysis_tractometry_flow_single_subject_per_peak(tmp_path):
    subject = tmp_path / "10001"
    os.mkdir(subject)
    _write_per_peak_subject(subject)
    out_dir = tmp_path / "output"

    BundleAnalysisTractometryFlow().run(
        subject / "model_bundles", subject, no_disks=10, out_dir=out_dir
    )

    npt.assert_equal(
        sorted(p.name for p in out_dir.iterdir()),
        [
            "temp_fa_profile.npy",
            "temp_peak_v_profile.npy",
            "temp_peak_v_w_profile.npy",
        ],
    )
    npt.assert_almost_equal(np.load(out_dir / "temp_fa_profile.npy"), np.full(10, 0.5))
    # Bundle runs along x: the x-peak value is sampled.
    npt.assert_almost_equal(
        np.load(out_dir / "temp_peak_v_profile.npy"), np.full(10, 2.0)
    )
    # Longest prefix wins: peak_v_w uses peak_v_w_dirs (x-peak in slot 1).
    npt.assert_almost_equal(
        np.load(out_dir / "temp_peak_v_w_profile.npy"), np.full(10, 7.0)
    )


@pytest.mark.skipif(
    not have_pandas or not have_tables, reason="Requires Pandas and PyTables"
)
def test_buan_bundle_profiles_per_peak(tmp_path):
    _write_per_peak_subject(tmp_path)
    out_dir = tmp_path / "output"
    os.mkdir(out_dir)

    buan_bundle_profiles(
        tmp_path / "model_bundles",
        tmp_path / "rec_bundles",
        tmp_path / "org_bundles",
        tmp_path / "anatomical_measures",
        group_id=1,
        subject="10001",
        no_disks=10,
        out_dir=out_dir,
    )

    npt.assert_equal(
        sorted(p.name for p in out_dir.iterdir()),
        ["temp_fa.h5", "temp_peak_v.h5", "temp_peak_v_w.h5"],
    )
    for name, expected in (("fa", 0.5), ("peak_v", 2.0), ("peak_v_w", 7.0)):
        df = pd.read_hdf(out_dir / f"temp_{name}.h5")
        npt.assert_equal(len(df), 200)
        npt.assert_almost_equal(df[name].to_numpy(), expected)
        npt.assert_equal(df["streamline"].to_numpy(), np.repeat(np.arange(10), 20))
        npt.assert_equal(set(df["disk"]), set(range(1, 11)))


@pytest.mark.skipif(
    not have_pandas or not have_statsmodels or not have_tables or not have_matplotlib,
    reason="Requires Pandas, StatsModels, PyTables, and matplotlib",
)
@set_random_number_generator()
def test_bundle_shape_analysis_flow(tmp_path, rng=None):
    data_path = get_fnames(name="fornix")
    fornix = load_tractogram(data_path, "same", bbox_valid_check=False).streamlines

    f = Streamlines(fornix)

    mb = tmp_path / "model_bundles"
    sub = tmp_path / "subjects"

    os.mkdir(mb)
    sft = StatefulTractogram(f, data_path, Space.RASMM)
    save_tractogram(sft, Path(mb) / "temp.trk", bbox_valid_check=False)

    os.mkdir(sub)

    os.mkdir(sub / "patient")

    os.mkdir(sub / "control")

    p = sub / "patient" / "10001"
    os.mkdir(p)

    c = sub / "control" / "20002"
    os.mkdir(c)

    for pre in [p, c]:
        os.mkdir(Path(pre) / "rec_bundles")

        sft = StatefulTractogram(f, data_path, Space.RASMM)
        save_tractogram(
            sft,
            Path(pre) / "rec_bundles" / "temp.trk",
            bbox_valid_check=False,
        )
        os.mkdir(Path(pre) / "org_bundles")

        sft = StatefulTractogram(f, data_path, Space.RASMM)
        save_tractogram(
            sft,
            Path(pre) / "org_bundles" / "temp.trk",
            bbox_valid_check=False,
        )
        os.mkdir(Path(pre) / "anatomical_measures")

        fa = rng.random((255, 255, 255))

        save_nifti(
            Path(pre) / "anatomical_measures" / "fa.nii.gz",
            fa,
            affine=np.eye(4),
        )

    out_dir = tmp_path / "output"
    os.mkdir(out_dir)

    sm_flow = BundleShapeAnalysis()

    sm_flow.run(sub, out_dir=out_dir)

    assert_true(Path(Path(out_dir) / "temp.npy").exists())
