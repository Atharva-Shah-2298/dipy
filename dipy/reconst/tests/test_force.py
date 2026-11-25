"""Tests for FORCE reconstruction module."""

from concurrent.futures import ThreadPoolExecutor
import json
import types
import warnings

import numpy as np
import numpy.testing as npt
from numpy.testing import assert_almost_equal, assert_array_less
import pytest

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.reconst.force import (
    DEFAULT_FORCE_SEED,
    DEFAULT_NUM_ODI_VALUES,
    DEFAULT_ODI_RANGE,
    PEAK_METRICS,
    FORCEFit,
    FORCEModel,
    SignalIndex,
    _fwhm_kde_batch,
    _odi_grid_matches,
    _posterior_peak_params,
    _seed_matches,
    _weighted_percentile,
    compute_microstructure_uncertainty_ambiguity,
    compute_uncertainty_ambiguity,
    create_signal_index,
    force_peaks,
    normalize_signals,
    softmax_stable,
)
from dipy.reconst.multi_voxel import MultiVoxelFit
from dipy.reconst.shm import descoteaux07_legacy_msg, sf_to_sh


def _make_gtab(shells):
    """Minimal GradientTable: two b0s + 6 directions per non-zero shell."""
    from dipy.core.gradients import gradient_table

    dirs = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )
    bvals = [0, 0]
    bvecs = [[0, 0, 0], [0, 0, 0]]
    for b in shells:
        bvals.extend([b] * 6)
        bvecs.extend(dirs.tolist())
    return gradient_table(np.array(bvals, dtype=float), bvecs=np.array(bvecs))


@pytest.mark.parametrize("with_odf", [False, True])
@pytest.mark.parametrize("use_mask", [False, True])
def test_force_peaks_volume(with_odf, use_mask):
    """Preserve peak placement and SH across sparse, non-contiguous volumes."""
    shape = (3, 67, 101)
    fits = np.full((67, 101, 3), None, dtype=object).transpose(2, 0, 1)
    fit_mask = np.zeros(shape, dtype=bool)
    peak_mask = np.zeros(shape, dtype=bool)
    expected_dirs = np.zeros((*shape, 5, 3), dtype=np.float32)
    expected_inds = np.full((*shape, 5), -1, dtype=np.int32)
    expected_vals = np.zeros((*shape, 5), dtype=np.float32)
    expected_sh = np.zeros((*shape, 28)) if with_odf else None
    normalized_odfs = []
    original_odfs = []

    # Leave the beginning empty and include separated voxels near the end.
    for i, flat_index in enumerate((10_000, 10_001, 19_999, 20_000, 20_300)):
        voxel = np.unravel_index(flat_index, shape)
        indices = np.array([i + 2, i + 23])
        labels = np.zeros(len(default_sphere.vertices), dtype=np.int8)
        labels[indices] = 1
        fracs = np.array([0.6, 0.3, 0.0], dtype=np.float32)
        odf = None
        if with_odf:
            odf = np.linspace(i, i + 2, labels.size, dtype=np.float32)
            if i == 0:
                odf *= 0.4
            elif i == 2:
                odf.fill(2)
            original_odfs.append((odf, odf.copy()))
            normalized_odfs.append(
                (odf - odf.min()) / (odf.max() - odf.min() + 1e-12)
                if odf.max() > 1
                else odf
            )
        fits[voxel] = FORCEFit(None, {"labels": labels, "fracs": fracs, "odf": odf})
        fit_mask[voxel] = True
        peak_mask[voxel] = i != 1
        if not use_mask or peak_mask[voxel]:
            expected_dirs[voxel][:2] = default_sphere.vertices[indices]
            expected_inds[voxel][:2] = indices
            expected_vals[voxel][:3] = fracs

    fit = MultiVoxelFit(None, fits, fit_mask)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        if with_odf:
            expected_sh[fit_mask] = sf_to_sh(
                np.asarray(normalized_odfs), default_sphere, sh_order_max=6
            )
        pam = force_peaks(fit, mask=peak_mask if use_mask else None, sh_order=6)

    npt.assert_array_equal(pam.peak_dirs, expected_dirs)
    npt.assert_array_equal(pam.peak_indices, expected_inds)
    npt.assert_array_equal(pam.peak_values, expected_vals)
    if with_odf:
        npt.assert_allclose(pam.shm_coeff, expected_sh, rtol=1e-12, atol=1e-12)
        for odf, original in original_odfs:
            npt.assert_array_equal(odf, original)
    else:
        assert pam.shm_coeff is None


def test_force_peaks_single_voxel():
    """Single-voxel conversion retains its original ODF scaling."""
    indices = np.array([6, 42])
    labels = np.zeros(len(default_sphere.vertices), dtype=np.int8)
    labels[indices] = 1
    fracs = np.array([0.6, 0.3, 0.0], dtype=np.float32)
    odf = np.linspace(1, 3, labels.size, dtype=np.float32)
    fit = FORCEFit(None, {"labels": labels, "fracs": fracs, "odf": odf})
    expected_dirs = np.zeros((5, 3), dtype=np.float32)
    expected_dirs[:2] = default_sphere.vertices[indices]
    expected_inds = np.full(5, -1, dtype=np.int32)
    expected_inds[:2] = indices
    expected_vals = np.zeros(5, dtype=np.float32)
    expected_vals[:3] = fracs

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        expected_sh = sf_to_sh(odf, default_sphere, sh_order_max=6)
        pam = force_peaks(fit, sh_order=6)

    npt.assert_array_equal(pam.peak_dirs, expected_dirs)
    npt.assert_array_equal(pam.peak_indices, expected_inds)
    npt.assert_array_equal(pam.peak_values, expected_vals)
    npt.assert_allclose(pam.shm_coeff, expected_sh, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("use_posterior", [False, True])
def test_force_peak_metrics_describe_the_matched_fibers(
    use_posterior, tmp_path, monkeypatch
):
    """Best-match per-peak maps and force_peaks describe its fibers in order."""
    monkeypatch.setenv("DIPY_HOME", str(tmp_path))
    gtab = _make_gtab([1000, 2000])
    model = FORCEModel(
        gtab,
        penalty=0,
        n_neighbors=8,
        use_posterior=use_posterior,
        peak_source="best_match",
    )
    model.generate(num_simulations=400, num_cpus=1, verbose=False)
    sims = model.simulations
    # Matching a library entry against itself returns that entry.
    query = np.flatnonzero((sims["fiber_indices"] >= 0).sum(axis=1) >= 2)[:20]
    query = np.concatenate([query, np.flatnonzero(sims["fiber_indices"][:, 0] < 0)[:5]])
    fit = model.fit(sims["signals"][query])

    dirs, inds = fit.peak_dirs, fit.peak_indices
    assert dirs.shape == (query.size, 5, 3)
    for name in (
        "fiber_fraction",
        "fiber_density",
        "neurite_density",
        "axonal_diffusivity",
        "extra_axonal_diffusivity",
        "odi",
    ):
        assert getattr(fit, f"peak_{name}").shape == (query.size, 5)

    for row, sim in enumerate(query):
        present = sims["fiber_indices"][sim] >= 0
        n_peaks = present.sum()
        fractions = sims["fraction_array"][sim][present]
        intra = sims["f_ins"][sim][present]
        order = np.argsort(-fractions, kind="stable")
        expected_inds = sims["fiber_indices"][sim][present][order]

        npt.assert_array_equal(inds[row, :n_peaks], expected_inds)
        npt.assert_array_equal(inds[row, n_peaks:], -1)
        npt.assert_allclose(dirs[row, :n_peaks], default_sphere.vertices[expected_inds])
        npt.assert_array_equal(dirs[row, n_peaks:], 0)
        npt.assert_allclose(fit.peak_fiber_fraction[row, :n_peaks], fractions[order])
        npt.assert_allclose(fit.peak_neurite_density[row, :n_peaks], intra[order])
        npt.assert_allclose(
            fit.peak_fiber_density[row, :n_peaks],
            sims["wm_fraction"][sim] * fractions[order] * intra[order],
            rtol=1e-6,
        )
        # Every fiber has its own diffusivities and dispersion.
        for name, key in (
            ("axonal_diffusivity", "fiber_d_par"),
            ("extra_axonal_diffusivity", "fiber_d_perp_extra"),
            ("odi", "fiber_disp"),
        ):
            values = getattr(fit, f"peak_{name}")[row]
            npt.assert_allclose(values[:n_peaks], sims[key][sim][present][order])
            npt.assert_array_equal(values[n_peaks:], 0)
        npt.assert_array_equal(fit.peak_fiber_fraction[row, n_peaks:], 0)

    # force_peaks shares the slot order: peak k's value is fiber k's fraction.
    pam = force_peaks(fit)
    npt.assert_array_equal(pam.peak_dirs, dirs)
    npt.assert_array_equal(pam.peak_indices, inds)
    npt.assert_array_equal(pam.peak_values, fit.peak_fiber_fraction)
    single = force_peaks(fit[0])
    npt.assert_array_equal(single.peak_dirs, dirs[0])
    npt.assert_array_equal(single.peak_values, fit.peak_fiber_fraction[0])


def _fixel_library():
    """Hand-made library whose neighbors list fibers x and y in either order."""
    v = default_sphere.vertices
    x = 0
    y = int(np.argmin(np.abs(v @ v[x])))
    z = int(np.argmax(np.abs(v @ np.cross(v[x], v[y]))))
    # A fiber about 20 degrees away from x.
    near_x = int(np.argmin(np.abs(np.degrees(np.arccos(np.abs(v @ v[x]))) - 20)))
    d = {
        "fiber_indices": np.array(
            [[x, y, -1], [y, x, -1], [x, -1, -1], [z, -1, -1], [near_x, -1, -1]],
            dtype=np.int16,
        ),
        "fraction_array": np.array(
            [[0.6, 0.4, 0], [0.7, 0.3, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]
        ),
        "f_ins": np.array(
            [[0.7, 0.5, 0], [0.6, 0.8, 0], [0.4, 0, 0], [0.9, 0, 0], [0.3, 0, 0]]
        ),
        "wm_fraction": np.array([0.8, 0.6, 0.9, 0.5, 0.7]),
        "wm_d_par": np.array([2.0e-3, 2.5e-3, 3.0e-3, 2.2e-3, 2.4e-3]),
        "wm_d_perp_extra": np.array([5e-4, 7e-4, 9e-4, 6e-4, 8e-4]),
        "wm_disp": np.array([0.05, 0.1, 0.2, 0.15, 0.25]),
        "fiber_d_par": np.array(
            [
                [2.0e-3, 2.9e-3, 0],
                [2.5e-3, 2.1e-3, 0],
                [3.0e-3, 0, 0],
                [2.2e-3, 0, 0],
                [2.4e-3, 0, 0],
            ]
        ),
        "fiber_d_perp_extra": np.array(
            [
                [5e-4, 1.1e-3, 0],
                [7e-4, 4e-4, 0],
                [9e-4, 0, 0],
                [6e-4, 0, 0],
                [8e-4, 0, 0],
            ]
        ),
        "fiber_disp": np.array(
            [[0.05, 0.2, 0], [0.1, 0.03, 0], [0.2, 0, 0], [0.15, 0, 0], [0.25, 0, 0]]
        ),
    }
    return d, (x, y, z, near_x)


def _expected_fixel(d, neighbors, W, members):
    """Hand-computed posterior fixel parameters from (neighbor, slot) pairs."""
    w = np.array([W[k] * d["fraction_array"][neighbors[k], j] for k, j in members])
    sims = [neighbors[k] for k, _ in members]
    slots = [j for _, j in members]
    intra = d["f_ins"][sims, slots]
    expected = {"fiber_fraction": w.sum()}
    for name, x in (
        ("neurite_density", intra),
        ("axonal_diffusivity", d["fiber_d_par"][sims, slots]),
        ("extra_axonal_diffusivity", d["fiber_d_perp_extra"][sims, slots]),
        ("odi", d["fiber_disp"][sims, slots]),
    ):
        mean = np.sum(w * x) / w.sum()
        expected[name] = mean
        expected[f"{name}_std"] = np.sqrt(np.sum(w * (x - mean) ** 2) / w.sum())
    per_neighbor = np.zeros(len(W))
    for k, j in members:
        s = neighbors[k]
        per_neighbor[k] += (
            d["wm_fraction"][s] * d["fraction_array"][s, j] * d["f_ins"][s, j]
        )
    fd = np.sum(W * per_neighbor)
    expected["fiber_density"] = fd
    expected["fiber_density_std"] = np.sqrt(np.sum(W * (per_neighbor - fd) ** 2))
    expected["support"] = sum(W[k] for k in {k for k, _ in members})
    return expected


def test_posterior_fixels_pool_neighbor_fibers():
    """Fixels pool neighbor fibers regardless of their slot order."""
    d, (x, y, z, _) = _fixel_library()
    v = default_sphere.vertices
    neighbors = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    # The second voxel weighs fiber z enough to form its own fixel.
    W = np.array([[0.5, 0.28, 0.2, 0.02], [0.1, 0.7, 0.1, 0.1]])
    params = _posterior_peak_params(neighbors, W, d, peak_source="fibers")

    expected_fixels = (
        ((x, [(0, 0), (1, 1), (2, 0)]), (y, [(0, 1), (1, 0)])),
        ((y, [(0, 1), (1, 0)]), (x, [(0, 0), (1, 1), (2, 0)]), (z, [(3, 0)])),
    )
    for row, fixels in enumerate(expected_fixels):
        n = len(fixels)
        npt.assert_array_equal(params["peak_indices"][row, :n], [f[0] for f in fixels])
        npt.assert_array_equal(params["peak_indices"][row, n:], -1)
        npt.assert_array_equal(params["peak_dirs"][row, n:], 0)
        for p, (vertex, members) in enumerate(fixels):
            cos = np.abs(params["peak_dirs"][row, p] @ v[vertex])
            assert cos > np.cos(np.deg2rad(1))
            # Directions keep the orientation of their sphere vertex.
            assert params["peak_dirs"][row, p] @ v[vertex] > 0
            expected = _expected_fixel(d, neighbors[row], W[row], members)
            for name, value in expected.items():
                npt.assert_allclose(
                    params[f"peak_{name}"][row, p], value, rtol=1e-5, atol=1e-9
                )
        for name in (*expected, "fiber_fraction"):
            npt.assert_array_equal(params[f"peak_{name}"][row, n:], 0)


def test_posterior_fixels_assign_angle():
    """Fibers beyond ``assign_angle`` from every fixel join no fixel."""
    d, (x, _, _, near_x) = _fixel_library()
    neighbors = np.array([[2, 4]])
    W = np.array([[0.97, 0.03]])
    # The fiber 20 degrees off x is too light for a fixel of its own.
    absorbed = _posterior_peak_params(neighbors, W, d, peak_source="fibers")
    npt.assert_array_equal(absorbed["peak_indices"][0], [x, -1, -1, -1, -1])
    npt.assert_allclose(absorbed["peak_fiber_fraction"][0, 0], 1.0, rtol=1e-6)
    npt.assert_allclose(absorbed["peak_support"][0, 0], 1.0, rtol=1e-6)

    excluded = _posterior_peak_params(
        neighbors, W, d, peak_source="fibers", assign_angle=10
    )
    npt.assert_array_equal(excluded["peak_indices"][0], [x, -1, -1, -1, -1])
    npt.assert_allclose(excluded["peak_fiber_fraction"][0, 0], 0.97, rtol=1e-6)
    npt.assert_allclose(excluded["peak_support"][0, 0], 0.97, rtol=1e-6)
    npt.assert_allclose(excluded["peak_neurite_density"][0, 0], 0.4, rtol=1e-6)
    npt.assert_allclose(excluded["peak_neurite_density_std"][0, 0], 0, atol=1e-7)


def test_posterior_fixels_hybrid_keeps_best_match_layout():
    """Hybrid fixels are the best match's fibers with pooled metrics."""
    d, (x, y, z, _) = _fixel_library()
    v = default_sphere.vertices
    neighbors = np.array([[0, 1, 2, 3]])
    W = np.array([[0.5, 0.28, 0.2, 0.02]])
    # The best match (library entry 1) lists y (0.7) before x (0.3).
    params = _posterior_peak_params(
        neighbors, W, d, peak_source="hybrid", lib_idx=np.array([1])
    )

    npt.assert_array_equal(params["peak_indices"][0], [y, x, -1, -1, -1])
    npt.assert_allclose(params["peak_fiber_fraction"][0], [0.7, 0.3, 0, 0, 0])
    # Fiber z, far from both best-match fibers, joins neither fixel.
    pooled = 1 - W[0, 3]
    for p, (vertex, members) in enumerate(
        ((y, [(0, 1), (1, 0)]), (x, [(0, 0), (1, 1), (2, 0)]))
    ):
        assert abs(params["peak_dirs"][0, p] @ v[vertex]) > np.cos(np.deg2rad(1))
        expected = _expected_fixel(d, neighbors[0], W[0], members)
        for name in (
            "neurite_density",
            "axonal_diffusivity",
            "extra_axonal_diffusivity",
            "odi",
            "neurite_density_std",
            "support",
        ):
            npt.assert_allclose(params[f"peak_{name}"][0, p], expected[name], rtol=1e-5)
        npt.assert_allclose(
            params["peak_fiber_density"][0, p],
            expected["fiber_density"] / pooled,
            rtol=1e-5,
        )
    # A best match without fibers gives no fixels, whatever the neighbors say.
    d["fiber_indices"][3] = -1
    empty = _posterior_peak_params(
        neighbors, W, d, peak_source="hybrid", lib_idx=np.array([3])
    )
    npt.assert_array_equal(empty["peak_indices"][0], -1)


def test_force_peak_source_validation():
    """Unknown peak sources are rejected."""
    with pytest.raises(ValueError, match="peak_source"):
        FORCEModel(_make_gtab([1000]), peak_source="mean")


def test_force_best_match_peak_source_in_posterior_mode(tmp_path, monkeypatch):
    """``peak_source='best_match'`` reproduces the best-match peaks."""
    monkeypatch.setenv("DIPY_HOME", str(tmp_path))
    gtab = _make_gtab([1000, 2000])
    model = FORCEModel(gtab, n_neighbors=8, peak_source="best_match")
    model.generate(num_simulations=400, num_cpus=1, verbose=False)
    sims = model.simulations
    rng = np.random.default_rng(0)
    data = sims["signals"][:30] + rng.normal(0, 0.02, (30, sims["signals"].shape[1]))
    exact = FORCEModel(gtab, simulations=sims, n_neighbors=8, use_posterior=False)

    fit, fit_exact = model.fit(data), exact.fit(data)
    assert fit[0].entropy is not None
    for name in ("dirs", "indices", *PEAK_METRICS):
        npt.assert_array_equal(
            getattr(fit, f"peak_{name}"), getattr(fit_exact, f"peak_{name}")
        )
    assert fit[0].peak_support is None
    assert fit[0].peak_odi_std is None


def test_force_posterior_fixels_fit(tmp_path, monkeypatch):
    """Posterior fixels from the fiber and ODF densities find the crossings."""
    from dipy.data import get_sphere

    monkeypatch.setenv("DIPY_HOME", str(tmp_path))
    dirs = get_sphere(name="repulsion100").vertices[:30]
    gtab = gradient_table(
        np.r_[0, 0, [1000] * 30, [2000] * 30].astype(float),
        bvecs=np.r_[np.zeros((2, 3)), dirs, dirs],
    )
    model = FORCEModel(gtab, n_neighbors=8)
    model.generate(num_simulations=2000, num_cpus=1, verbose=False)
    sims = model.simulations
    v = default_sphere.vertices

    # Library crossings of two well-separated, comparable, coherent fibers.
    idx = sims["fiber_indices"]
    two = np.flatnonzero((idx[:, 0] >= 0) & (idx[:, 1] >= 0) & (idx[:, 2] < 0))
    cos = np.abs(np.sum(v[idx[two, 0]] * v[idx[two, 1]], axis=1))
    query = two[
        (cos < np.cos(np.deg2rad(60)))
        & (sims["fraction_array"][two, :2].min(axis=1) > 0.3)
        & (sims["fiber_disp"][two, :2].max(axis=1) <= 0.1)
    ]
    assert query.size >= 5
    data = sims["signals"][query]

    fit = model.fit(data)
    model.peak_source = "odf"
    fit_odf = model.fit(data)

    for f in (fit, fit_odf):
        assert f.peak_support.shape == (query.size, 5)
        assert np.all((f.peak_support >= 0) & (f.peak_support <= 1 + 1e-6))
        assert np.all(f.peak_fiber_fraction.sum(axis=1) <= 1 + 1e-5)
        for name in PEAK_METRICS[1:]:
            assert np.all(getattr(f, f"peak_{name}_std") >= 0)
        # force_peaks reports the posterior fixels.
        pam = force_peaks(f)
        npt.assert_array_equal(pam.peak_dirs, f.peak_dirs)
        npt.assert_array_equal(pam.peak_indices, f.peak_indices)
        npt.assert_array_equal(pam.peak_values, f.peak_fiber_fraction)

    for row, sim in enumerate(query):
        truth = v[idx[sim, :2]]
        for f in (fit, fit_odf):
            assert np.all(f.peak_indices[row, :2] >= 0)
            npt.assert_array_equal(f.peak_indices[row, 2:], -1)
            found = f.peak_dirs[row, :2]
            angles = np.degrees(np.arccos(np.clip(np.abs(found @ truth.T), 0, 1)))
            assert np.all(angles.min(axis=1) < 3)
            assert np.all(angles.min(axis=0) < 3)
        # Both densities yield the same fixels.
        between = np.abs(np.sum(fit.peak_dirs[row] * fit_odf.peak_dirs[row], -1))
        assert np.all(np.degrees(np.arccos(np.clip(between[:2], 0, 1))) < 3)


def test_force_peaks_mask_empties_unfitted_and_unmasked_voxels(tmp_path, monkeypatch):
    """Peak slots outside the fit mask or the peak mask are empty."""
    monkeypatch.setenv("DIPY_HOME", str(tmp_path))
    gtab = _make_gtab([1000, 2000])
    model = FORCEModel(gtab, penalty=0, n_neighbors=8)
    model.generate(num_simulations=200, num_cpus=1, verbose=False)
    sims = model.simulations
    with_fibers = np.flatnonzero(sims["fiber_indices"][:, 0] >= 0)[:4]
    data = sims["signals"][with_fibers].reshape(2, 2, 1, -1)
    fit_mask = np.array([[[True], [True]], [[True], [False]]])
    peak_mask = np.array([[[True], [False]], [[True], [True]]])

    pam = force_peaks(model.fit(data, mask=fit_mask), mask=peak_mask)

    kept = fit_mask & peak_mask
    assert np.all(pam.peak_indices[kept][:, 0] >= 0)
    npt.assert_array_equal(pam.peak_indices[~kept], -1)
    npt.assert_array_equal(pam.peak_values[~kept], 0)
    npt.assert_array_equal(pam.peak_dirs[~kept], 0)


def test_normalize_signals():
    """Test signal normalization."""
    signals = np.array([[3, 4], [0, 0], [1, 0]], dtype=np.float32)
    normalized = normalize_signals(signals)

    # First row should have unit norm
    assert_almost_equal(np.linalg.norm(normalized[0]), 1.0)

    # Zero row should remain zero (handled by eps)
    assert normalized[1, 0] == 0.0

    # Third row should have unit norm
    assert_almost_equal(np.linalg.norm(normalized[2]), 1.0)


def test_softmax_stable():
    """Test numerically stable softmax."""
    x = np.array([[1000, 1001, 1002], [0, 0, 0]], dtype=np.float32)
    result = softmax_stable(x, axis=1)

    # Should sum to 1 along axis
    assert_almost_equal(np.sum(result, axis=1), [1.0, 1.0])

    # Should not have NaN or Inf
    assert np.all(np.isfinite(result))


def test_compute_uncertainty_ambiguity():
    """Test uncertainty and ambiguity metrics."""
    scores = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )

    uncertainty, ambiguity = compute_uncertainty_ambiguity(scores)

    assert uncertainty.shape == (2,)
    assert ambiguity.shape == (2,)

    # First row has spread, second is uniform
    assert uncertainty[0] > uncertainty[1]


def test_signal_index():
    """Test SignalIndex inner product search."""
    index = SignalIndex(10)

    # Add some vectors
    vectors = np.random.randn(100, 10).astype(np.float32)
    index.add(vectors)

    assert index.ntotal == 100

    # Search
    query = np.random.randn(5, 10).astype(np.float32)
    D, neighbors = index.search(query, k=10)

    assert D.shape == (5, 10)
    assert neighbors.shape == (5, 10)

    # Distances should be in descending order
    for i in range(5):
        assert np.all(D[i, :-1] >= D[i, 1:])


def test_create_signal_index():
    """Test signal index creation."""
    signals = np.random.randn(100, 50).astype(np.float32)
    signals_norm = signals / np.linalg.norm(signals, axis=1, keepdims=True)

    index = create_signal_index(signals_norm)

    assert index.ntotal == 100
    assert index.d == 50


def test_signal_search():
    """Test signal matching search."""
    # Create mock index
    signals = np.random.randn(100, 50).astype(np.float32)
    signals_norm = signals / np.linalg.norm(signals, axis=1, keepdims=True)
    index = create_signal_index(signals_norm)

    # Query signals
    query = np.random.randn(10, 50).astype(np.float32)
    query_norm = query / np.linalg.norm(query, axis=1, keepdims=True)

    D, neighbors = index.search(query_norm, k=20)

    assert D.shape == (10, 20)
    assert neighbors.shape == (10, 20)


def test_weighted_percentile():
    """Test weighted percentile computation."""
    # Simple case: uniform weights should give standard percentile
    vals = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
    weights = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]], dtype=np.float32)

    q50 = _weighted_percentile(vals, weights, 0.5)
    assert q50.shape == (1,)
    # Median of [1,2,3,4,5] with uniform weights should be around 3
    assert 2.0 <= q50[0] <= 4.0

    # Concentrated weights: all weight on first value
    weights_conc = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    q50_conc = _weighted_percentile(vals, weights_conc, 0.5)
    assert_almost_equal(q50_conc[0], 1.0)

    # Test batch processing
    vals_batch = np.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]], dtype=np.float32)
    weights_batch = np.array(
        [[0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]], dtype=np.float32
    )
    q75 = _weighted_percentile(vals_batch, weights_batch, 0.75)
    assert q75.shape == (2,)
    # Second row values are 10x larger
    assert q75[1] > q75[0]


def test_fwhm_kde_batch():
    """Test FWHM via weighted KDE."""
    # Concentrated distribution should have small FWHM
    vals_narrow = np.array([[5.0, 5.1, 5.0, 4.9, 5.0]], dtype=np.float32)
    weights_uniform = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]], dtype=np.float32)

    fwhm_narrow = _fwhm_kde_batch(vals_narrow, weights_uniform)
    assert fwhm_narrow.shape == (1,)
    assert fwhm_narrow[0] >= 0

    # Spread distribution should have larger FWHM
    vals_wide = np.array([[0.0, 2.5, 5.0, 7.5, 10.0]], dtype=np.float32)
    fwhm_wide = _fwhm_kde_batch(vals_wide, weights_uniform)

    assert fwhm_wide[0] > fwhm_narrow[0]

    # Test batch processing
    vals_batch = np.vstack([vals_narrow, vals_wide])
    weights_batch = np.vstack([weights_uniform, weights_uniform])
    fwhm_batch = _fwhm_kde_batch(vals_batch, weights_batch)

    assert fwhm_batch.shape == (2,)
    assert fwhm_batch[1] > fwhm_batch[0]


def test_compute_microstructure_uncertainty_ambiguity():
    """Test microstructure uncertainty and ambiguity metrics."""
    # Case 1: Concentrated values - low uncertainty and ambiguity
    vals_narrow = np.array(
        [[0.5, 0.51, 0.49, 0.5, 0.5], [0.5, 0.51, 0.49, 0.5, 0.5]],
        dtype=np.float32,
    )
    weights_uniform = np.array(
        [[0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]],
        dtype=np.float32,
    )
    prior_range = 1.0

    unc_narrow, amb_narrow = compute_microstructure_uncertainty_ambiguity(
        vals_narrow, weights_uniform, prior_range
    )

    assert unc_narrow.shape == (2,)
    assert amb_narrow.shape == (2,)
    # Should be low values since distribution is concentrated
    assert_array_less(unc_narrow, 0.5)
    assert_array_less(amb_narrow, 0.5)

    # Case 2: Spread values - higher uncertainty and ambiguity
    vals_wide = np.array(
        [[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0]],
        dtype=np.float32,
    )

    unc_wide, amb_wide = compute_microstructure_uncertainty_ambiguity(
        vals_wide, weights_uniform, prior_range
    )

    # Wide distribution should have higher uncertainty than narrow
    assert unc_wide[0] > unc_narrow[0]
    assert amb_wide[0] > amb_narrow[0]

    # Case 3: Values are in [0, 1], results normalized by prior_range
    assert np.all(unc_wide >= 0)
    assert np.all(amb_wide >= 0)

    # Case 4: Test with concentrated weights
    weights_conc = np.array(
        [[0.9, 0.025, 0.025, 0.025, 0.025], [0.9, 0.025, 0.025, 0.025, 0.025]],
        dtype=np.float32,
    )
    unc_conc, amb_conc = compute_microstructure_uncertainty_ambiguity(
        vals_wide, weights_conc, prior_range
    )

    # Concentrated weights should reduce uncertainty even with spread values
    assert unc_conc[0] < unc_wide[0]


def test_compute_microstructure_uncertainty_ambiguity_different_ranges():
    """Test that prior_range correctly normalizes uncertainty/ambiguity."""
    vals = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=np.float32)
    weights = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]], dtype=np.float32)

    # Same values but different prior ranges
    unc_small, amb_small = compute_microstructure_uncertainty_ambiguity(
        vals, weights, prior_range=1.0
    )
    unc_large, amb_large = compute_microstructure_uncertainty_ambiguity(
        vals, weights, prior_range=10.0
    )

    # Larger prior range should give smaller normalized values
    assert unc_large[0] < unc_small[0]
    assert amb_large[0] < amb_small[0]


def _tiny_model(n_sims=64, n_grad=12, seed=0):
    """A FORCEModel backed by a small hand-made library (no generation)."""
    rng = np.random.default_rng(seed)
    bvals = np.concatenate(([0.0], np.full(n_grad - 1, 1000.0)))
    bvecs = np.zeros((n_grad, 3))
    bvecs[1:, 0] = 1.0

    sims = {
        "signals": rng.random((n_sims, n_grad)).astype(np.float32) + 0.5,
        "labels": np.zeros((n_sims, len(default_sphere.vertices)), dtype=np.uint8),
        "num_fibers": np.ones(n_sims, dtype=np.float32),
        "fraction_array": rng.random((n_sims, 3)).astype(np.float32),
    }
    for key in (
        "fa",
        "md",
        "rd",
        "wm_fraction",
        "gm_fraction",
        "csf_fraction",
        "dispersion",
        "nd",
    ):
        sims[key] = rng.random(n_sims).astype(np.float32)

    gtab = gradient_table(bvals, bvecs=bvecs)
    return FORCEModel(gtab, simulations=sims, n_neighbors=8), sims


def test_micro_postprocessing_serial_inside_ray_worker(monkeypatch):
    """The per-parameter loop uses threads normally, but not inside a Ray worker.

    Also checks both paths produce the same uncertainty and ambiguity maps.
    """
    model, sims = _tiny_model()
    query = sims["signals"][:12]

    pool_uses = []

    def spy(*args, **kwargs):
        pool_uses.append(kwargs.get("max_workers"))
        return ThreadPoolExecutor(*args, **kwargs)

    monkeypatch.setattr("dipy.reconst.force.ThreadPoolExecutor", spy)
    # Pin the core count so the threaded path does not depend on runner size.
    monkeypatch.setattr(
        "dipy.reconst.force.get_usable_cpu_affinity", lambda: {0, 1, 2, 3}
    )

    # Outside a Ray worker the loop is threaded. Force the flag rather than
    # relying on ambient state: another test may have left Ray initialised,
    # since paramap never shuts it down.
    monkeypatch.setattr("dipy.reconst.force.has_ray", False)
    threaded = model.fit(query)
    assert pool_uses, "expected the thread pool to be used outside a Ray worker"

    # Inside a live Ray worker it stays serial.
    pool_uses.clear()
    monkeypatch.setattr("dipy.reconst.force.has_ray", True)
    monkeypatch.setattr(
        "dipy.reconst.force.ray", types.SimpleNamespace(is_initialized=lambda: True)
    )
    serial = model.fit(query)
    assert not pool_uses, "expected no thread pool inside a Ray worker"

    for param in ("fa", "nd"):
        for metric in ("uncertainty", "ambiguity"):
            npt.assert_array_equal(
                np.asarray(getattr(threaded, f"{metric}_{param}")),
                np.asarray(getattr(serial, f"{metric}_{param}")),
            )


def test_odi_grid_matches_legacy_and_exact():
    """Cache matching on the ODI grid, incl. legacy (keyless) entries."""
    # A legacy entry (written before the ODI grid was part of the key) is
    # treated as the historical default grid.
    legacy = {"num_simulations": 100}
    assert _odi_grid_matches(legacy, DEFAULT_ODI_RANGE, DEFAULT_NUM_ODI_VALUES)
    assert not _odi_grid_matches(legacy, (0.01, 0.6), 19)
    assert not _odi_grid_matches(legacy, DEFAULT_ODI_RANGE, 5)

    # A modern entry matches only its exact range and grid.
    entry = {"odi_range": [0.01, 0.6], "num_odi_values": 19}
    assert _odi_grid_matches(entry, (0.01, 0.6), 19)
    assert not _odi_grid_matches(entry, (0.01, 0.6), 10)  # same range, diff grid
    assert not _odi_grid_matches(entry, (0.01, 0.3), 19)  # diff range, same grid


def test_seed_matches_legacy_and_exact():
    """Cache matching on the generation seed, incl. legacy (keyless) entries."""
    # A legacy entry (written before the seed was part of the key) was not
    # generated reproducibly, so it only satisfies seed=None requests.
    legacy = {"num_simulations": 100}
    assert _seed_matches(legacy, None)
    assert not _seed_matches(legacy, DEFAULT_FORCE_SEED)

    # A modern entry matches only its exact seed.
    entry = {"seed": DEFAULT_FORCE_SEED}
    assert _seed_matches(entry, DEFAULT_FORCE_SEED)
    assert not _seed_matches(entry, 42)
    assert not _seed_matches(entry, None)


def _registry(dipy_home):
    path = dipy_home / "force_simulations" / "cache_registry.json"
    return json.load(open(path)) if path.exists() else []


def test_force_cache_keys_on_odi_grid(tmp_path, monkeypatch):
    """The simulation cache is keyed on odi_range and the resolved grid."""
    monkeypatch.setenv("DIPY_HOME", str(tmp_path))
    gtab = _make_gtab([1000])

    def gen(**kwargs):
        FORCEModel(gtab).generate(
            num_simulations=60, num_cpus=1, verbose=False, **kwargs
        )

    def entries():
        return [
            (tuple(e["odi_range"]), e["num_odi_values"]) for e in _registry(tmp_path)
        ]

    # 1. default range -> one entry recorded at the autoscaled grid (10).
    gen()
    assert entries() == [((0.01, 0.3), DEFAULT_NUM_ODI_VALUES)]

    # 2. a wider range is a distinct library (autoscaled grid 19), not a hit.
    gen(odi_range=(0.01, 0.6))
    assert len(entries()) == 2
    assert ((0.01, 0.6), 19) in entries()

    # 3. re-running the default params hits the cache (no new entry).
    n_before = len(entries())
    gen()
    assert len(entries()) == n_before

    # 4. an explicit grid at the default range is again distinct.
    gen(num_odi_values=5)
    assert ((0.01, 0.3), 5) in entries()
    assert len(entries()) == 3


def test_force_cache_keys_on_seed(tmp_path, monkeypatch):
    """The simulation cache is keyed on the generation seed."""
    monkeypatch.setenv("DIPY_HOME", str(tmp_path))
    gtab = _make_gtab([1000])

    def gen(**kwargs):
        model = FORCEModel(gtab)
        model.generate(num_simulations=60, num_cpus=1, verbose=False, **kwargs)
        return model

    # 1. the default seed is recorded in the registry.
    first = gen()
    assert [e["seed"] for e in _registry(tmp_path)] == [DEFAULT_FORCE_SEED]

    # 2. re-running with the default seed hits the cache (no new entry) and
    # serves the identical library.
    again = gen()
    assert len(_registry(tmp_path)) == 1
    npt.assert_array_equal(again.simulations["signals"], first.simulations["signals"])

    # 3. a different seed is a distinct library, not a cache hit.
    other = gen(seed=42)
    assert len(_registry(tmp_path)) == 2
    assert {e["seed"] for e in _registry(tmp_path)} == {DEFAULT_FORCE_SEED, 42}
    assert not np.array_equal(
        other.simulations["signals"], first.simulations["signals"]
    )


def test_force_cache_reports_seedless_legacy_entry(tmp_path, monkeypatch):
    """A legacy cache miss explains why a seeded library is regenerated."""
    monkeypatch.setenv("DIPY_HOME", str(tmp_path))
    gtab = _make_gtab([1000])

    FORCEModel(gtab).generate(
        num_simulations=10,
        num_cpus=1,
        seed=None,
        compute_dti=False,
        verbose=False,
    )

    registry_path = tmp_path / "force_simulations" / "cache_registry.json"
    registry = _registry(tmp_path)
    registry[0].pop("seed")
    with open(registry_path, "w") as f:
        json.dump(registry, f)

    messages = []
    monkeypatch.setattr("dipy.reconst.force.logger.info", messages.append)
    FORCEModel(gtab).generate(
        num_simulations=10,
        num_cpus=1,
        compute_dti=False,
        verbose=False,
    )

    assert any(
        "predate seed tracking" in message
        and "new seeded library will be generated" in message
        for message in messages
    )
    assert _registry(tmp_path)[-1]["seed"] == DEFAULT_FORCE_SEED


def test_cache_registry_separates_min_crossing_angles(tmp_path, monkeypatch):
    """Libraries built with different crossing-angle limits get separate cache slots."""
    from dipy.core.gradients import gradient_table

    monkeypatch.setenv("DIPY_HOME", str(tmp_path))

    dirs = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )
    bvals = np.array([0.0] + [1000.0] * 6 + [2000.0] * 6)
    bvecs = np.vstack(([[0.0, 0.0, 0.0]], dirs, dirs))
    gtab = gradient_table(bvals, bvecs=bvecs)

    strict = FORCEModel(gtab)
    strict.generate(num_simulations=60, num_cpus=1, verbose=False)

    relaxed = FORCEModel(gtab)
    relaxed.generate(
        num_simulations=60, num_cpus=1, two_fiber_min_angle=0.0, verbose=False
    )

    cache_dir = tmp_path / "force_simulations"
    assert len(list(cache_dir.glob("force_sim_*.npz"))) == 2

    # The relaxed request must not be served the strict library, and a repeat
    # of the relaxed request must reuse the one just written.
    again = FORCEModel(gtab)
    again.generate(
        num_simulations=60, num_cpus=1, two_fiber_min_angle=0.0, verbose=False
    )
    assert len(list(cache_dir.glob("force_sim_*.npz"))) == 2
    npt.assert_array_equal(again.simulations["signals"], relaxed.simulations["signals"])


def test_force_cache_skips_libraries_without_fiber_metadata(tmp_path, monkeypatch):
    """Cached libraries that predate per-fiber metadata are regenerated."""
    monkeypatch.setenv("DIPY_HOME", str(tmp_path))
    gtab = _make_gtab([1000])
    FORCEModel(gtab).generate(num_simulations=60, num_cpus=1, verbose=False)

    registry_path = tmp_path / "force_simulations" / "cache_registry.json"
    registry = _registry(tmp_path)
    registry[0].pop("fiber_metadata")
    with open(registry_path, "w") as f:
        json.dump(registry, f)

    model = FORCEModel(gtab)
    model.generate(num_simulations=60, num_cpus=1, verbose=False)
    assert len(_registry(tmp_path)) == 2
    assert "fiber_indices" in model.simulations

    # The regenerated library is then reused.
    FORCEModel(gtab).generate(num_simulations=60, num_cpus=1, verbose=False)
    assert len(_registry(tmp_path)) == 2
