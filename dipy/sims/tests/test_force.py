"""Tests for FORCE simulation module."""

import numpy as np
import numpy.testing as npt
import pytest

from dipy.sims.force import (
    DEFAULT_FORCE_SEED,
    DEFAULT_NUM_ODI_VALUES,
    DEFAULT_ODI_RANGE,
    dispersion_lut,
    get_default_diffusivity_config,
    init_worker,
    resolve_num_odi_values,
    smallest_shell_bval,
    validate_diffusivity_config,
)


def test_init_worker_base_seed_deprecated():
    """The removed base_seed argument remains available during deprecation."""
    import random

    numpy_state = np.random.get_state()
    random_state = random.getstate()
    try:
        with pytest.warns(DeprecationWarning, match='"base_seed" was deprecated'):
            init_worker(base_seed=42)
    finally:
        np.random.set_state(numpy_state)
        random.setstate(random_state)


def test_dispersion_lut_structure():
    """Test dispersion_lut returns correct structure."""
    sphere = np.random.randn(10, 3)
    sphere = sphere / np.linalg.norm(sphere, axis=1, keepdims=True)
    odi_list = np.array([0.1, 0.2, 0.3])

    result = dispersion_lut(sphere, odi_list)

    assert isinstance(result, dict)
    assert len(result) == 10
    for i in range(10):
        assert i in result
        assert isinstance(result[i], dict)
        for odi in odi_list:
            assert odi in result[i]


def test_validate_diffusivity_config_valid():
    """Test validation of valid diffusivity config."""
    config = get_default_diffusivity_config()
    assert validate_diffusivity_config(config) is True


def test_validate_diffusivity_config_missing_key():
    """Test validation fails for missing keys."""
    config = {"wm_d_par_range": (2e-3, 3e-3)}
    with pytest.raises(ValueError, match="Missing required key"):
        validate_diffusivity_config(config)


def test_validate_diffusivity_config_invalid_range():
    """Test validation fails for invalid range."""
    config = get_default_diffusivity_config()
    config["wm_d_par_range"] = (3e-3, 2e-3)  # min > max
    with pytest.raises(ValueError, match="min must be <= max"):
        validate_diffusivity_config(config)


def test_get_default_diffusivity_config():
    """Test default config has all required keys."""
    config = get_default_diffusivity_config()

    assert "wm_d_par_range" in config
    assert "wm_d_perp_range" in config
    assert "gm_d_iso_range" in config
    assert "csf_d" in config

    # Check reasonable values
    assert config["csf_d"] > 0
    assert config["wm_d_par_range"][0] > 0
    assert config["wm_d_par_range"][1] >= config["wm_d_par_range"][0]


def test_smallest_shell_bval():
    """Test smallest shell b-value finding for n=1 (default)."""
    bvals = np.array([0, 0, 1000, 1000, 2000, 2000, 3000])

    min_shell, mask = smallest_shell_bval(bvals)

    assert len(min_shell) == 1
    assert min_shell[0] == 1000
    assert mask.sum() == 2
    assert mask[2] and mask[3]
    # Other shells must NOT be in the mask
    assert not mask[4] and not mask[5] and not mask[6]


def test_smallest_shell_bval_n2():
    """Test smallest_shell_bval returns two smallest shells with n=2."""
    bvals = np.array([0, 0, 1000, 1000, 2000, 2000, 3000, 3000])

    min_shells, mask = smallest_shell_bval(bvals, n=2)

    assert len(min_shells) == 2
    assert min_shells[0] == 1000
    assert min_shells[1] == 2000
    # b0s excluded from shell mask
    assert not mask[0] and not mask[1]
    # Both shells selected
    assert mask[2] and mask[3]  # b=1000
    assert mask[4] and mask[5]  # b=2000
    # Largest shell excluded
    assert not mask[6] and not mask[7]


def test_smallest_shell_bval_n_too_large():
    """Test smallest_shell_bval raises when n exceeds available shells."""
    bvals = np.array([0, 1000, 2000])
    with pytest.raises(ValueError, match="unique shells found"):
        smallest_shell_bval(bvals, n=3)


def test_smallest_shell_bval_no_nonzero():
    """Test smallest_shell_bval raises for all-zero bvals."""
    bvals = np.array([0, 0, 10, 20, 30])  # all below threshold
    with pytest.raises(ValueError, match="No non-b0 volumes"):
        smallest_shell_bval(bvals)


def test_save_load_force_simulations(tmp_path):
    """Test saving and loading FORCE simulations."""
    from dipy.sims.force import load_force_simulations, save_force_simulations

    # Create test simulations
    test_sims = {
        "signals": np.random.randn(10, 100).astype(np.float32),
        "labels": np.random.randint(0, 2, (10, 50)).astype(np.uint8),
        "fa": np.random.rand(10).astype(np.float32),
    }

    output_path = tmp_path / "test_sims.npz"
    save_force_simulations(test_sims, str(output_path))

    # Load and verify
    loaded = load_force_simulations(str(output_path))

    assert set(loaded.keys()) == set(test_sims.keys())
    for key in test_sims:
        np.testing.assert_array_equal(loaded[key], test_sims[key])


def _make_gtab(shells):
    """Build a minimal GradientTable with 6 isotropic directions per shell.

    Parameters
    ----------
    shells : list of int
        Non-zero b-values (one shell per entry). Two b0 volumes are prepended.
    """
    from dipy.core.gradients import gradient_table

    # 6 directions on a hemisphere (roughly isotropic)
    dirs = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=float,
    )
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    bvals = [0, 0]
    bvecs = [[0, 0, 0], [0, 0, 0]]
    for b in shells:
        bvals.extend([b] * 6)
        bvecs.extend(dirs.tolist())

    return gradient_table(np.array(bvals, dtype=float), bvecs=np.array(bvecs))


def test_generate_force_simulations_compute_dti():
    """generate_force_simulations with compute_dti=True returns FA/MD/RD."""
    from dipy.sims.force import generate_force_simulations

    gtab = _make_gtab([1000])
    sims = generate_force_simulations(
        gtab,
        num_simulations=20,
        batch_size=20,
        num_cpus=1,
        compute_dti=True,
        compute_dki=False,
        verbose=False,
    )

    for key in ("fa", "md", "rd"):
        assert key in sims, f"Key '{key}' missing from simulations"
        arr = sims[key]
        assert arr.shape == (20,), f"Expected shape (20,) for '{key}', got {arr.shape}"
        assert np.any(arr != 0), f"'{key}' is all-zeros – DTI fitting appears skipped"

    # DKI keys must NOT be present
    for key in ("ak", "rk", "mk", "kfa"):
        assert key not in sims, f"DKI key '{key}' should be absent"


def test_generate_force_simulations_compute_dki():
    """generate_force_simulations with compute_dki=True returns AK/RK/MK/KFA."""
    from dipy.sims.force import generate_force_simulations

    # DKI requires at least 2 non-zero shells
    gtab = _make_gtab([1000, 2000])
    sims = generate_force_simulations(
        gtab,
        num_simulations=20,
        batch_size=20,
        num_cpus=1,
        compute_dti=True,
        compute_dki=True,
        verbose=False,
    )

    dki_keys = ("ak", "rk", "mk", "kfa")
    for key in dki_keys:
        assert key in sims, f"Key '{key}' missing from simulations"
        arr = sims[key]
        assert arr.shape == (20,), f"Expected shape (20,) for '{key}', got {arr.shape}"
        assert np.any(arr != 0), f"'{key}' is all-zeros – DKI fitting appears skipped"

    # DTI keys also present when compute_dti=True
    for key in ("fa", "md", "rd"):
        assert key in sims, f"DTI key '{key}' missing"


def test_generate_force_simulations_no_dti_no_dki():
    """generate_force_simulations with both flags False omits metric keys."""
    from dipy.sims.force import generate_force_simulations

    gtab = _make_gtab([1000])
    sims = generate_force_simulations(
        gtab,
        num_simulations=10,
        batch_size=10,
        num_cpus=1,
        compute_dti=False,
        compute_dki=False,
        verbose=False,
    )

    for key in ("fa", "md", "rd", "ak", "rk", "mk", "kfa"):
        if key in ("fa", "md", "rd"):
            # These keys exist but should be all zeros (initialised to zeros,
            # never filled when compute_dti=False)
            assert key in sims
            assert np.all(sims[key] == 0), f"'{key}' should be zero when DTI disabled"
        else:
            assert key not in sims, f"DKI key '{key}' should be absent"


def test_resolve_num_odi_values_autoscale():
    """None autoscales the ODI grid to keep sampling density constant."""
    # Default range resolves to the historical fixed grid (backward compatible).
    assert (
        resolve_num_odi_values(DEFAULT_ODI_RANGE, num_odi_values=None)
        == DEFAULT_NUM_ODI_VALUES
    )

    # Doubling the span roughly doubles the number of grid points.
    assert resolve_num_odi_values((0.01, 0.6), num_odi_values=None) == 19

    # Narrower span -> fewer points; wider -> more.
    n_narrow = resolve_num_odi_values((0.05, 0.15), num_odi_values=None)
    n_wide = resolve_num_odi_values((0.01, 0.9), num_odi_values=None)
    assert n_narrow < DEFAULT_NUM_ODI_VALUES < n_wide

    # A degenerate (zero-width) range still yields a valid grid (>= 2).
    assert resolve_num_odi_values((0.2, 0.2), num_odi_values=None) == 2


def test_resolve_num_odi_values_explicit_and_invalid():
    """An explicit count is passed through; counts < 2 are rejected."""
    # Explicit value is honored regardless of the range.
    assert resolve_num_odi_values((0.01, 0.9), num_odi_values=7) == 7
    assert resolve_num_odi_values(DEFAULT_ODI_RANGE, num_odi_values=3) == 3

    for bad in (1, 0, -5):
        with pytest.raises(ValueError, match="must be >= 2"):
            resolve_num_odi_values(DEFAULT_ODI_RANGE, num_odi_values=bad)


def test_generate_force_simulations_honors_odi_grid():
    """A resolve error for num_odi_values < 2 propagates through generation."""
    from dipy.sims.force import generate_force_simulations

    gtab = _make_gtab([1000])
    with pytest.raises(ValueError, match="must be >= 2"):
        generate_force_simulations(
            gtab,
            num_simulations=10,
            batch_size=10,
            num_cpus=1,
            num_odi_values=1,
            verbose=False,
        )


def _library_crossing_angles(sims, num_fibers):
    """Antipodal-symmetric angles between the labelled fibers of *num_fibers* voxels."""
    from dipy.data import default_sphere

    vertices = default_sphere.vertices
    angles = []
    for labels in sims["labels"][sims["num_fibers"] == num_fibers]:
        dirs = vertices[np.flatnonzero(labels == 1)]
        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                cos = np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0)
                angles.append(np.rad2deg(np.arccos(abs(cos))))
    return np.asarray(angles)


def test_generate_force_simulations_default_min_crossing_angles():
    """By default the library holds no crossing below 30 (two) or 60 (three) degrees."""
    from dipy.sims.force import generate_force_simulations

    gtab = _make_gtab([1000, 2000])
    sims = generate_force_simulations(
        gtab, num_simulations=300, batch_size=100, num_cpus=1, verbose=False
    )

    two = _library_crossing_angles(sims, 2)
    three = _library_crossing_angles(sims, 3)
    assert two.size and three.size
    assert two.min() >= 30.0
    assert three.min() >= 60.0


def test_generate_force_simulations_relaxed_min_crossing_angles():
    """Lowering the limits lets shallow crossings into the library."""
    from dipy.sims.force import generate_force_simulations

    gtab = _make_gtab([1000, 2000])
    sims = generate_force_simulations(
        gtab,
        num_simulations=300,
        batch_size=100,
        num_cpus=1,
        two_fiber_min_angle=0.0,
        three_fiber_min_angle=0.0,
        verbose=False,
    )

    two = _library_crossing_angles(sims, 2)
    three = _library_crossing_angles(sims, 3)
    assert two.size and three.size
    assert two.min() < 30.0
    assert three.min() < 60.0


def test_generate_force_simulations_seed_reproducible():
    """Identical seeds reproduce the library; different seeds do not."""
    from dipy.sims.force import generate_force_simulations

    gtab = _make_gtab([1000])
    kwargs = {
        "num_simulations": 40,
        "batch_size": 10,
        "num_cpus": 1,
        "compute_dti": False,
        "verbose": False,
    }

    sims_a = generate_force_simulations(gtab, seed=42, **kwargs)
    sims_b = generate_force_simulations(gtab, seed=42, **kwargs)
    for key in ("signals", "labels", "num_fibers", "wm_fraction", "nd"):
        np.testing.assert_array_equal(sims_a[key], sims_b[key])

    # The default seed is applied when none is given, so two default runs
    # are also identical.
    sims_c = generate_force_simulations(gtab, **kwargs)
    sims_d = generate_force_simulations(gtab, **kwargs)
    np.testing.assert_array_equal(sims_c["signals"], sims_d["signals"])

    # A different seed yields a different library.
    sims_e = generate_force_simulations(gtab, seed=43, **kwargs)
    assert not np.array_equal(sims_a["signals"], sims_e["signals"])

    # seed=None draws fresh entropy on every run.
    sims_f = generate_force_simulations(gtab, seed=None, **kwargs)
    sims_g = generate_force_simulations(gtab, seed=None, **kwargs)
    assert not np.array_equal(sims_f["signals"], sims_g["signals"])


@pytest.mark.parametrize("seed", [DEFAULT_FORCE_SEED, None])
def test_generate_force_simulations_preserves_numpy_rng_state(seed):
    """Serial generation does not alter the caller's global NumPy RNG."""
    from dipy.sims.force import generate_force_simulations

    state = np.random.get_state()
    try:
        np.random.seed(1234)
        expected = np.random.random(5)
        np.random.seed(1234)

        generate_force_simulations(
            _make_gtab([1000]),
            num_simulations=10,
            batch_size=10,
            num_cpus=1,
            seed=seed,
            compute_dti=False,
            verbose=False,
        )

        np.testing.assert_array_equal(np.random.random(5), expected)
    finally:
        np.random.set_state(state)


def test_seeded_serial_generation_skips_worker_initialization(monkeypatch):
    """A deterministic batch seed makes worker initialization unnecessary."""
    from dipy.sims.force import generate_force_simulations

    def fail_if_called():
        pytest.fail("init_worker should not run for seeded serial generation")

    monkeypatch.setattr("dipy.sims.force.init_worker", fail_if_called)
    generate_force_simulations(
        _make_gtab([1000]),
        num_simulations=10,
        batch_size=10,
        num_cpus=1,
        seed=42,
        compute_dti=False,
        verbose=False,
    )


def test_generate_force_simulations_seed_independent_of_num_cpus():
    """The same seed gives the same library for any worker count."""
    from dipy.sims.force import generate_force_simulations

    gtab = _make_gtab([1000])
    kwargs = {
        "num_simulations": 40,
        "batch_size": 10,
        "seed": 42,
        "compute_dti": False,
        "verbose": False,
    }

    serial = generate_force_simulations(gtab, num_cpus=1, **kwargs)
    parallel = generate_force_simulations(gtab, num_cpus=2, **kwargs)
    for key in ("signals", "labels", "num_fibers", "wm_fraction", "nd"):
        np.testing.assert_array_equal(serial[key], parallel[key])


def test_generate_force_simulations_batches_draw_distinct_streams():
    """Each batch draws from its own stream, so batches are not clones."""
    from dipy.sims.force import generate_force_simulations

    gtab = _make_gtab([1000])
    sims = generate_force_simulations(
        gtab,
        num_simulations=40,
        batch_size=10,
        num_cpus=1,
        compute_dti=False,
        verbose=False,
    )

    batches = sims["signals"].reshape(4, 10, -1)
    for i in range(len(batches)):
        for j in range(i + 1, len(batches)):
            assert not np.array_equal(batches[i], batches[j]), (
                f"Batches {i} and {j} are identical – the same seed was "
                "propagated to every batch"
            )


@pytest.mark.parametrize(
    "kwargs",
    [{"two_fiber_min_angle": 90.0}, {"three_fiber_min_angle": -1.0}],
)
def test_generate_force_simulations_invalid_min_crossing_angle(kwargs):
    """Angles outside [0, 90) degrees are rejected."""
    from dipy.sims.force import generate_force_simulations

    gtab = _make_gtab([1000])
    with pytest.raises(ValueError, match="must be in .0, 90. degrees"):
        generate_force_simulations(
            gtab, num_simulations=10, batch_size=10, num_cpus=1, verbose=False, **kwargs
        )


def _mixed_signal(seed, *, relaxation_weights=None, n=20):
    """Draw ``n`` mixed voxels straight from the Cython simulator."""
    from dipy.data import get_sphere
    from dipy.sims._force_core import create_mixed_signal
    from dipy.sims._multi_tensor_omp import multi_tensor
    from dipy.sims.voxel import all_tensor_evecs

    gtab = _make_gtab([1000, 3000])
    sphere = np.ascontiguousarray(get_sphere(name="repulsion100").vertices)
    evecs = np.array([all_tensor_evecs(tuple(v)) for v in sphere])
    odi = np.array([0.05, 0.2])
    lut = dispersion_lut(sphere, odi)
    bvals = np.ascontiguousarray(gtab.bvals, dtype=np.float64)
    bvecs = np.ascontiguousarray(gtab.bvecs, dtype=np.float64)

    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return bvals, [
            create_mixed_signal(
                sphere,
                evecs,
                lut,
                odi,
                bvals,
                bvecs,
                multi_tensor,
                0.5,
                False,
                30.0,
                60.0,
                relaxation_weights,
            )
            for _ in range(n)
        ]
    finally:
        np.random.set_state(state)


def test_relaxation_weights_follow_spin_echo_model():
    """TE and TR weighting match the analytic spin-echo expression."""
    from dipy.sims.force import (
        RELAXATION_COMPARTMENTS,
        get_default_relaxation_params,
        relaxation_weights,
    )

    params = get_default_relaxation_params()
    t1 = np.array(
        [
            params[c[:2] if c.startswith("wm") else c]["t1"]
            for c in RELAXATION_COMPARTMENTS
        ]
    )
    t2 = np.array(
        [
            params[c[:2] if c.startswith("wm") else c]["t2"]
            for c in RELAXATION_COMPARTMENTS
        ]
    )

    te, tr = 0.1, 3.0
    npt.assert_allclose(relaxation_weights(te=te), np.exp(-te / t2))
    npt.assert_allclose(relaxation_weights(tr=tr), 1 - np.exp(-tr / t1))
    npt.assert_allclose(
        relaxation_weights(te=te, tr=tr),
        relaxation_weights(te=te) * relaxation_weights(tr=tr),
    )
    # No metadata, no weighting.
    npt.assert_array_equal(relaxation_weights(), np.ones(4))

    # Longer TE attenuates every compartment, short-T2 tissue the most, so
    # the CSF signal share grows with TE.
    w_te = np.array([relaxation_weights(te=t) for t in (0.06, 0.1, 0.14)])
    assert np.all(np.diff(w_te, axis=0) < 0)
    assert np.all(np.diff(w_te[:, 3] / w_te[:, 0]) > 0)

    # Longer TR recovers every compartment towards 1, long-T1 CSF last.
    w_tr = np.array([relaxation_weights(tr=t) for t in (2.0, 5.0, 8.0, 60.0)])
    assert np.all(np.diff(w_tr, axis=0) > 0)
    assert np.all(w_tr[:3, 3] < w_tr[:3, 0])
    npt.assert_allclose(w_tr[-1], 1.0, atol=1e-6)


def test_relaxation_times_are_seconds():
    """Millisecond inputs are rejected instead of being silently rescaled."""
    from dipy.sims.force import relaxation_weights, validate_te_tr

    assert validate_te_tr(te=0.0895, tr=5.52) == (0.0895, 5.52)
    assert validate_te_tr() == (None, None)
    # The same acquisition expressed in ms and converted gives equal weights.
    npt.assert_array_equal(
        relaxation_weights(te=89.5 / 1000, tr=5520 / 1000),
        relaxation_weights(te=0.0895, tr=5.52),
    )
    for kwargs in (
        {"te": 89.5},
        {"tr": 5520},
        {"te": 0.0},
        {"tr": -1.0},
        {"te": np.nan},
        {"te": "long"},
        {"te": 0.5, "tr": 0.2},
    ):
        with pytest.raises(ValueError):
            validate_te_tr(**kwargs)
    with pytest.raises(ValueError, match="milliseconds"):
        relaxation_weights(te=0.1, relaxation_params={"wm": {"t2": 75}})


def test_validate_relaxation_params():
    """Defaults fill the gaps; WM sub-compartments inherit from WM."""
    from dipy.sims.force import (
        DEFAULT_RELAXATION_PARAMS,
        get_default_relaxation_params,
        validate_relaxation_params,
    )

    defaults = get_default_relaxation_params()
    resolved = validate_relaxation_params(None)
    assert set(resolved) == {"wm_intra", "wm_extra", "gm", "csf"}
    assert resolved["wm_intra"] == resolved["wm_extra"] == defaults["wm"]
    assert resolved["gm"] == defaults["gm"]

    # A returned copy can be edited without touching the module defaults.
    defaults["wm"]["t2"] = 0.5
    assert DEFAULT_RELAXATION_PARAMS["wm"]["t2"] != 0.5

    custom = validate_relaxation_params(
        {"wm": {"t2": 0.07}, "wm_intra": {"t2": 0.09}, "csf": {"t1": 4.0}}
    )
    assert custom["wm_intra"]["t2"] == 0.09
    assert custom["wm_extra"]["t2"] == 0.07
    assert custom["wm_extra"]["t1"] == DEFAULT_RELAXATION_PARAMS["wm"]["t1"]
    assert custom["csf"] == {"t1": 4.0, "t2": DEFAULT_RELAXATION_PARAMS["csf"]["t2"]}

    for bad in (
        {"bone": {"t1": 1.0}},
        {"wm": {"t3": 1.0}},
        {"gm": {"t1": 0.05, "t2": 0.08}},
        {"gm": {"t2": 0.0}},
    ):
        with pytest.raises(ValueError):
            validate_relaxation_params(bad)


def test_resolve_relaxation_disabled_or_without_metadata():
    """Relaxation is opt-in and never invents a TE or TR."""
    from dipy.sims.force import resolve_relaxation

    assert resolve_relaxation() is None
    # Disabled: timing is ignored entirely, even if implausible.
    assert resolve_relaxation(te=0.1, tr=5.0) is None
    assert resolve_relaxation(te=100, tr=5000, relaxation_model=None) is None

    with pytest.warns(UserWarning, match="neither te nor tr"):
        assert resolve_relaxation(relaxation_model="tissue") is None

    both = resolve_relaxation(te=0.1, tr=3.0, relaxation_model="tissue")
    assert both["terms"] == ("t2", "t1")

    # TE alone: T2 weighting only, flagged as partial.
    with pytest.warns(UserWarning, match="only the T2"):
        te_only = resolve_relaxation(te=0.1, relaxation_model="tissue")
    assert te_only["terms"] == ("t2",) and te_only["tr"] is None
    assert np.all(te_only["weights"] > both["weights"])

    # TR alone: a T1-only correction is not justified, original FORCE is used.
    with pytest.warns(UserWarning, match="te is not available"):
        assert resolve_relaxation(tr=3.0, relaxation_model="tissue") is None

    with pytest.raises(ValueError, match="relaxation_model"):
        resolve_relaxation(te=0.1, relaxation_model="voxel")


def test_generate_force_simulations_unchanged_without_relaxation():
    """Disabled or metadata-free calls reproduce the original library."""
    from dipy.sims.force import generate_force_simulations

    gtab = _make_gtab([1000, 2000])
    kwargs = {"num_simulations": 60, "batch_size": 20, "verbose": False}

    original = generate_force_simulations(gtab, **kwargs)
    disabled = generate_force_simulations(
        gtab, te=0.1, tr=5.0, relaxation_model=None, **kwargs
    )
    ignored = generate_force_simulations(gtab, te=0.1, tr=5.0, **kwargs)
    with pytest.warns(UserWarning, match="neither te nor tr"):
        no_metadata = generate_force_simulations(
            gtab, relaxation_model="tissue", **kwargs
        )

    for sims in (disabled, ignored, no_metadata):
        assert list(sims) == list(original)
        for key in original:
            npt.assert_array_equal(sims[key], original[key])


def test_relaxation_branch_matches_original_mixture():
    """Unit weights reproduce the original signal bit for bit."""
    _, original = _mixed_signal(11)
    _, unit = _mixed_signal(11, relaxation_weights=np.ones(4))
    for res_o, res_u in zip(original, unit, strict=True):
        for out_o, out_u in zip(res_o, res_u, strict=True):
            npt.assert_array_equal(out_u, out_o)


def test_relaxation_weights_compartments_before_mixing():
    """Each compartment is scaled by its own weight, then mixed."""
    bvals, original = _mixed_signal(3)

    # Nulling CSF removes exactly csf_fraction * S_csf(b).
    _, no_csf = _mixed_signal(3, relaxation_weights=[1.0, 1.0, 1.0, 0.0])
    # Halving only the CSF weight removes half of it.
    _, half_csf = _mixed_signal(3, relaxation_weights=[1.0, 1.0, 1.0, 0.5])
    # Keeping only WM, with unequal intra/extra weights.
    _, wm_only = _mixed_signal(3, relaxation_weights=[1.0, 1.0, 0.0, 0.0])
    _, intra_only = _mixed_signal(3, relaxation_weights=[1.0, 0.0, 0.0, 0.0])
    _, extra_only = _mixed_signal(3, relaxation_weights=[0.0, 1.0, 0.0, 0.0])

    csf = 100.0 * np.exp(-bvals * 3.0e-3)
    for i, res in enumerate(original):
        csf_fraction = res[6]
        npt.assert_allclose(res[0] - no_csf[i][0], csf_fraction * csf, atol=1e-9)
        npt.assert_allclose(
            res[0] - half_csf[i][0], 0.5 * csf_fraction * csf, atol=1e-9
        )
        npt.assert_allclose(
            intra_only[i][0] + extra_only[i][0], wm_only[i][0], atol=1e-9
        )
        assert intra_only[i][0].max() > 0 and extra_only[i][0].max() > 0
        # Sampled microstructure is untouched by the weights.
        for j in range(1, len(res)):
            npt.assert_array_equal(no_csf[i][j], res[j])


def test_relaxation_preserves_sampled_microstructure_and_priors():
    """Same seed: same configurations, different signals, no new parameter."""
    from dipy.sims.force import generate_force_simulations

    gtab = _make_gtab([1000, 2000])
    kwargs = {"num_simulations": 2000, "batch_size": 500, "verbose": False}
    original = generate_force_simulations(gtab, **kwargs)
    aware = generate_force_simulations(
        gtab, te=0.12, tr=3.0, relaxation_model="tissue", **kwargs
    )

    signal_derived = {"signals", "fa", "md", "rd"}
    assert set(aware) - set(original) == {"relaxation_weights"}
    assert set(original) - set(aware) == set()
    assert aware["relaxation_weights"].shape == (4,)
    for key in set(original) - signal_derived:
        npt.assert_array_equal(aware[key], original[key])
    assert not np.array_equal(aware["signals"], original["signals"])

    # The priors themselves: Dirichlet(2, 1, 1) tissue fractions and
    # 10/20/70 % one/two/three-fiber voxels.
    fractions = np.stack(
        [aware["wm_fraction"], aware["gm_fraction"], aware["csf_fraction"]], axis=1
    )
    npt.assert_allclose(fractions.sum(axis=1), 1.0, atol=1e-5)
    npt.assert_allclose(fractions.mean(axis=0), [0.5, 0.25, 0.25], atol=0.03)
    counts = np.bincount(aware["num_fibers"].astype(int), minlength=4)[1:]
    npt.assert_allclose(counts / counts.sum(), [0.1, 0.2, 0.7], atol=0.04)


def _unit_rows(signals):
    signals = np.asarray(signals, dtype=np.float64)
    return signals / np.linalg.norm(signals, axis=1, keepdims=True)


def test_relaxation_changes_normalized_signal_only_if_compartments_differ():
    """Distinct relaxation times reshape the signal; equal ones only scale."""
    from dipy.sims.force import generate_force_simulations, relaxation_weights

    gtab = _make_gtab([1000, 2000])
    kwargs = {
        "num_simulations": 60,
        "batch_size": 20,
        "compute_dti": False,
        "verbose": False,
        "dtype": np.float64,
    }
    original = generate_force_simulations(gtab, **kwargs)

    same = {c: {"t1": 1.0, "t2": 0.08} for c in ("wm", "gm", "csf")}
    common = generate_force_simulations(
        gtab,
        te=0.1,
        tr=3.0,
        relaxation_model="tissue",
        relaxation_params=same,
        **kwargs,
    )
    weights = relaxation_weights(te=0.1, tr=3.0, relaxation_params=same)
    assert np.ptp(weights) == 0
    npt.assert_allclose(common["signals"], weights[0] * original["signals"])
    npt.assert_allclose(
        _unit_rows(common["signals"]), _unit_rows(original["signals"]), atol=1e-12
    )

    aware = generate_force_simulations(
        gtab, te=0.1, tr=3.0, relaxation_model="tissue", **kwargs
    )
    shape_change = np.abs(
        _unit_rows(aware["signals"]) - _unit_rows(original["signals"])
    ).max(axis=1)
    assert np.median(shape_change) > 1e-3

    # Longer TE moves the normalized signal further from the original, and a
    # different TR gives yet another library.
    longer = generate_force_simulations(
        gtab, te=0.14, tr=3.0, relaxation_model="tissue", **kwargs
    )
    longer_change = np.abs(
        _unit_rows(longer["signals"]) - _unit_rows(original["signals"])
    ).max(axis=1)
    assert np.median(longer_change) > np.median(shape_change)
    other_tr = generate_force_simulations(
        gtab, te=0.1, tr=8.0, relaxation_model="tissue", **kwargs
    )
    assert not np.allclose(
        _unit_rows(other_tr["signals"]), _unit_rows(aware["signals"]), atol=1e-6
    )
