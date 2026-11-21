"""
FORCE: Forward matching for complex microstructure estimation

This module provides signal simulation for matching-based diffusion MRI
reconstruction using multi-compartment tissue models.

The FORCE method generates a library of simulated signals representing
various tissue configurations (white matter with 1-3 fibers, gray matter,
CSF) and uses this library to reconstruct microstructural parameters
from real diffusion MRI data.

References
----------
:footcite:p:`Shah2026`
"""

import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import warnings

import numpy as np

from dipy.data import default_sphere
from dipy.reconst.bingham import _single_bingham_to_sf as bingham_to_sf
from dipy.sims._force_core import (
    DEFAULT_THREE_FIBER_MIN_ANGLE,
    DEFAULT_TWO_FIBER_MIN_ANGLE,
)
from dipy.sims.voxel import all_tensor_evecs
from dipy.utils.deprecator import deprecated_params, warning_for_keywords
from dipy.utils.logging import logger
from dipy.utils.multiproc import determine_num_processes

# Reference ODI sampling grid. The library samples orientation-dispersion
# values from ``linspace(*odi_range, num_odi_values)``; these constants define
# the default grid and its density, which ``resolve_num_odi_values`` uses to
# autoscale the number of grid points to any custom ``odi_range``.
DEFAULT_ODI_RANGE = (0.01, 0.3)
DEFAULT_NUM_ODI_VALUES = 10
DEFAULT_FORCE_SEED = 2298

# Optional TE/TR-aware simulation -------------------------------------------
# Relaxation models understood by ``resolve_relaxation``.
RELAXATION_MODELS = ("tissue",)
# Order of the relaxation weights handed to the Cython simulator.
RELAXATION_COMPARTMENTS = ("wm_intra", "wm_extra", "gm", "csf")
# Default relaxation times in SECONDS for human brain at 3 T. They are kept
# separate from the diffusivity/microstructure priors and are not fitted.
#   wm/gm t1 : in vivo at 3 T (Wansapura et al. 1999).
#   wm/gm t2 : in vivo CPMG at 3 T, mean of frontal and occipital regions
#              (Lu et al. 2005).
#   csf t1   : in vivo inversion recovery at 3 T (Lu et al. 2005).
#   csf t2   : in vivo ventricular CSF at 3 T (Spijkerman et al. 2018).
DEFAULT_RELAXATION_PARAMS = {
    "wm": {"t1": 0.832, "t2": 0.075},
    "gm": {"t1": 1.331, "t2": 0.083},
    "csf": {"t1": 3.817, "t2": 2.03},
}
# Upper plausibility bounds in seconds. A value above its bound is almost
# certainly expressed in milliseconds and is rejected instead of rescaled.
_MAX_TE = 1.0
_MAX_TR = 100.0
_MAX_RELAXATION_TIME = 10.0


def _check_seconds(name, value, upper):
    """Return ``value`` as a float after checking it is a time in seconds.

    Parameters
    ----------
    name : str
        Name used in error messages.
    value : float
        Time in seconds.
    upper : float
        Largest plausible value in seconds.

    Returns
    -------
    value : float
        Validated time in seconds.
    """
    try:
        value = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be a number in seconds, got {value!r}") from e
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    if value > upper:
        raise ValueError(
            f"{name}={value:g} is not plausible in seconds (expected <= "
            f"{upper:g} s). FORCE expects seconds, as in BIDS; a value in "
            f"milliseconds must be divided by 1000, e.g. {value / 1000:g}."
        )
    return value


def validate_te_tr(*, te=None, tr=None):
    """Validate echo and repetition times given in seconds.

    Parameters
    ----------
    te : float or None, optional
        Echo time in seconds.
    tr : float or None, optional
        Repetition time in seconds.

    Returns
    -------
    te : float or None
        Validated echo time in seconds.
    tr : float or None
        Validated repetition time in seconds.

    Raises
    ------
    ValueError
        If a value is not positive and finite, looks like milliseconds, or
        if ``te >= tr``.
    """
    if te is not None:
        te = _check_seconds("te", te, _MAX_TE)
    if tr is not None:
        tr = _check_seconds("tr", tr, _MAX_TR)
    if te is not None and tr is not None and te >= tr:
        raise ValueError(f"te ({te:g} s) must be shorter than tr ({tr:g} s)")
    return te, tr


def get_default_relaxation_params():
    """Get the default compartment relaxation times.

    Returns
    -------
    params : dict
        ``{"wm": {"t1", "t2"}, "gm": {...}, "csf": {...}}`` in seconds, for
        human brain at 3 T :footcite:p:`Wansapura1999`, :footcite:p:`Lu2005`,
        :footcite:p:`Spijkerman2018`.

    References
    ----------
    .. footbibliography::
    """
    return {k: dict(v) for k, v in DEFAULT_RELAXATION_PARAMS.items()}


def validate_relaxation_params(params):
    """Validate relaxation times and expand them to the FORCE compartments.

    Parameters
    ----------
    params : dict or None
        Relaxation times in seconds, keyed by ``"wm"``, ``"gm"``, ``"csf"``
        and optionally ``"wm_intra"`` / ``"wm_extra"``, each a dict with
        ``"t1"`` and/or ``"t2"``. Missing entries fall back to
        :func:`get_default_relaxation_params`; ``"wm_intra"`` and
        ``"wm_extra"`` fall back to ``"wm"``. ``None`` gives the defaults.

    Returns
    -------
    resolved : dict
        ``{compartment: {"t1": float, "t2": float}}`` for each of
        ``RELAXATION_COMPARTMENTS``.

    Raises
    ------
    ValueError
        On unknown keys, non-positive values, values that look like
        milliseconds, or ``t2 > t1``.
    """
    params = {} if params is None else params
    allowed = {"wm", "gm", "csf", "wm_intra", "wm_extra"}
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(
            f"Unknown relaxation compartment(s) {sorted(unknown)}; "
            f"expected a subset of {sorted(allowed)}"
        )
    for name, times in params.items():
        extra = set(times) - {"t1", "t2"}
        if extra:
            raise ValueError(
                f"Unknown relaxation key(s) {sorted(extra)} for '{name}'; "
                "expected 't1' and/or 't2'"
            )

    resolved = {}
    for name in RELAXATION_COMPARTMENTS:
        tissue = "wm" if name.startswith("wm") else name
        times = dict(DEFAULT_RELAXATION_PARAMS[tissue])
        times.update(params.get(tissue, {}))
        times.update(params.get(name, {}))
        t1 = _check_seconds(f"{name} t1", times["t1"], _MAX_RELAXATION_TIME)
        t2 = _check_seconds(f"{name} t2", times["t2"], _MAX_RELAXATION_TIME)
        if t2 > t1:
            raise ValueError(f"{name}: t2 ({t2:g} s) cannot exceed t1 ({t1:g} s)")
        resolved[name] = {"t1": t1, "t2": t2}
    return resolved


def relaxation_weights(*, te=None, tr=None, relaxation_params=None):
    r"""Spin-echo relaxation weight of each FORCE compartment.

    For compartment $c$ the weight is

    .. math::

        R_c(TE, TR) = \left(1 - e^{-TR / T_{1,c}}\right) e^{-TE / T_{2,c}}

    A missing ``te`` drops the $T_2$ factor and a missing ``tr`` drops the
    $T_1$ factor: no acquisition value is assumed. This function only
    evaluates the weights; :func:`resolve_relaxation` decides whether a
    partial set of weights should be used for simulation.

    Parameters
    ----------
    te : float or None, optional
        Echo time in seconds.
    tr : float or None, optional
        Repetition time in seconds.
    relaxation_params : dict or None, optional
        Relaxation times in seconds, see :func:`validate_relaxation_params`.

    Returns
    -------
    weights : ndarray (4,)
        Weights ordered as ``RELAXATION_COMPARTMENTS``
        (``wm_intra``, ``wm_extra``, ``gm``, ``csf``).

    Notes
    -----
    The $T_1$ factor is the long-TR spin-echo approximation, valid for
    $TE \ll TR$. Proton-density differences between compartments are not
    modeled.
    """
    te, tr = validate_te_tr(te=te, tr=tr)
    params = validate_relaxation_params(relaxation_params)
    weights = np.ones(len(RELAXATION_COMPARTMENTS), dtype=np.float64)
    for i, name in enumerate(RELAXATION_COMPARTMENTS):
        if te is not None:
            weights[i] *= np.exp(-te / params[name]["t2"])
        if tr is not None:
            weights[i] *= 1.0 - np.exp(-tr / params[name]["t1"])
    return weights


def resolve_relaxation(
    *, te=None, tr=None, relaxation_model=None, relaxation_params=None
):
    """Decide whether and how relaxation weighting is applied.

    Parameters
    ----------
    te : float or None, optional
        Echo time in seconds.
    tr : float or None, optional
        Repetition time in seconds.
    relaxation_model : str or None, optional
        ``None`` disables relaxation weighting, whatever ``te`` and ``tr``
        are. ``"tissue"`` weights the WM, GM and CSF compartment signals by
        :func:`relaxation_weights`.
    relaxation_params : dict or None, optional
        Relaxation times in seconds, see :func:`validate_relaxation_params`.

    Returns
    -------
    relaxation : dict or None
        ``None`` when the original FORCE simulation applies: the model is
        disabled, or it is enabled but ``te`` is unknown (a ``UserWarning``
        is issued in that case). Otherwise a dict with ``"model"``,
        ``"te"``, ``"tr"``, ``"terms"`` (the applied factors, ``("t2",)`` or
        ``("t2", "t1")``), ``"params"`` and ``"weights"``.

    Notes
    -----
    Incomplete timing is handled as follows.

    * ``te`` only: the $T_2$ weighting is applied and a ``UserWarning``
      states that $T_1$ saturation is not modeled. Long-$T_1$ CSF is then
      treated as fully recovered and its fraction is underestimated; the
      effect shrinks as TR grows.
    * ``tr`` only: nothing is applied. In brain tissue the $T_1$ factor
      attenuates CSF while the unmodeled $T_2$ factor favors it by a larger
      amount, so a $T_1$-only correction moves the simulated signals away
      from the measured ones.
    """
    if relaxation_model is None:
        return None
    if relaxation_model not in RELAXATION_MODELS:
        raise ValueError(
            f"relaxation_model must be None or one of {RELAXATION_MODELS}, "
            f"got {relaxation_model!r}"
        )
    te, tr = validate_te_tr(te=te, tr=tr)
    params = validate_relaxation_params(relaxation_params)
    if te is None:
        reason = (
            "neither te nor tr is available"
            if tr is None
            else "te is not available and a T1-only (tr) correction would "
            "increase the mismatch with the measured signal"
        )
        warnings.warn(
            f"relaxation_model={relaxation_model!r} was requested but "
            f"{reason}. No relaxation weighting is applied and the original "
            "FORCE simulation is used.",
            UserWarning,
            stacklevel=3,
        )
        return None
    if tr is None:
        warnings.warn(
            "tr is not available: only the T2 (te) weighting is applied. "
            "T1 saturation is not modeled, so the CSF fraction is "
            "underestimated unless tr is much longer than the CSF T1.",
            UserWarning,
            stacklevel=3,
        )
    terms = ("t2",) if tr is None else ("t2", "t1")
    return {
        "model": relaxation_model,
        "te": te,
        "tr": tr,
        "terms": terms,
        "params": params,
        "weights": relaxation_weights(te=te, tr=tr, relaxation_params=params),
    }


@warning_for_keywords(from_version="1.13.0")
def resolve_num_odi_values(odi_range, *, num_odi_values=None):
    """Resolve the number of ODI grid points, autoscaling when unset.

    Parameters
    ----------
    odi_range : tuple
        ``(min, max)`` orientation-dispersion-index range.
    num_odi_values : int or None, optional
        Explicit number of grid points. When ``None``, the grid is autoscaled
        so its spacing matches the default grid
        (``DEFAULT_NUM_ODI_VALUES`` points across ``DEFAULT_ODI_RANGE``),
        keeping sampling density constant as ``odi_range`` widens or narrows.

    Returns
    -------
    n : int
        Number of ODI grid points (always >= 2).
    """
    if num_odi_values is not None:
        n = int(num_odi_values)
        if n < 2:
            raise ValueError(f"num_odi_values must be >= 2; got {n}.")
        return n
    lo, hi = float(odi_range[0]), float(odi_range[1])
    default_lo, default_hi = DEFAULT_ODI_RANGE
    spacing = (default_hi - default_lo) / (DEFAULT_NUM_ODI_VALUES - 1)
    return max(2, int(round((hi - lo) / spacing)) + 1)


def dispersion_lut(target_sphere, odi_list):
    """Generate spherical functions for all directions and ODI values.
    Currently uses Bingham distribution, but can be extended to other
    dispersion models in the future.

    Parameters
    ----------
    target_sphere : ndarray (N, 3)
        Unit sphere vertices.
    odi_list : ndarray
        List of orientation dispersion index values.

    Returns
    -------
    SF : dict
        Nested dictionary mapping (vertex_index, odi) to spherical function (SF)
        values.
    """
    dispersion_sf_lut = {}
    for i in range(len(target_sphere)):
        vertex_key = tuple(target_sphere[i])
        dispersion_sf_lut[i] = {}

        for odi in odi_list:
            k = 1 / np.tan(np.pi / 2 * odi)
            evecs = all_tensor_evecs(vertex_key)
            major_axis, minor_axis = evecs[:, 1], evecs[:, 2]

            dispersion_sf_lut[i][odi] = bingham_to_sf(
                1, k, k, major_axis, minor_axis, target_sphere
            )

    return dispersion_sf_lut


def smallest_shell_bval(bvals, *, b0_threshold=50, shell_tolerance=50, n=1):
    """Find the smallest non-zero b-value shell.

    Parameters
    ----------
    bvals : ndarray
        B-values array.
    b0_threshold : float, optional
        Maximum b-value for b0 volumes.
    shell_tolerance : float, optional
        Tolerance for grouping shells.
    n : int, optional
        Number of smallest shells to return.

    Returns
    -------
    min_shell : float
        Smallest non-zero shell b-value.
    shell_mask : ndarray
        Boolean mask for volumes in that shell.
    """
    bvals = np.asarray(bvals, dtype=float)
    non_b0 = bvals > b0_threshold
    if not np.any(non_b0):
        raise ValueError("No non-b0 volumes found.")
    rounded = np.round(bvals[non_b0] / shell_tolerance) * shell_tolerance
    unique_shells = np.unique(rounded)
    if len(unique_shells) < n:
        raise ValueError(
            f"Only {len(unique_shells)} unique shells found, but n={n} requested."
        )
    else:
        min_shells = unique_shells[:n]

    rounded_all = np.round(bvals / shell_tolerance) * shell_tolerance
    shell_mask = np.isin(rounded_all, min_shells)
    return min_shells, shell_mask


@warning_for_keywords(from_version="1.13.0")
@deprecated_params("base_seed", since="1.13.0", until="2.0.0")
def init_worker(*, base_seed=None):
    """Initialize a ProcessPoolExecutor worker with a unique RNG state.

    Seeds the global :mod:`numpy.random` and :mod:`random` states from the
    PID and high-resolution time so forked workers do not inherit identical
    streams from the parent.

    Parameters
    ----------
    base_seed : int or None, optional
        Deprecated and ignored.
    """
    import random
    import time

    # Unique per process and run: PID + high-resolution time
    seed = (os.getpid() * (2**16) + int(time.perf_counter_ns())) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _create_memmap(output_dir, name, dtype, shape):
    """Create a memory-mapped file."""
    path = output_dir / f"{name}.mmap"
    return np.memmap(path, mode="w+", dtype=dtype, shape=shape), path


def _generate_batch_worker(
    start_idx,
    batch_size,
    batch_seed,
    sphere,
    evecs,
    bingham,
    odi,
    bval,
    bvec,
    wm_thresh,
    tort,
    memmap_info,
    diffusivity_cfg,
    min_crossing_angles,
    relaxation_weights=None,
):
    """Worker function for parallel batch generation."""
    from dipy.sims._force_core import create_mixed_signal, set_diffusivity_ranges
    from dipy.sims._multi_tensor_omp import multi_tensor

    if batch_seed is not None:
        np.random.seed(batch_seed)

    # Unpack memmap info
    (
        signals_path,
        signals_shape,
        signals_dtype,
        labels_path,
        labels_shape,
        labels_dtype,
        num_fibers_path,
        num_fibers_shape,
        num_fibers_dtype,
        dispersion_path,
        dispersion_shape,
        dispersion_dtype,
        wm_fraction_path,
        wm_fraction_shape,
        wm_fraction_dtype,
        gm_fraction_path,
        gm_fraction_shape,
        gm_fraction_dtype,
        csf_fraction_path,
        csf_fraction_shape,
        csf_fraction_dtype,
        nd_path,
        nd_shape,
        nd_dtype,
        odfs_path,
        odfs_shape,
        odfs_dtype,
        ufa_wm_path,
        ufa_wm_shape,
        ufa_wm_dtype,
        ufa_voxel_path,
        ufa_voxel_shape,
        ufa_voxel_dtype,
        fraction_array_path,
        fraction_array_shape,
        fraction_array_dtype,
        wm_disp_path,
        wm_disp_shape,
        wm_disp_dtype,
        wm_d_par_path,
        wm_d_par_shape,
        wm_d_par_dtype,
        wm_d_perp_path,
        wm_d_perp_shape,
        wm_d_perp_dtype,
        gm_d_par_path,
        gm_d_par_shape,
        gm_d_par_dtype,
        csf_d_par_path,
        csf_d_par_shape,
        csf_d_par_dtype,
        f_ins_path,
        f_ins_shape,
        f_ins_dtype,
    ) = memmap_info

    # Apply diffusivity ranges in worker
    set_diffusivity_ranges(**diffusivity_cfg)

    # Open memmaps in read-write mode
    signals_mm = np.memmap(
        signals_path, mode="r+", dtype=signals_dtype, shape=signals_shape
    )
    labels_mm = np.memmap(
        labels_path, mode="r+", dtype=labels_dtype, shape=labels_shape
    )
    num_fibers_mm = np.memmap(
        num_fibers_path, mode="r+", dtype=num_fibers_dtype, shape=num_fibers_shape
    )
    dispersion_mm = np.memmap(
        dispersion_path, mode="r+", dtype=dispersion_dtype, shape=dispersion_shape
    )
    wm_fraction_mm = np.memmap(
        wm_fraction_path, mode="r+", dtype=wm_fraction_dtype, shape=wm_fraction_shape
    )
    gm_fraction_mm = np.memmap(
        gm_fraction_path, mode="r+", dtype=gm_fraction_dtype, shape=gm_fraction_shape
    )
    csf_fraction_mm = np.memmap(
        csf_fraction_path, mode="r+", dtype=csf_fraction_dtype, shape=csf_fraction_shape
    )
    nd_mm = np.memmap(nd_path, mode="r+", dtype=nd_dtype, shape=nd_shape)
    odfs_mm = np.memmap(odfs_path, mode="r+", dtype=odfs_dtype, shape=odfs_shape)
    ufa_wm_mm = np.memmap(
        ufa_wm_path, mode="r+", dtype=ufa_wm_dtype, shape=ufa_wm_shape
    )
    ufa_voxel_mm = np.memmap(
        ufa_voxel_path, mode="r+", dtype=ufa_voxel_dtype, shape=ufa_voxel_shape
    )
    fraction_array_mm = np.memmap(
        fraction_array_path,
        mode="r+",
        dtype=fraction_array_dtype,
        shape=fraction_array_shape,
    )
    wm_disp_mm = np.memmap(
        wm_disp_path, mode="r+", dtype=wm_disp_dtype, shape=wm_disp_shape
    )
    wm_d_par_mm = np.memmap(
        wm_d_par_path, mode="r+", dtype=wm_d_par_dtype, shape=wm_d_par_shape
    )
    wm_d_perp_mm = np.memmap(
        wm_d_perp_path, mode="r+", dtype=wm_d_perp_dtype, shape=wm_d_perp_shape
    )
    gm_d_par_mm = np.memmap(
        gm_d_par_path, mode="r+", dtype=gm_d_par_dtype, shape=gm_d_par_shape
    )
    csf_d_par_mm = np.memmap(
        csf_d_par_path, mode="r+", dtype=csf_d_par_dtype, shape=csf_d_par_shape
    )
    f_ins_mm = np.memmap(f_ins_path, mode="r+", dtype=f_ins_dtype, shape=f_ins_shape)

    for i in range(batch_size):
        idx = start_idx + i
        res = create_mixed_signal(
            sphere,
            evecs,
            bingham,
            odi,
            bval,
            bvec,
            multi_tensor,
            wm_thresh,
            tort,
            min_crossing_angles[0],
            min_crossing_angles[1],
            relaxation_weights,
        )
        (
            signals_mm[idx],
            labels_mm[idx],
            num_fibers_mm[idx],
            dispersion_mm[idx],
            wm_fraction_mm[idx],
            gm_fraction_mm[idx],
            csf_fraction_mm[idx],
            nd_mm[idx],
            odfs_mm[idx],
            ufa_wm_mm[idx],
            ufa_voxel_mm[idx],
            fraction_array_mm[idx],
            wm_disp_mm[idx],
            wm_d_par_mm[idx],
            wm_d_perp_mm[idx],
            gm_d_par_mm[idx],
            csf_d_par_mm[idx],
            f_ins_mm[idx],
        ) = res

    # Flush memmaps
    for mm in [
        signals_mm,
        labels_mm,
        num_fibers_mm,
        dispersion_mm,
        wm_fraction_mm,
        gm_fraction_mm,
        csf_fraction_mm,
        nd_mm,
        odfs_mm,
        ufa_wm_mm,
        ufa_voxel_mm,
        fraction_array_mm,
        wm_disp_mm,
        wm_d_par_mm,
        wm_d_perp_mm,
        gm_d_par_mm,
        csf_d_par_mm,
        f_ins_mm,
    ]:
        mm.flush()

    return batch_size


def _main_is_guarded():
    """Return True if ``__main__`` has a top-level ``if __name__ == '__main__':`` guard.

    Parses the source of the ``__main__`` module with :mod:`ast` and inspects
    only the direct children of the module node (top-level statements).  Returns
    ``True`` also when the main module was invoked with ``python -m pkg.mod``
    (``__spec__`` is not None), or from an interactive session (no ``__file__``),
    because those cases do not cause spawn re-execution problems.

    Returns
    -------
    guarded : bool
        ``True`` if spawning workers is safe without causing recursive
        re-execution of the caller's script.
    """
    main = sys.modules.get("__main__")
    if main is None:
        return True
    if getattr(main, "__spec__", None) is not None:
        # Invoked as `python -m pkg.mod` — spawn imports the module, no re-run
        return True
    main_file = getattr(main, "__file__", None)
    if main_file is None:
        # Interactive session or frozen executable
        return True
    try:
        with open(main_file) as fh:
            source = fh.read()
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return False
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        ):
            return True
    return False


def generate_force_simulations(
    gtab,
    *,
    num_simulations=100000,
    output_dir=None,
    num_cpus=1,
    batch_size=1000,
    seed=DEFAULT_FORCE_SEED,
    wm_threshold=0.5,
    tortuosity=False,
    odi_range=DEFAULT_ODI_RANGE,
    num_odi_values=None,
    two_fiber_min_angle=DEFAULT_TWO_FIBER_MIN_ANGLE,
    three_fiber_min_angle=DEFAULT_THREE_FIBER_MIN_ANGLE,
    diffusivity_config=None,
    dtype=np.float32,
    compute_dti=True,
    compute_dki=False,
    verbose=True,
    te=None,
    tr=None,
    relaxation_model=None,
    relaxation_params=None,
):
    """Generate FORCE simulations.

    Creates a library of simulated diffusion MRI signals with
    corresponding microstructural parameters for matching-based
    reconstruction.

    Parameters
    ----------
    gtab : GradientTable
        Gradient table with b-values and b-vectors.
    num_simulations : int, optional
        Number of simulated voxels.
    output_dir : str or Path, optional
        Directory for output files. If None, uses a temporary directory.
    num_cpus : int or None, optional
        Number of CPU cores for parallel processing.
        ``1`` runs in-process (no subprocess, safe everywhere).
        ``None`` or ``-1`` uses all available cores.
        Values ``< -1`` leave that many cores idle
        (e.g. ``-2`` = all minus one). ``0`` raises ``ValueError``.
    batch_size : int, optional
        Batch size for processing.
    seed : int or None, optional
        Seed for reproducible simulation generation. Each batch draws from
        an independent random stream derived from ``(seed, batch_index)``,
        so identical seeds yield identical libraries for any ``num_cpus``.
        Note that changing ``batch_size`` changes the batch layout and
        therefore the library. When None, every run uses fresh entropy.
    wm_threshold : float, optional
        Minimum WM fraction to include fiber labels.
    tortuosity : bool, optional
        Use tortuosity constraint for perpendicular diffusivity.
    odi_range : tuple, optional
        (min, max) orientation dispersion index range.
    num_odi_values : int or None, optional
        Number of ODI values in the sampling grid. When ``None`` (default),
        the grid is autoscaled from ``odi_range`` to keep the sampling density
        constant (see :func:`resolve_num_odi_values`).
    two_fiber_min_angle : float, optional
        Minimum crossing angle for two-fiber simulations, in degrees. Crossings
        below this angle are never simulated. Use ``0`` to allow any crossing.
    three_fiber_min_angle : float, optional
        Minimum pairwise crossing angle for three-fiber simulations, in
        degrees.
    diffusivity_config : dict, optional
        Custom diffusivity ranges.
    dtype : dtype, optional
        Data type for outputs.
    compute_dti : bool, optional
        Compute DTI metrics (FA, MD, RD).
    compute_dki : bool, optional
        Compute DKI metrics (AK, RK, MK, KFA).
    verbose : bool, optional
        Enable progress output.
    te : float or None, optional
        Echo time in seconds. Only used when ``relaxation_model`` is set.
    tr : float or None, optional
        Repetition time in seconds. Only used when ``relaxation_model`` is
        set.
    relaxation_model : str or None, optional
        ``None`` (default) generates the original FORCE library. ``"tissue"``
        multiplies every compartment signal by its spin-echo relaxation
        weight (see :func:`relaxation_weights`) before the compartments are
        mixed. The sampled microstructure, and therefore every stored
        parameter except the signals and the signal-derived DTI/DKI metrics,
        is identical to the original library for the same seed. If neither
        ``te`` nor ``tr`` is given the original library is generated and a
        warning is issued.
    relaxation_params : dict or None, optional
        Compartment relaxation times in seconds, see
        :func:`validate_relaxation_params`. Defaults to
        :func:`get_default_relaxation_params`.

    Returns
    -------
    simulations : dict
        Simulations containing signals and parameters.
    """
    from tqdm import tqdm

    from dipy.reconst import dti
    from dipy.sims._force_core import set_diffusivity_ranges

    num_cpus = determine_num_processes(num_cpus)

    # Setup output directory
    _tmpdir_ctx = None
    if output_dir is None:
        _tmpdir_ctx = tempfile.TemporaryDirectory(prefix="force_sim_")
        output_dir = _tmpdir_ctx.name
    os.makedirs(output_dir, exist_ok=True)

    # Setup diffusivity config
    if diffusivity_config is None:
        diffusivity_config = {
            "wm_d_par_range": (2.0e-3, 3.0e-3),
            "wm_d_perp_range": (0.3e-3, 1.5e-3),
            "gm_d_iso_range": (0.7e-3, 1.2e-3),
            "csf_d": 3.0e-3,
        }

    set_diffusivity_ranges(**diffusivity_config)

    for name, angle in (
        ("two_fiber_min_angle", two_fiber_min_angle),
        ("three_fiber_min_angle", three_fiber_min_angle),
    ):
        if not 0 <= angle < 90:
            raise ValueError(f"{name} must be in [0, 90) degrees, got {angle}")
    min_crossing_angles = (float(two_fiber_min_angle), float(three_fiber_min_angle))

    relaxation = resolve_relaxation(
        te=te,
        tr=tr,
        relaxation_model=relaxation_model,
        relaxation_params=relaxation_params,
    )
    relaxation_w = None if relaxation is None else relaxation["weights"]

    # Setup sphere and ODI values
    sphere = default_sphere
    target_sphere = np.ascontiguousarray(sphere.vertices, dtype=np.float64)
    n_dirs = target_sphere.shape[0]

    bvals = np.ascontiguousarray(gtab.bvals.astype(np.float64))
    bvecs = np.ascontiguousarray(gtab.bvecs.astype(np.float64))
    n_bvals = bvals.shape[0]

    # Pre-compute eigenvectors and dispersion lookup table
    evecs = np.array(
        [all_tensor_evecs(tuple(point)) for point in target_sphere],
        dtype=np.float64,
    )
    num_odi_values = resolve_num_odi_values(odi_range, num_odi_values=num_odi_values)
    odi_list = np.linspace(odi_range[0], odi_range[1], num_odi_values).astype(
        np.float64
    )
    if verbose:
        logger.info(
            f"ODI sampling grid: {num_odi_values} values over "
            f"[{odi_range[0]:.3g}, {odi_range[1]:.3g}]"
        )
    dispersion_sf = dispersion_lut(target_sphere, odi_list)

    label_dtype = np.uint8

    # Create memmaps for large arrays
    memmap_paths = {}

    def create_mm(_name, dt, shape):
        _mm, _path = _create_memmap(Path(output_dir), _name, dt, shape)
        memmap_paths[_name] = _path
        return _mm

    signals_mm = create_mm("signals", dtype, (num_simulations, n_bvals))
    labels_mm = create_mm("labels", label_dtype, (num_simulations, n_dirs))
    num_fibers_mm = create_mm("num_fibers", dtype, (num_simulations,))
    dispersion_mm = create_mm("dispersion", dtype, (num_simulations,))
    wm_fraction_mm = create_mm("wm_fraction", dtype, (num_simulations,))
    gm_fraction_mm = create_mm("gm_fraction", dtype, (num_simulations,))
    csf_fraction_mm = create_mm("csf_fraction", dtype, (num_simulations,))
    nd_mm = create_mm("nd", dtype, (num_simulations,))
    odfs_mm = create_mm("odfs", np.float16, (num_simulations, n_dirs))
    ufa_wm_mm = create_mm("ufa_wm", dtype, (num_simulations,))
    ufa_voxel_mm = create_mm("ufa_voxel", dtype, (num_simulations,))
    fraction_array_mm = create_mm("fraction_array", dtype, (num_simulations, 3))
    wm_disp_mm = create_mm("wm_disp", dtype, (num_simulations,))
    wm_d_par_mm = create_mm("wm_d_par", dtype, (num_simulations,))
    wm_d_perp_mm = create_mm("wm_d_perp", dtype, (num_simulations,))
    gm_d_par_mm = create_mm("gm_d_par", dtype, (num_simulations,))
    csf_d_par_mm = create_mm("csf_d_par", dtype, (num_simulations,))
    f_ins_mm = create_mm("f_ins", dtype, (num_simulations, 3))

    memmaps = [
        signals_mm,
        labels_mm,
        num_fibers_mm,
        dispersion_mm,
        wm_fraction_mm,
        gm_fraction_mm,
        csf_fraction_mm,
        nd_mm,
        odfs_mm,
        ufa_wm_mm,
        ufa_voxel_mm,
        fraction_array_mm,
        wm_disp_mm,
        wm_d_par_mm,
        wm_d_perp_mm,
        gm_d_par_mm,
        csf_d_par_mm,
        f_ins_mm,
    ]

    # Build batch specs
    num_batches_full = num_simulations // batch_size
    remainder = num_simulations % batch_size
    total_batches = num_batches_full + (1 if remainder > 0 else 0)

    batch_specs = []
    current_start = 0
    for batch_idx in range(total_batches):
        bs = batch_size if batch_idx < num_batches_full else remainder
        if seed is None:
            batch_seed = None
        else:
            batch_seed = int(
                np.random.SeedSequence([int(seed), batch_idx]).generate_state(1)[0]
            )
        batch_specs.append((current_start, bs, batch_seed))
        current_start += bs

    # Pack memmap info for workers
    memmap_info = (
        memmap_paths["signals"],
        (num_simulations, n_bvals),
        dtype,
        memmap_paths["labels"],
        (num_simulations, n_dirs),
        label_dtype,
        memmap_paths["num_fibers"],
        (num_simulations,),
        dtype,
        memmap_paths["dispersion"],
        (num_simulations,),
        dtype,
        memmap_paths["wm_fraction"],
        (num_simulations,),
        dtype,
        memmap_paths["gm_fraction"],
        (num_simulations,),
        dtype,
        memmap_paths["csf_fraction"],
        (num_simulations,),
        dtype,
        memmap_paths["nd"],
        (num_simulations,),
        dtype,
        memmap_paths["odfs"],
        (num_simulations, n_dirs),
        np.float16,
        memmap_paths["ufa_wm"],
        (num_simulations,),
        dtype,
        memmap_paths["ufa_voxel"],
        (num_simulations,),
        dtype,
        memmap_paths["fraction_array"],
        (num_simulations, 3),
        dtype,
        memmap_paths["wm_disp"],
        (num_simulations,),
        dtype,
        memmap_paths["wm_d_par"],
        (num_simulations,),
        dtype,
        memmap_paths["wm_d_perp"],
        (num_simulations,),
        dtype,
        memmap_paths["gm_d_par"],
        (num_simulations,),
        dtype,
        memmap_paths["csf_d_par"],
        (num_simulations,),
        dtype,
        memmap_paths["f_ins"],
        (num_simulations, 3),
        dtype,
    )

    # Run simulations with progress bar
    pbar = tqdm(total=num_simulations, desc="Simulating", disable=not verbose)

    def run_serial():
        """Run batches in-process without altering the caller's NumPy RNG."""
        state = np.random.get_state()
        try:
            if seed is None:
                init_worker()
            for start_idx, bs, batch_seed in batch_specs:
                batch_done = _generate_batch_worker(
                    start_idx,
                    bs,
                    batch_seed,
                    target_sphere,
                    evecs,
                    dispersion_sf,
                    odi_list,
                    bvals,
                    bvecs,
                    wm_threshold,
                    tortuosity,
                    memmap_info,
                    diffusivity_config,
                    min_crossing_angles,
                    relaxation_w,
                )
                pbar.update(batch_done)
        finally:
            np.random.set_state(state)

    if num_cpus == 1:
        # Serial in-process path — no subprocess spawned, safe on all platforms
        run_serial()
    elif sys.platform == "win32":
        if _main_is_guarded():
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=num_cpus,
                initializer=init_worker if seed is None else None,
                mp_context=ctx,
            ) as executor:
                futures = {
                    executor.submit(
                        _generate_batch_worker,
                        start_idx,
                        bs,
                        batch_seed,
                        target_sphere,
                        evecs,
                        dispersion_sf,
                        odi_list,
                        bvals,
                        bvecs,
                        wm_threshold,
                        tortuosity,
                        memmap_info,
                        diffusivity_config,
                        min_crossing_angles,
                        relaxation_w,
                    ): (start_idx, bs)
                    for start_idx, bs, batch_seed in batch_specs
                }
                for future in as_completed(futures):
                    batch_done = future.result()
                    pbar.update(batch_done)
        else:
            logger.warning(
                "Parallel simulation on Windows requires your script to be "
                "protected with `if __name__ == '__main__':`. "
                "Falling back to serial execution."
            )
            run_serial()
    else:
        # fork: workers are copies of the parent — __main__ is NOT re-run,
        # so no `if __name__ == '__main__':` guard is needed in the caller's
        # script. Safe for CLI scientific-computing scripts on macOS/Linux
        # (no GUI/ObjC).
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=num_cpus,
            initializer=init_worker if seed is None else None,
            mp_context=ctx,
        ) as executor:
            futures = {
                executor.submit(
                    _generate_batch_worker,
                    start_idx,
                    bs,
                    batch_seed,
                    target_sphere,
                    evecs,
                    dispersion_sf,
                    odi_list,
                    bvals,
                    bvecs,
                    wm_threshold,
                    tortuosity,
                    memmap_info,
                    diffusivity_config,
                    min_crossing_angles,
                    relaxation_w,
                ): (start_idx, bs)
                for start_idx, bs, batch_seed in batch_specs
            }

            for future in as_completed(futures):
                batch_done = future.result()
                pbar.update(batch_done)

    pbar.close()

    # Flush memmaps
    for mm in memmaps:
        mm.flush()

    # Compute DTI metrics
    fa_dti = np.zeros(num_simulations, dtype=dtype)
    md_dti = np.zeros(num_simulations, dtype=dtype)
    rd_dti = np.zeros(num_simulations, dtype=dtype)

    # Compute DKI metrics
    ak_arr = np.zeros(num_simulations, dtype=dtype)
    rk_arr = np.zeros(num_simulations, dtype=dtype)
    mk_arr = np.zeros(num_simulations, dtype=dtype)
    kfa_arr = np.zeros(num_simulations, dtype=dtype)

    if compute_dti:
        from dipy.core.gradients import gradient_table as gt_func

        min_b, shell_mask = smallest_shell_bval(bvals)
        b0_mask = bvals <= gtab.b0_threshold
        use_mask = shell_mask | b0_mask

        gtab_small = gt_func(bvals[use_mask], bvecs=bvecs[use_mask])
        dti_model = dti.TensorModel(gtab_small)

        dti_batch_size = 2000
        pbar = tqdm(total=num_simulations, desc="DTI fitting", disable=not verbose)

        for start in range(0, num_simulations, dti_batch_size):
            end = min(start + dti_batch_size, num_simulations)
            data_batch = signals_mm[start:end][:, use_mask]
            dti_fit = dti_model.fit(data_batch)
            fa_dti[start:end] = dti_fit.fa.astype(dtype)
            md_dti[start:end] = dti_fit.md.astype(dtype)
            rd_dti[start:end] = dti_fit.rd.astype(dtype)
            pbar.update(end - start)

        pbar.close()

    if compute_dki:
        from dipy.core.gradients import gradient_table as gt_func
        from dipy.reconst.dki import DiffusionKurtosisModel

        min_bs, shell_mask = smallest_shell_bval(bvals, n=2)
        b0_mask = bvals <= gtab.b0_threshold
        use_mask = shell_mask | b0_mask

        gtab_dki = gt_func(bvals[use_mask], bvecs=bvecs[use_mask])
        dki_model = DiffusionKurtosisModel(gtab_dki)

        dki_batch_size = 2000
        pbar = tqdm(
            total=num_simulations, desc="Kurtosis Estimation", disable=not verbose
        )

        for start in range(0, num_simulations, dki_batch_size):
            end = min(start + dki_batch_size, num_simulations)
            data_batch = signals_mm[start:end][:, use_mask]
            dki_fit = dki_model.multi_fit(data_batch)[0]
            ak_arr[start:end] = dki_fit.ak().astype(dtype)
            rk_arr[start:end] = dki_fit.rk().astype(dtype)
            mk_arr[start:end] = dki_fit.mk().astype(dtype)
            kfa_arr[start:end] = dki_fit.kfa.astype(dtype)
            pbar.update(end - start)

        pbar.close()

    # Build output simulations dict
    simulations = {
        "signals": np.array(signals_mm),
        "labels": np.array(labels_mm),
        "num_fibers": np.array(num_fibers_mm),
        "dispersion": np.array(dispersion_mm),
        "wm_fraction": np.array(wm_fraction_mm),
        "gm_fraction": np.array(gm_fraction_mm),
        "csf_fraction": np.array(csf_fraction_mm),
        "nd": np.array(nd_mm),
        "odfs": np.array(odfs_mm),
        "fa": fa_dti,
        "md": md_dti,
        "rd": rd_dti,
        "ufa_wm": np.array(ufa_wm_mm),
        "ufa_voxel": np.array(ufa_voxel_mm),
        "fraction_array": np.array(fraction_array_mm),
    }

    # Provenance: present only in relaxation-aware libraries.
    if relaxation_w is not None:
        simulations["relaxation_weights"] = relaxation_w

    if compute_dki:
        simulations.update(
            {
                "ak": ak_arr,
                "rk": rk_arr,
                "mk": mk_arr,
                "kfa": kfa_arr,
            }
        )

    # Cleanup memmaps
    for mm in memmaps:
        if hasattr(mm, "base") and hasattr(mm.base, "close"):
            mm.base.close()
        del mm
    gc.collect()

    for path in memmap_paths.values():
        try:
            if path.exists():
                os.remove(path)
        except OSError:
            pass

    gc.collect()

    if _tmpdir_ctx is not None:
        _tmpdir_ctx.cleanup()

    return simulations


def save_force_simulations(simulations, output_path):
    """Save FORCE simulations to compressed NPZ file.

    Parameters
    ----------
    simulations : dict
        FORCE simulations.
    output_path : str
        Path for output file (.npz extension).
    """
    np.savez_compressed(output_path, **simulations)


def load_force_simulations(input_path):
    """Load FORCE simulations from NPZ file.

    Parameters
    ----------
    input_path : str
        Path to simulations file.

    Returns
    -------
    simulations : dict
        FORCE simulations.
    """
    data = np.load(input_path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def validate_diffusivity_config(config):
    """Validate diffusivity configuration dictionary.

    Parameters
    ----------
    config : dict
        Diffusivity configuration with keys:
        wm_d_par_range, wm_d_perp_range, gm_d_iso_range, csf_d.

    Returns
    -------
    valid : bool
        True if configuration is valid.

    Raises
    ------
    ValueError
        If configuration is invalid.
    """
    required_keys = ["wm_d_par_range", "wm_d_perp_range", "gm_d_iso_range", "csf_d"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key: {key}")

    for key in ["wm_d_par_range", "wm_d_perp_range", "gm_d_iso_range"]:
        val = config[key]
        if hasattr(val, "__len__"):
            if len(val) != 2:
                raise ValueError(f"{key} must be scalar or (min, max) tuple")
            if val[0] > val[1]:
                raise ValueError(f"{key} min must be <= max")

    csf_d = config["csf_d"]
    if not isinstance(csf_d, (int, float)) or csf_d <= 0:
        raise ValueError("csf_d must be a positive number")

    return True


def get_default_diffusivity_config():
    """Get default diffusivity configuration.

    Returns
    -------
    config : dict
        Default diffusivity ranges for FORCE simulation.
    """
    return {
        "wm_d_par_range": (2.0e-3, 3.0e-3),
        "wm_d_perp_range": (0.3e-3, 1.5e-3),
        "gm_d_iso_range": (0.7e-3, 1.2e-3),
        "csf_d": 3.0e-3,
    }
