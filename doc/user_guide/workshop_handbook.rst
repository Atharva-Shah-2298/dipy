.. _workshop_handbook:

====================================
DIPY Workshop Handbook for Beginners
====================================

Welcome to the DIPY Workshop! This handbook is designed to guide beginners through
the fundamentals of DIPY and diffusion MRI analysis. Whether you're new to Python,
diffusion imaging, or both, this guide will help you get started.

.. contents:: Table of Contents
   :depth: 3
   :local:

****************
1. Introduction
****************

What is DIPY?
=============

DIPY (Diffusion Imaging in Python) is a free and open-source software library for
the analysis of diffusion MRI (dMRI) data. It provides:

* **Preprocessing tools**: Denoising, artifact correction, registration
* **Reconstruction methods**: DTI, DKI, CSD, SHORE, and many more
* **Tractography algorithms**: Deterministic and probabilistic fiber tracking
* **Visualization tools**: Interactive 3D visualization of brain data
* **Analysis tools**: Connectivity analysis, bundle extraction, statistics

What is Diffusion MRI?
======================

Diffusion MRI is a non-invasive imaging technique that measures the diffusion of
water molecules in biological tissues. In the brain, water diffusion is restricted
by cellular structures, particularly white matter fiber bundles. By analyzing these
diffusion patterns, we can:

* Map white matter pathways in the brain
* Study brain connectivity
* Detect abnormalities in tissue microstructure
* Investigate brain development and diseases

Why Use DIPY?
=============

* **Comprehensive**: State-of-the-art algorithms for all stages of dMRI analysis
* **Well-documented**: Extensive tutorials and examples
* **Active community**: Regular updates and responsive support
* **Research-grade**: Used in numerous scientific publications
* **Free and open-source**: BSD license allows commercial use

***********************
2. Installation & Setup
***********************

Prerequisites
=============

Before installing DIPY, ensure you have:

* **Python 3.10 or later** (Python 3.11 or 3.12 recommended)
* **Basic knowledge of Python** (variables, functions, imports)
* **NumPy, SciPy** (installed automatically with DIPY)

Installing DIPY
===============

**Using pip (recommended for beginners)**::

    pip install dipy

**Using conda**::

    conda install -c conda-forge dipy

**Verifying installation**::

    python -c "import dipy; print(dipy.__version__)"

Setting Up Your Environment
============================

We recommend using:

* **Jupyter Notebook** or **JupyterLab** for interactive analysis::

    pip install jupyter
    jupyter notebook

* **IDE**: Visual Studio Code, PyCharm, or Spyder
* **Virtual environment** to manage dependencies::

    python -m venv dipy_env
    source dipy_env/bin/activate  # On Windows: dipy_env\Scripts\activate
    pip install dipy jupyter matplotlib

Essential Python Libraries
==========================

DIPY works seamlessly with::

    pip install matplotlib nibabel scipy numpy

* **nibabel**: Reading/writing neuroimaging files (NIfTI, DICOM)
* **matplotlib**: Creating plots and visualizations
* **scipy**: Scientific computing tools
* **numpy**: Array operations

***************************
3. Understanding the Basics
***************************

Key Concepts in Diffusion MRI
==============================

**B-values**
    The diffusion weighting factor (in s/mm²). Higher b-values mean stronger
    diffusion weighting. Typical values: b=1000, b=2000, b=3000.

**B-vectors**
    Unit vectors indicating the direction of diffusion gradients. Multiple
    directions are acquired to sample diffusion in 3D space.

**Gradient Table**
    A combination of b-values and b-vectors that describes the diffusion
    acquisition scheme.

**DTI (Diffusion Tensor Imaging)**
    A model that describes diffusion as an ellipsoid (tensor) at each voxel.

**FA (Fractional Anisotropy)**
    A measure of directionality of diffusion (0=isotropic, 1=highly directional).

**MD (Mean Diffusivity)**
    The average diffusion rate in all directions.

**Tractography**
    The process of reconstructing white matter pathways from diffusion data.

DIPY Data Structure
===================

**Main data types**:

* **4D array**: ``data[x, y, z, gradient]`` - the diffusion-weighted images
* **Gradient table**: Contains b-values and b-vectors
* **Affine matrix**: Maps voxel coordinates to real-world coordinates
* **Mask**: Binary array indicating brain vs. background voxels

*****************************
4. Core DIPY Modules Overview
*****************************

DIPY is organized into specialized modules:

``dipy.data``
=============
Access example datasets for learning and testing::

    from dipy.data import get_fnames, read_stanford_labels
    
    # Get example diffusion data
    fraw, fbval, fbvec = get_fnames('stanford_hardi')

``dipy.io``
===========
Input/output operations for neuroimaging data::

    from dipy.io.image import load_nifti, save_nifti
    from dipy.io import read_bvals_bvecs
    
    # Load diffusion data
    data, affine = load_nifti('dwi.nii.gz')
    bvals, bvecs = read_bvals_bvecs('dwi.bval', 'dwi.bvec')
    
    # Save results
    save_nifti('output.nii.gz', result_data, affine)

``dipy.core.gradients``
========================
Create and manipulate gradient tables::

    from dipy.core.gradients import gradient_table
    
    gtab = gradient_table(bvals, bvecs=bvecs)

``dipy.denoise``
================
Image denoising algorithms::

    from dipy.denoise.nlmeans import nlmeans
    from dipy.denoise.noise_estimate import estimate_sigma
    
    sigma = estimate_sigma(data, N=4)
    denoised = nlmeans(data, sigma=sigma)

``dipy.reconst``
================
Reconstruction models for diffusion data::

    from dipy.reconst.dti import TensorModel
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
    
    # DTI model
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask)
    
    # Get FA and color FA
    fa = tenfit.fa
    color_fa = tenfit.color_fa

``dipy.tracking``
=================
Tractography and streamline operations::

    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
    from dipy.tracking import utils
    
    # Perform tractography
    streamlines = LocalTracking(...)

``dipy.segment``
================
Bundle segmentation and clustering::

    from dipy.segment.bundles import RecoBundles
    from dipy.segment.clustering import QuickBundles

``dipy.viz``
============
Visualization tools::

    from dipy.viz import window, actor
    
    # Create interactive 3D visualization
    scene = window.Scene()
    stream_actor = actor.line(streamlines)
    scene.add(stream_actor)
    window.show(scene)

``dipy.align``
==============
Image registration::

    from dipy.align.imaffine import AffineRegistration
    from dipy.align.transforms import TranslationTransform3D

*******************************
5. Your First DIPY Analysis
*******************************

Example 1: Computing Fractional Anisotropy (FA)
================================================

This example shows the most basic DIPY workflow - loading data and computing FA::

    # Import necessary modules
    from dipy.io.image import load_nifti, save_nifti
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel
    import numpy as np
    
    # Step 1: Load diffusion data
    data, affine = load_nifti('dwi.nii.gz')
    bvals, bvecs = read_bvals_bvecs('dwi.bval', 'dwi.bvec')
    
    # Step 2: Create gradient table
    gtab = gradient_table(bvals, bvecs=bvecs)
    
    # Step 3: Create brain mask (simple thresholding)
    # In practice, use proper brain extraction tools
    mask = data[..., 0] > 200
    
    # Step 4: Fit tensor model
    print("Fitting Tensor Model...")
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask=mask)
    
    # Step 5: Compute FA
    fa = tenfit.fa
    
    # Step 6: Save results
    save_nifti('fa.nii.gz', fa, affine)
    print("FA map saved!")
    
    # Optional: Compute other metrics
    md = tenfit.md  # Mean diffusivity
    rd = tenfit.rd  # Radial diffusivity
    ad = tenfit.ad  # Axial diffusivity
    color_fa = tenfit.color_fa  # RGB color FA

**What's happening here?**

1. We load the 4D diffusion data and its spatial transformation (affine)
2. We load the acquisition parameters (b-values and b-vectors)
3. We create a gradient table object that DIPY uses
4. We create a simple brain mask to speed up processing
5. We fit the tensor model to estimate diffusion tensors
6. We extract FA values and save them

Example 2: Denoising Diffusion Data
====================================

Improving signal quality before analysis::

    from dipy.denoise.nlmeans import nlmeans
    from dipy.denoise.noise_estimate import estimate_sigma
    from dipy.io.image import load_nifti, save_nifti
    
    # Load data
    data, affine = load_nifti('dwi.nii.gz')
    
    # Estimate noise level
    # N = number of coils in acquisition (typically 4, 8, 12, or 32)
    sigma = estimate_sigma(data, N=4)
    print(f"Estimated noise sigma: {sigma}")
    
    # Denoise using Non-Local Means
    denoised = nlmeans(
        data,
        sigma=sigma,
        patch_radius=1,  # Size of patches
        block_radius=5,  # Search radius
        rician=True      # Assume Rician noise (typical for MRI)
    )
    
    # Save denoised data
    save_nifti('dwi_denoised.nii.gz', denoised, affine)

Example 3: Creating Color FA Maps
==================================

Visualize fiber directions with color-coded FA::

    from dipy.io.image import load_nifti, save_nifti
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel
    
    # Load data
    data, affine = load_nifti('dwi.nii.gz')
    bvals, bvecs = read_bvals_bvecs('dwi.bval', 'dwi.bvec')
    gtab = gradient_table(bvals, bvecs=bvecs)
    
    # Fit model
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data)
    
    # Get color FA (RGB image showing fiber directions)
    # Red = Left-Right, Green = Anterior-Posterior, Blue = Superior-Inferior
    cfa = tenfit.color_fa
    
    # Save as NIfTI (note: 4D array with RGB in last dimension)
    save_nifti('color_fa.nii.gz', cfa, affine)

**Color FA Interpretation**:
* Red: fibers running left-right
* Green: fibers running front-back
* Blue: fibers running top-bottom
* Mixed colors indicate complex fiber configurations

********************************
6. Common Workflows Step-by-Step
********************************

Workflow 1: Basic DTI Analysis Pipeline
========================================

Complete pipeline from raw data to metrics::

    import numpy as np
    from dipy.io.image import load_nifti, save_nifti
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel
    from dipy.segment.mask import median_otsu
    
    # 1. Load your data
    data, affine = load_nifti('dwi.nii.gz')
    bvals, bvecs = read_bvals_bvecs('dwi.bval', 'dwi.bvec')
    
    # 2. Create gradient table
    gtab = gradient_table(bvals, bvecs)
    
    # 3. Brain extraction (automatic)
    # median_otsu removes background and extracts brain
    b0_mask, mask = median_otsu(data, vol_idx=range(10), median_radius=3,
                                  numpass=1, autocrop=False, dilate=2)
    
    # 4. Fit DTI model
    print("Fitting DTI model...")
    dti_model = TensorModel(gtab)
    dti_fit = dti_model.fit(data, mask=mask)
    
    # 5. Compute all metrics
    fa = dti_fit.fa
    md = dti_fit.md
    rd = dti_fit.rd
    ad = dti_fit.ad
    color_fa = dti_fit.color_fa
    
    # 6. Clean up NaN values
    fa[np.isnan(fa)] = 0
    
    # 7. Save all outputs
    save_nifti('dti_fa.nii.gz', fa, affine)
    save_nifti('dti_md.nii.gz', md, affine)
    save_nifti('dti_rd.nii.gz', rd, affine)
    save_nifti('dti_ad.nii.gz', ad, affine)
    save_nifti('dti_color_fa.nii.gz', color_fa, affine)
    
    print("DTI analysis complete!")

Workflow 2: Fiber Tracking (Tractography)
==========================================

Create streamlines representing white matter pathways::

    from dipy.io.image import load_nifti
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                        auto_response_ssst)
    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
    from dipy.tracking import utils
    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.streamline import Streamlines
    from dipy.segment.mask import median_otsu
    
    # 1. Load data
    data, affine = load_nifti('dwi.nii.gz')
    bvals, bvecs = read_bvals_bvecs('dwi.bval', 'dwi.bvec')
    gtab = gradient_table(bvals, bvecs)
    
    # 2. Brain mask
    b0_mask, mask = median_otsu(data, vol_idx=range(10), median_radius=3,
                                  numpass=1)
    
    # 3. Fit CSD model for fiber orientation
    print("Estimating response function...")
    response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
    
    print("Fitting CSD model...")
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
    csd_fit = csd_model.fit(data, mask=mask)
    
    # 4. Generate seeds (starting points for tracking)
    # Use white matter mask (FA > 0.2)
    from dipy.reconst.dti import TensorModel
    tensor_model = TensorModel(gtab)
    tensor_fit = tensor_model.fit(data, mask=mask)
    fa = tensor_fit.fa
    
    seed_mask = fa > 0.2
    seeds = utils.seeds_from_mask(seed_mask, affine, density=1)
    
    # 5. Create stopping criterion
    stopping_criterion = ThresholdStoppingCriterion(fa, 0.2)
    
    # 6. Perform tracking
    print(f"Starting tractography from {len(seeds)} seeds...")
    streamlines_generator = LocalTracking(
        csd_fit,
        stopping_criterion,
        seeds,
        affine=affine,
        step_size=0.5
    )
    
    # 7. Convert to Streamlines object
    streamlines = Streamlines(streamlines_generator)
    
    print(f"Generated {len(streamlines)} streamlines")
    
    # 8. Save streamlines
    from nibabel.streamlines import save as save_trk
    from nibabel.streamlines import Tractogram
    
    tractogram = Tractogram(streamlines, affine_to_rasmm=affine)
    save_trk('tractogram.trk', tractogram, shape=data.shape[:3])

Workflow 3: Working with Example Data
======================================

DIPY provides example datasets for learning::

    import numpy as np
    from dipy.data import get_fnames
    from dipy.io.image import load_nifti
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel
    
    # Get Stanford HARDI dataset
    fraw, fbval, fbvec = get_fnames('stanford_hardi')
    
    # Load data
    data, affine = load_nifti(fraw)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)
    
    # Now analyze as usual
    print(f"Data shape: {data.shape}")
    print(f"Number of gradients: {gtab.bvals.shape[0]}")
    print(f"B-values: {np.unique(gtab.bvals)}")
    
    # Fit DTI
    mask = data[..., 0] > 50
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask=mask)
    
    print("FA range:", tenfit.fa.min(), "-", tenfit.fa.max())

***********************
7. Visualization Tips
***********************

Using Matplotlib for 2D Slices
===============================

Quick visualization of metrics::

    import matplotlib.pyplot as plt
    import numpy as np
    from dipy.io.image import load_nifti
    
    # Load FA map
    fa, affine = load_nifti('dti_fa.nii.gz')
    
    # Display middle slices
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Axial slice (looking from top)
    slice_z = fa.shape[2] // 2
    axes[0].imshow(fa[:, :, slice_z].T, cmap='gray', origin='lower',
                   vmin=0, vmax=1)
    axes[0].set_title('Axial')
    axes[0].axis('off')
    
    # Coronal slice (looking from front)
    slice_y = fa.shape[1] // 2
    axes[1].imshow(fa[:, slice_y, :].T, cmap='gray', origin='lower',
                   vmin=0, vmax=1)
    axes[1].set_title('Coronal')
    axes[1].axis('off')
    
    # Sagittal slice (looking from side)
    slice_x = fa.shape[0] // 2
    axes[2].imshow(fa[slice_x, :, :].T, cmap='gray', origin='lower',
                   vmin=0, vmax=1)
    axes[2].set_title('Sagittal')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('fa_slices.png', dpi=150, bbox_inches='tight')
    plt.show()

Interactive 3D Visualization with DIPY
=======================================

Visualize tractography results::

    from dipy.viz import window, actor, colormap
    from dipy.io.streamline import load_tractogram
    
    # Load streamlines
    tractogram = load_tractogram('tractogram.trk', 'same')
    streamlines = tractogram.streamlines
    
    # Create scene
    scene = window.Scene()
    
    # Color streamlines by direction
    stream_actor = actor.line(streamlines, colormap.line_colors(streamlines))
    scene.add(stream_actor)
    
    # Show interactive window
    window.show(scene, size=(800, 800), reset_camera=True)
    
    # Or save a screenshot
    window.record(scene, out_path='tractography.png', size=(800, 800))

Simple Slice Viewer
===================

Create a simple volume viewer::

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    import numpy as np
    from dipy.io.image import load_nifti
    
    data, affine = load_nifti('dti_fa.nii.gz')
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    
    # Initial slice
    slice_idx = data.shape[2] // 2
    img = ax.imshow(data[:, :, slice_idx].T, cmap='gray',
                    origin='lower', vmin=0, vmax=1)
    ax.set_title(f'Slice {slice_idx}')
    
    # Slider for slice selection
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, data.shape[2]-1,
                    valinit=slice_idx, valstep=1)
    
    def update(val):
        idx = int(slider.val)
        img.set_data(data[:, :, idx].T)
        ax.set_title(f'Slice {idx}')
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.show()

***************************
8. Advanced Reconstruction
***************************

Beyond DTI: Multi-shell Models
===============================

For multi-shell data (multiple b-values), use advanced models::

    import numpy as np
    from dipy.reconst.dki import DiffusionKurtosisModel
    from dipy.io.image import load_nifti, save_nifti
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.segment.mask import median_otsu
    
    # Load multi-shell data
    data, affine = load_nifti('dwi.nii.gz')
    bvals, bvecs = read_bvals_bvecs('dwi.bval', 'dwi.bvec')
    gtab = gradient_table(bvals, bvecs)
    
    # Create brain mask
    b0_mask, mask = median_otsu(data, vol_idx=range(10), median_radius=3)
    
    # Check for multi-shell
    print("B-values:", np.unique(gtab.bvals))
    
    # Fit DKI (requires at least 2 shells)
    dki_model = DiffusionKurtosisModel(gtab)
    dki_fit = dki_model.fit(data, mask=mask)
    
    # Get kurtosis metrics
    mk = dki_fit.mk(0, 3)  # Mean kurtosis
    ak = dki_fit.ak(0, 3)  # Axial kurtosis
    rk = dki_fit.rk(0, 3)  # Radial kurtosis
    
    # Also get DTI metrics from DKI
    fa = dki_fit.fa
    md = dki_fit.md

Constrained Spherical Deconvolution (CSD)
==========================================

Better for crossing fibers::

    from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                        auto_response_ssst)
    from dipy.data import default_sphere
    from dipy.direction import peaks_from_model
    from dipy.io.image import load_nifti
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.segment.mask import median_otsu
    
    # Load data
    data, affine = load_nifti('dwi.nii.gz')
    bvals, bvecs = read_bvals_bvecs('dwi.bval', 'dwi.bvec')
    gtab = gradient_table(bvals, bvecs)
    
    # Create brain mask
    b0_mask, mask = median_otsu(data, vol_idx=range(10), median_radius=3)
    
    # Estimate response function
    response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
    
    # Fit CSD
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=8)
    csd_fit = csd_model.fit(data, mask=mask)
    
    # Get fiber orientation distribution function (fODF)
    csd_odf = csd_fit.odf(default_sphere)
    
    # Peak detection
    csd_peaks = peaks_from_model(
        csd_model,
        data,
        default_sphere,
        relative_peak_threshold=0.5,
        min_separation_angle=25,
        mask=mask
    )

*****************************
9. Tips and Best Practices
*****************************

Data Quality Checks
===================

Always check your data before analysis::

    import numpy as np
    from dipy.io.image import load_nifti
    from dipy.io import read_bvals_bvecs
    
    # Load data
    data, affine = load_nifti('dwi.nii.gz')
    bvals, bvecs = read_bvals_bvecs('dwi.bval', 'dwi.bvec')
    
    # Check dimensions
    print(f"Data shape: {data.shape}")
    print(f"Expected: (x, y, z, n_gradients)")
    
    # Check b-values
    print(f"B-values: {np.unique(bvals)}")
    print(f"Number of b0 images: {np.sum(bvals < 50)}")
    
    # Check for zero or negative values
    print(f"Min value: {data.min()}, Max value: {data.max()}")
    
    # Check b-vectors are normalized
    print(f"B-vector norms: {np.linalg.norm(bvecs, axis=1)[:10]}")

Memory Management
=================

For large datasets::

    # Process data in chunks
    def process_in_chunks(data, mask, model, chunk_size=1000):
        """Process voxels in chunks to save memory"""
        n_voxels = np.sum(mask)
        indices = np.array(np.where(mask)).T
        
        results = np.zeros(data.shape[:3])
        
        for i in range(0, n_voxels, chunk_size):
            chunk_indices = indices[i:i+chunk_size]
            chunk_data = data[chunk_indices[:, 0],
                             chunk_indices[:, 1],
                             chunk_indices[:, 2]]
            
            chunk_fit = model.fit(chunk_data)
            chunk_results = chunk_fit.fa
            
            results[chunk_indices[:, 0],
                   chunk_indices[:, 1],
                   chunk_indices[:, 2]] = chunk_results
        
        return results

Common Pitfalls to Avoid
========================

1. **Not using a brain mask**: Always use a mask to speed up processing and avoid background voxels

2. **Ignoring NaN values**: DTI fitting can produce NaN values - clean them::

    fa[np.isnan(fa)] = 0

3. **Wrong affine matrix**: Always use the same affine when saving results::

    save_nifti('output.nii.gz', result, affine)  # Use original affine!

4. **Not normalizing b-vectors**: Ensure b-vectors have unit length::

    from dipy.core.gradients import gradient_table
    gtab = gradient_table(bvals, bvecs=bvecs)  # Automatically normalizes

5. **Incompatible models**: Use DTI for single-shell, DKI/CSD for multi-shell data

Optimizing Performance
======================

Speed up your analysis::

    # Use parallel processing
    from dipy.reconst.dti import TensorModel
    
    tenmodel = TensorModel(gtab)
    # Fit will automatically use parallel processing if possible
    tenfit = tenmodel.fit(data, mask=mask)
    
    # For very large datasets, consider:
    # - Processing slices independently
    # - Using smaller masks
    # - Reducing spatial resolution
    # - Using faster models (DTI vs CSD)

*********************
10. Troubleshooting
*********************

Common Errors and Solutions
============================

**Error: "shapes not aligned"**
    - **Problem**: Data dimensions don't match
    - **Solution**: Check that data is 4D and gradient table matches number of volumes::

        print(f"Data shape: {data.shape}")
        print(f"Gradient table size: {len(gtab.bvals)}")

**Error: "NaN values in output"**
    - **Problem**: Fitting failed in some voxels
    - **Solution**: Use a better mask or replace NaN with zeros::

        result[np.isnan(result)] = 0

**Error: "Cannot import dipy"**
    - **Problem**: DIPY not installed properly
    - **Solution**: Reinstall::

        pip install --upgrade dipy

**Error: Memory Error**
    - **Problem**: Dataset too large
    - **Solution**: Process in chunks or use smaller mask

**Streamlines not visible**
    - **Problem**: Visualization issue
    - **Solution**: Check if streamlines were generated::

        print(f"Number of streamlines: {len(streamlines)}")

Getting Help
============

When you encounter issues:

1. **Check the error message** carefully
2. **Search DIPY documentation**: https://docs.dipy.org
3. **Look at examples**: https://docs.dipy.org/stable/examples_built/
4. **Ask on GitHub Discussions**: https://github.com/dipy/dipy/discussions
5. **Check GitHub Issues**: https://github.com/dipy/dipy/issues

When asking for help, provide:
    - DIPY version (``import dipy; print(dipy.__version__)``)
    - Python version
    - Operating system
    - Full error message
    - Minimal code to reproduce the issue

**********************************
11. Resources for Further Learning
**********************************

Official Documentation
======================

* **DIPY Website**: https://dipy.org
* **Full Documentation**: https://docs.dipy.org
* **Tutorials**: https://docs.dipy.org/stable/examples_built/
* **API Reference**: https://docs.dipy.org/stable/reference/

Scientific Background
=====================

Key papers to understand diffusion MRI:

* **DTI**: Basser et al. (1994) - Original DTI paper
* **CSD**: Tournier et al. (2007) - Constrained spherical deconvolution
* **DIPY**: Garyfallidis et al. (2014) - DIPY library paper

Recommended Reading:
    - "Introduction to Diffusion Tensor Imaging" by Mori
    - "Diffusion MRI: Theory, Methods, and Applications" by Jones

Community Resources
===================

* **GitHub Discussions**: https://github.com/dipy/dipy/discussions
* **Mailing List**: https://mail.python.org/mailman3/lists/dipy.python.org/
* **Gitter Chat**: https://gitter.im/dipy/dipy
* **Twitter**: @dipymri

Online Courses and Tutorials
=============================

* **DIPY Examples Gallery**: Step-by-step tutorials on specific topics
* **Neuroimaging in Python**: https://nipy.org/
* **Andy's Brain Book**: Neuroimaging tutorials including diffusion MRI

Example Datasets
================

DIPY provides several example datasets::

    from dipy.data import get_fnames
    
    # Stanford HARDI
    fraw, fbval, fbvec = get_fnames('stanford_hardi')
    
    # Stanford T1
    t1_fname = get_fnames('stanford_t1')
    
    # Sherbrooke multi-shell
    fraw, fbval, fbvec = get_fnames('sherbrooke_3shell')

Practice Projects
=================

Try these projects to build your skills:

1. **Beginner**: Compute FA maps for different brain regions
2. **Intermediate**: Perform whole-brain tractography
3. **Advanced**: Implement bundle segmentation and connectivity analysis

********************
12. Quick Reference
********************

Essential Imports
=================

Copy this as a starting template::

    # I/O operations
    from dipy.io.image import load_nifti, save_nifti
    from dipy.io import read_bvals_bvecs
    
    # Core
    from dipy.core.gradients import gradient_table
    
    # Reconstruction
    from dipy.reconst.dti import TensorModel
    from dipy.reconst.dki import DiffusionKurtosisModel
    from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                        auto_response_ssst)
    
    # Denoising
    from dipy.denoise.nlmeans import nlmeans
    from dipy.denoise.noise_estimate import estimate_sigma
    
    # Tracking
    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
    from dipy.tracking import utils
    
    # Visualization
    from dipy.viz import window, actor
    import matplotlib.pyplot as plt
    
    # Utilities
    import numpy as np
    from dipy.segment.mask import median_otsu

Common Commands Cheat Sheet
============================

**Load data**::

    data, affine = load_nifti('file.nii.gz')
    bvals, bvecs = read_bvals_bvecs('file.bval', 'file.bvec')
    gtab = gradient_table(bvals, bvecs=bvecs)

**Brain extraction**::

    from dipy.segment.mask import median_otsu
    b0_mask, mask = median_otsu(data, vol_idx=range(10), median_radius=3)

**Fit DTI**::

    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask=mask)
    fa = tenfit.fa

**Denoise**::

    sigma = estimate_sigma(data, N=4)
    denoised = nlmeans(data, sigma=sigma)

**Save results**::

    save_nifti('output.nii.gz', result, affine)

**Plot slice**::

    import matplotlib.pyplot as plt
    plt.imshow(fa[:, :, 50].T, cmap='gray', origin='lower')
    plt.colorbar()
    plt.show()

Typical File Formats
====================

* **.nii.gz**: Compressed NIfTI image (recommended)
* **.bval**: Text file with b-values (one line or space-separated)
* **.bvec**: Text file with b-vectors (3 rows x N columns or N rows x 3 columns)
* **.trk**: Tractography streamlines (TrackVis format)

**********
Conclusion
**********

Congratulations! You now have a solid foundation for working with DIPY. Remember:

1. **Start simple**: Begin with example data and basic DTI analysis
2. **Read documentation**: DIPY has extensive examples and tutorials
3. **Ask questions**: The community is helpful and responsive
4. **Practice**: The more you use DIPY, the more comfortable you'll become
5. **Share**: Contribute back to the community with your findings

Happy analyzing, and welcome to the DIPY community!

***********
Appendix
***********

Installing DIPY from Source (Advanced)
=======================================

For developers or those wanting the latest features::

    git clone https://github.com/dipy/dipy.git
    cd dipy
    pip install -e .

Additional Useful Packages
===========================

Consider installing::

    pip install vtk  # For advanced 3D visualization
    pip install fury  # DIPY's visualization backend
    pip install matplotlib  # 2D plotting
    pip install nibabel  # Neuroimaging file I/O

DIPY Version Information
========================

Check your DIPY installation::

    import dipy
    print(f"DIPY version: {dipy.__version__}")
    
    # Check available modules
    print(dir(dipy))

Workshop Checklist
==================

Before the workshop, ensure you have:

- [ ] Python 3.10+ installed
- [ ] DIPY installed (``pip install dipy``)
- [ ] Jupyter notebook installed (``pip install jupyter``)
- [ ] matplotlib installed (``pip install matplotlib``)
- [ ] Downloaded example data (done automatically by DIPY)
- [ ] Tested installation (``import dipy; print(dipy.__version__)``)
- [ ] Basic Python knowledge (variables, functions, imports)
- [ ] A text editor or IDE ready

You're all set for the workshop!

.. include:: ../links_names.inc
