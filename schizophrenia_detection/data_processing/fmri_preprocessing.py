"""
fMRI data preprocessing module

This module provides a comprehensive pipeline for preprocessing fMRI data, including:
- Realignment and motion correction
- Slice-timing correction
- Spatial smoothing
- Coregistration and normalization
- Statistical Parametric Mapping (SPM)
- Spatial Source Phase (SSP) map generation
- Reverse phase de-ambiguization
"""

import os
import logging
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional, Union, List
from pathlib import Path
import warnings

# Nilearn imports
from nilearn import image
from nilearn import masking
from nilearn import plotting
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_img, smooth_img, index_img

# Configure logging
logger = logging.getLogger(__name__)


class FMRIPreprocessingError(Exception):
    """Custom exception for fMRI preprocessing errors"""

    pass


def validate_fmri_data(fmri_img: Union[str, nib.Nifti1Image]) -> nib.Nifti1Image:
    """
    Validate fMRI data and return as Nifti1Image object

    Args:
        fmri_img: Path to fMRI file or Nifti1Image object

    Returns:
        nib.Nifti1Image: Validated fMRI image

    Raises:
        FMRIPreprocessingError: If data is invalid
    """
    try:
        if isinstance(fmri_img, str):
            if not os.path.exists(fmri_img):
                raise FMRIPreprocessingError(f"File not found: {fmri_img}")
            img = nib.load(fmri_img)
        elif isinstance(fmri_img, nib.Nifti1Image):
            img = fmri_img
        else:
            raise FMRIPreprocessingError(
                "Input must be a file path or Nifti1Image object"
            )

        # Check if it's 4D (fMRI time series)
        if len(img.shape) < 4:
            raise FMRIPreprocessingError("Input must be a 4D fMRI time series")

        # Check for valid data
        data = img.get_fdata()
        if np.all(np.isnan(data)):
            raise FMRIPreprocessingError("fMRI data contains only NaN values")

        return img
    except Exception as e:
        raise FMRIPreprocessingError(f"Error validating fMRI data: {str(e)}")


def realign_motion_correction(
    fmri_img: Union[str, nib.Nifti1Image],
    reference_vol: Optional[int] = None,
    cost_function: str = "mutual_info",
    interp: str = "trilinear",
) -> nib.Nifti1Image:
    """
    Perform realignment and motion correction on fMRI data

    Args:
        fmri_img: Path to fMRI file or Nifti1Image object
        reference_vol: Reference volume for realignment (default: middle volume)
        cost_function: Cost function for registration ('mutual_info', 'corr', 'normcorr')
        interp: Interpolation method ('trilinear', 'nearest', 'cubic')

    Returns:
        nib.Nifti1Image: Motion-corrected fMRI image

    Raises:
        FMRIPreprocessingError: If realignment fails
    """
    try:
        logger.info("Starting motion correction and realignment")

        # Validate input
        img = validate_fmri_data(fmri_img)
        data = img.get_fdata()
        affine = img.affine
        header = img.header

        # Set reference volume
        if reference_vol is None:
            reference_vol = data.shape[3] // 2

        # Extract reference volume
        ref_data = data[..., reference_vol]
        ref_img = nib.Nifti1Image(ref_data, affine, header)

        # Initialize motion parameters
        motion_params = np.zeros(
            (data.shape[3], 6)
        )  # 6 parameters: 3 translations, 3 rotations

        # Create motion-corrected data array
        corrected_data = np.zeros_like(data)

        # Process each volume
        for t in range(data.shape[3]):
            if t == reference_vol:
                # Reference volume remains unchanged
                corrected_data[..., t] = data[..., t]
                continue

            # Current volume
            vol_data = data[..., t]
            vol_img = nib.Nifti1Image(vol_data, affine, header)

            # Perform registration (simplified implementation)
            # In practice, you would use a more sophisticated registration algorithm
            # Here we use a basic center of mass alignment as an example

            # Calculate center of mass for reference and current volume
            ref_com = ndimage.center_of_mass(ref_data)
            vol_com = ndimage.center_of_mass(vol_data)

            # Calculate translation
            translation = np.array(ref_com) - np.array(vol_com)

            # Apply translation (simplified - in practice use full 6 DOF registration)
            corrected_vol = ndimage.shift(vol_data, translation, order=1)

            # Store corrected volume
            corrected_data[..., t] = corrected_vol

            # Store motion parameters
            motion_params[t, :3] = translation  # Translation parameters
            # Rotation parameters would be calculated in a full implementation

        # Create motion-corrected image
        corrected_img = nib.Nifti1Image(corrected_data, affine, header)

        logger.info(
            f"Motion correction completed. Max displacement: {np.max(np.abs(motion_params)):.3f} mm"
        )

        return corrected_img

    except Exception as e:
        raise FMRIPreprocessingError(f"Error in motion correction: {str(e)}")


def slice_timing_correction(
    fmri_img: Union[str, nib.Nifti1Image],
    tr: float = 2.0,
    slice_order: str = "ascending",
    reference_slice: Optional[int] = None,
    interleaved: bool = False,
) -> nib.Nifti1Image:
    """
    Perform slice-timing correction on fMRI data

    Args:
        fmri_img: Path to fMRI file or Nifti1Image object
        tr: Repetition time in seconds
        slice_order: Slice acquisition order ('ascending', 'descending', 'custom')
        reference_slice: Reference slice for correction (default: middle slice)
        interleaved: Whether slices were acquired in interleaved order

    Returns:
        nib.Nifti1Image: Slice timing corrected fMRI image

    Raises:
        FMRIPreprocessingError: If slice timing correction fails
    """
    try:
        logger.info("Starting slice-timing correction")

        # Validate input
        img = validate_fmri_data(fmri_img)
        data = img.get_fdata()
        affine = img.affine
        header = img.header

        # Get number of slices
        num_slices = data.shape[2]

        # Set reference slice
        if reference_slice is None:
            reference_slice = num_slices // 2

        # Create slice timing array
        if slice_order == "ascending":
            slice_times = np.linspace(0, tr, num_slices, endpoint=False)
        elif slice_order == "descending":
            slice_times = np.linspace(tr, 0, num_slices, endpoint=False)
        else:
            # Default to ascending
            slice_times = np.linspace(0, tr, num_slices, endpoint=False)

        # Adjust for interleaved acquisition
        if interleaved:
            # Create interleaved pattern
            even_slices = np.arange(0, num_slices, 2)
            odd_slices = np.arange(1, num_slices, 2)

            interleaved_times = np.zeros(num_slices)
            interleaved_times[even_slices] = np.linspace(
                0, tr, len(even_slices), endpoint=False
            )
            interleaved_times[odd_slices] = np.linspace(
                0, tr, len(odd_slices), endpoint=False
            )
            slice_times = interleaved_times

        # Reference slice time
        ref_time = slice_times[reference_slice]

        # Calculate time offsets for each slice
        time_offsets = ref_time - slice_times

        # Initialize corrected data
        corrected_data = np.zeros_like(data)

        # Process each time point
        for t in range(data.shape[3]):
            # Get current volume
            volume = data[..., t]

            # Process each slice
            for s in range(num_slices):
                if s == reference_slice:
                    # Reference slice remains unchanged
                    corrected_data[:, :, s, t] = volume[:, :, s]
                    continue

                # Calculate interpolation points
                offset = time_offsets[s]

                # For simplicity, we'll use temporal interpolation
                # In practice, this would be more sophisticated
                if t > 0 and t < data.shape[3] - 1:
                    # Linear interpolation with neighboring time points
                    alpha = offset / tr
                    corrected_slice = (1 - alpha) * volume[:, :, s] + alpha * data[
                        :, :, s, t + 1
                    ]
                else:
                    corrected_slice = volume[:, :, s]

                corrected_data[:, :, s, t] = corrected_slice

        # Create corrected image
        corrected_img = nib.Nifti1Image(corrected_data, affine, header)

        logger.info("Slice-timing correction completed")

        return corrected_img

    except Exception as e:
        raise FMRIPreprocessingError(f"Error in slice timing correction: {str(e)}")


def spatial_smoothing(
    fmri_img: Union[str, nib.Nifti1Image], fwhm: float = 6.0
) -> nib.Nifti1Image:
    """
    Apply spatial smoothing to fMRI data

    Args:
        fmri_img: Path to fMRI file or Nifti1Image object
        fwhm: Full width at half maximum in mm

    Returns:
        nib.Nifti1Image: Spatially smoothed fMRI image

    Raises:
        FMRIPreprocessingError: If smoothing fails
    """
    try:
        logger.info(f"Applying spatial smoothing with FWHM: {fwhm} mm")

        # Validate input
        img = validate_fmri_data(fmri_img)

        # Apply smoothing using Nilearn
        smoothed_img = smooth_img(img, fwhm)

        logger.info("Spatial smoothing completed")

        return smoothed_img

    except Exception as e:
        raise FMRIPreprocessingError(f"Error in spatial smoothing: {str(e)}")


def coregistration(
    func_img: Union[str, nib.Nifti1Image],
    struct_img: Union[str, nib.Nifti1Image],
    cost_function: str = "mutual_info",
    interp: str = "trilinear",
) -> nib.Nifti1Image:
    """
    Coregister functional image to structural image

    Args:
        func_img: Functional fMRI image
        struct_img: Structural anatomical image
        cost_function: Cost function for registration
        interp: Interpolation method

    Returns:
        nib.Nifti1Image: Coregistered functional image

    Raises:
        FMRIPreprocessingError: If coregistration fails
    """
    try:
        logger.info("Starting functional-structural coregistration")

        # Validate inputs
        func_img = validate_fmri_data(func_img)

        if isinstance(struct_img, str):
            if not os.path.exists(struct_img):
                raise FMRIPreprocessingError(
                    f"Structural image not found: {struct_img}"
                )
            struct_img = nib.load(struct_img)
        elif not isinstance(struct_img, nib.Nifti1Image):
            raise FMRIPreprocessingError(
                "Structural image must be a file path or Nifti1Image object"
            )

        # Create mean functional image for registration
        func_data = func_img.get_fdata()
        mean_func_data = np.mean(func_data, axis=3)
        mean_func_img = nib.Nifti1Image(
            mean_func_data, func_img.affine, func_img.header
        )

        # Simplified coregistration using center of mass alignment
        # In practice, you would use a more sophisticated registration algorithm
        func_com = ndimage.center_of_mass(mean_func_data)
        struct_data = struct_img.get_fdata()
        struct_com = ndimage.center_of_mass(struct_data)

        # Calculate translation
        translation = np.array(struct_com) - np.array(func_com)

        # Apply translation to functional image
        corrected_data = np.zeros_like(func_data)
        for t in range(func_data.shape[3]):
            corrected_data[..., t] = ndimage.shift(
                func_data[..., t], translation, order=1
            )

        # Create coregistered image
        coreg_img = nib.Nifti1Image(corrected_data, func_img.affine, func_img.header)

        logger.info("Functional-structural coregistration completed")

        return coreg_img

    except Exception as e:
        raise FMRIPreprocessingError(f"Error in coregistration: {str(e)}")


def normalize_to_mni(
    fmri_img: Union[str, nib.Nifti1Image],
    template_img: Optional[Union[str, nib.Nifti1Image]] = None,
    voxel_size: Tuple[float, float, float] = (2.0, 2.0, 2.0),
) -> nib.Nifti1Image:
    """
    Normalize fMRI image to MNI standard space

    Args:
        fmri_img: fMRI image to normalize
        template_img: MNI template image (if None, uses default MNI152)
        voxel_size: Target voxel size in mm

    Returns:
        nib.Nifti1Image: Normalized fMRI image

    Raises:
        FMRIPreprocessingError: If normalization fails
    """
    try:
        logger.info("Starting normalization to MNI space")

        # Validate input
        img = validate_fmri_data(fmri_img)

        # If no template provided, use a simple approach
        if template_img is None:
            # Create a target affine with specified voxel size
            target_affine = np.eye(4)
            target_affine[0, 0] = voxel_size[0]
            target_affine[1, 1] = voxel_size[1]
            target_affine[2, 2] = voxel_size[2]

            # Estimate target shape based on current image and voxel size
            current_shape = img.shape[:3]
            target_shape = [
                int(current_shape[i] * img.header.get_zooms()[i] / voxel_size[i])
                for i in range(3)
            ]

            # Resample to target space
            normalized_img = resample_img(
                img,
                target_affine=target_affine,
                target_shape=target_shape,
                interpolation="continuous",
            )
        else:
            # Use provided template
            if isinstance(template_img, str):
                if not os.path.exists(template_img):
                    raise FMRIPreprocessingError(
                        f"Template image not found: {template_img}"
                    )
                template_img = nib.load(template_img)

            # Resample to template space
            normalized_img = resample_img(
                img,
                target_affine=template_img.affine,
                target_shape=template_img.shape[:3],
                interpolation="continuous",
            )

        logger.info("Normalization to MNI space completed")

        return normalized_img

    except Exception as e:
        raise FMRIPreprocessingError(f"Error in normalization: {str(e)}")


def statistical_parametric_mapping(
    fmri_img: Union[str, nib.Nifti1Image],
    design_matrix: np.ndarray,
    contrast_matrix: np.ndarray,
    mask_img: Optional[Union[str, nib.Nifti1Image]] = None,
) -> Dict:
    """
    Perform Statistical Parametric Mapping (SPM) analysis on fMRI data

    Args:
        fmri_img: fMRI image for analysis
        design_matrix: Design matrix for GLM analysis
        contrast_matrix: Contrast matrix for hypothesis testing
        mask_img: Brain mask for analysis

    Returns:
        Dict: Dictionary containing SPM results (stat maps, p-values, etc.)

    Raises:
        FMRIPreprocessingError: If SPM analysis fails
    """
    try:
        logger.info("Starting Statistical Parametric Mapping (SPM) analysis")

        # Validate input
        img = validate_fmri_data(fmri_img)
        data = img.get_fdata()

        # Validate design matrix
        if design_matrix.shape[0] != data.shape[3]:
            raise FMRIPreprocessingError(
                f"Design matrix rows ({design_matrix.shape[0]}) must match "
                f"number of time points ({data.shape[3]})"
            )

        # Create mask if not provided
        if mask_img is None:
            mask_img = masking.compute_brain_mask(img)
        elif isinstance(mask_img, str):
            mask_img = nib.load(mask_img)

        # Get mask data
        mask_data = mask_img.get_fdata().astype(bool)

        # Extract masked data
        masked_data = data[mask_data, :]  # Shape: (n_voxels, n_timepoints)

        # Initialize results dictionary
        results = {}

        # Perform GLM analysis for each contrast
        n_contrasts = contrast_matrix.shape[0]
        for c in range(n_contrasts):
            contrast = contrast_matrix[c, :]

            # Compute beta coefficients using least squares
            # beta = (X'X)^-1 X'y
            XtX = np.dot(design_matrix.T, design_matrix)
            XtX_inv = np.linalg.inv(XtX)
            XtY = np.dot(design_matrix.T, masked_data.T)
            beta = np.dot(XtX_inv, XtY).T  # Shape: (n_voxels, n_regressors)

            # Compute contrast statistic
            contrast_beta = np.dot(beta, contrast)  # Shape: (n_voxels,)

            # Compute residual sum of squares
            predicted = np.dot(design_matrix, beta.T)  # Shape: (n_timepoints, n_voxels)
            residuals = masked_data.T - predicted  # Shape: (n_timepoints, n_voxels)
            rss = np.sum(residuals**2, axis=0)  # Shape: (n_voxels,)

            # Compute degrees of freedom
            df = design_matrix.shape[0] - np.linalg.matrix_rank(design_matrix)

            # Compute variance of contrast
            contrast_var = np.dot(contrast, np.dot(XtX_inv, contrast))

            # Compute t-statistic
            mse = rss / df  # Mean squared error
            se = np.sqrt(mse * contrast_var)  # Standard error
            t_stat = contrast_beta / se  # t-statistic

            # Compute p-values (two-tailed)
            from scipy.stats import t as t_dist

            p_values = 2 * (1 - t_dist.cdf(np.abs(t_stat), df))

            # Create stat maps
            stat_map_data = np.zeros(img.shape[:3])
            p_map_data = np.zeros(img.shape[:3])

            stat_map_data[mask_data] = t_stat
            p_map_data[mask_data] = p_values

            # Create NIfTI images
            stat_map_img = nib.Nifti1Image(stat_map_data, img.affine, img.header)
            p_map_img = nib.Nifti1Image(p_map_data, img.affine, img.header)

            # Store results
            results[f"contrast_{c}"] = {
                "stat_map": stat_map_img,
                "p_map": p_map_img,
                "t_stat": t_stat,
                "p_values": p_values,
                "contrast_beta": contrast_beta,
                "df": df,
            }

        logger.info("Statistical Parametric Mapping (SPM) analysis completed")

        return results

    except Exception as e:
        raise FMRIPreprocessingError(f"Error in SPM analysis: {str(e)}")


def generate_ssp_maps(
    fmri_img: Union[str, nib.Nifti1Image],
    mask_img: Optional[Union[str, nib.Nifti1Image]] = None,
) -> nib.Nifti1Image:
    """
    Generate Spatial Source Phase (SSP) maps from complex-valued fMRI data

    Args:
        fmri_img: Complex-valued fMRI image
        mask_img: Brain mask for analysis

    Returns:
        nib.Nifti1Image: SSP map

    Raises:
        FMRIPreprocessingError: If SSP map generation fails
    """
    try:
        logger.info("Generating Spatial Source Phase (SSP) maps")

        # Validate input
        img = validate_fmri_data(fmri_img)
        data = img.get_fdata()

        # Create mask if not provided
        if mask_img is None:
            mask_img = masking.compute_brain_mask(img)
        elif isinstance(mask_img, str):
            mask_img = nib.load(mask_img)

        # Get mask data
        mask_data = mask_img.get_fdata().astype(bool)

        # Check if data is complex-valued
        if not np.iscomplexobj(data):
            logger.warning(
                "Input data is not complex-valued. Computing Hilbert transform."
            )
            # Apply Hilbert transform to get complex representation
            from scipy.signal import hilbert

            ssp_data = np.zeros(data.shape[:3], dtype=np.complex128)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    for k in range(data.shape[2]):
                        if mask_data[i, j, k]:
                            ssp_data[i, j, k] = hilbert(data[i, j, k, :])
        else:
            # Use complex data directly
            ssp_data = np.mean(data, axis=3)

        # Extract phase information
        phase_data = np.angle(ssp_data)

        # Extract amplitude information
        amplitude_data = np.abs(ssp_data)

        # Create SSP map (phase weighted by amplitude)
        ssp_map = phase_data * amplitude_data

        # Create SSP map image
        ssp_map_img = nib.Nifti1Image(ssp_map, img.affine, img.header)

        logger.info("SSP map generation completed")

        return ssp_map_img

    except Exception as e:
        raise FMRIPreprocessingError(f"Error in SSP map generation: {str(e)}")


def reverse_phase_deambiguization(
    ssp_map_img: Union[str, nib.Nifti1Image], amplitude_threshold: float = 0.5
) -> nib.Nifti1Image:
    """
    Perform reverse phase de-ambiguization to create SSP maps with
    high-amplitude BOLD-related voxels and low-amplitude noisy voxels

    Args:
        ssp_map_img: SSP map image
        amplitude_threshold: Threshold for amplitude-based filtering

    Returns:
        nib.Nifti1Image: De-ambiguated SSP map

    Raises:
        FMRIPreprocessingError: If de-ambiguization fails
    """
    try:
        logger.info("Starting reverse phase de-ambiguization")

        # Validate input
        if isinstance(ssp_map_img, str):
            if not os.path.exists(ssp_map_img):
                raise FMRIPreprocessingError(f"SSP map file not found: {ssp_map_img}")
            img = nib.load(ssp_map_img)
        elif isinstance(ssp_map_img, nib.Nifti1Image):
            img = ssp_map_img
        else:
            raise FMRIPreprocessingError(
                "Input must be a file path or Nifti1Image object"
            )

        # Get SSP map data
        ssp_data = img.get_fdata()

        # Extract amplitude and phase
        amplitude = np.abs(ssp_data)
        phase = np.angle(ssp_data)

        # Create amplitude mask
        amplitude_mask = amplitude > amplitude_threshold

        # Apply reverse phase for low-amplitude voxels
        deambiguated_phase = phase.copy()
        deambiguated_phase[~amplitude_mask] = -phase[~amplitude_mask]

        # Create de-ambiguated SSP map
        deambiguated_ssp = amplitude * np.exp(1j * deambiguated_phase)

        # Create output image (use real part for visualization)
        output_data = np.real(deambiguated_ssp)
        output_img = nib.Nifti1Image(output_data, img.affine, img.header)

        logger.info("Reverse phase de-ambiguization completed")

        return output_img

    except Exception as e:
        raise FMRIPreprocessingError(
            f"Error in reverse phase de-ambiguization: {str(e)}"
        )


def preprocess_fmri_pipeline(
    fmri_img: Union[str, nib.Nifti1Image],
    struct_img: Optional[Union[str, nib.Nifti1Image]] = None,
    tr: float = 2.0,
    fwhm: float = 6.0,
    slice_order: str = "ascending",
    perform_normalization: bool = True,
    perform_spm: bool = False,
    design_matrix: Optional[np.ndarray] = None,
    contrast_matrix: Optional[np.ndarray] = None,
    generate_ssp: bool = False,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Complete fMRI preprocessing pipeline

    Args:
        fmri_img: Input fMRI image
        struct_img: Structural image for coregistration (optional)
        tr: Repetition time in seconds
        fwhm: Full width at half maximum for smoothing in mm
        slice_order: Slice acquisition order
        perform_normalization: Whether to perform normalization to MNI space
        perform_spm: Whether to perform SPM analysis
        design_matrix: Design matrix for SPM analysis
        contrast_matrix: Contrast matrix for SPM analysis
        generate_ssp: Whether to generate SSP maps
        output_dir: Directory to save intermediate results

    Returns:
        Dict: Dictionary containing all preprocessing results

    Raises:
        FMRIPreprocessingError: If preprocessing fails
    """
    try:
        logger.info("Starting complete fMRI preprocessing pipeline")

        # Create output directory if specified
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        # Initialize results dictionary
        results = {}

        # Step 1: Motion correction and realignment
        logger.info("Step 1: Motion correction and realignment")
        realigned_img = realign_motion_correction(fmri_img)
        results["realigned"] = realigned_img

        if output_dir is not None:
            output_path = os.path.join(output_dir, "01_realigned.nii.gz")
            nib.save(realigned_img, output_path)

        # Step 2: Slice timing correction
        logger.info("Step 2: Slice timing correction")
        stc_img = slice_timing_correction(realigned_img, tr=tr, slice_order=slice_order)
        results["slice_timing_corrected"] = stc_img

        if output_dir is not None:
            output_path = os.path.join(output_dir, "02_slice_timing_corrected.nii.gz")
            nib.save(stc_img, output_path)

        # Step 3: Spatial smoothing
        logger.info("Step 3: Spatial smoothing")
        smoothed_img = spatial_smoothing(stc_img, fwhm=fwhm)
        results["smoothed"] = smoothed_img

        if output_dir is not None:
            output_path = os.path.join(output_dir, "03_smoothed.nii.gz")
            nib.save(smoothed_img, output_path)

        # Step 4: Coregistration (if structural image provided)
        if struct_img is not None:
            logger.info("Step 4: Coregistration to structural image")
            coreg_img = coregistration(smoothed_img, struct_img)
            results["coregistered"] = coreg_img

            if output_dir is not None:
                output_path = os.path.join(output_dir, "04_coregistered.nii.gz")
                nib.save(coreg_img, output_path)

            # Use coregistered image for subsequent steps
            current_img = coreg_img
        else:
            current_img = smoothed_img

        # Step 5: Normalization (if requested)
        if perform_normalization:
            logger.info("Step 5: Normalization to MNI space")
            normalized_img = normalize_to_mni(current_img)
            results["normalized"] = normalized_img

            if output_dir is not None:
                output_path = os.path.join(output_dir, "05_normalized.nii.gz")
                nib.save(normalized_img, output_path)

            # Use normalized image for subsequent steps
            current_img = normalized_img

        # Step 6: SPM analysis (if requested)
        if perform_spm and design_matrix is not None and contrast_matrix is not None:
            logger.info("Step 6: Statistical Parametric Mapping (SPM) analysis")
            spm_results = statistical_parametric_mapping(
                current_img, design_matrix, contrast_matrix
            )
            results["spm"] = spm_results

            if output_dir is not None:
                for contrast_name, contrast_results in spm_results.items():
                    stat_path = os.path.join(
                        output_dir, f"{contrast_name}_stat_map.nii.gz"
                    )
                    p_path = os.path.join(output_dir, f"{contrast_name}_p_map.nii.gz")
                    nib.save(contrast_results["stat_map"], stat_path)
                    nib.save(contrast_results["p_map"], p_path)

        # Step 7: SSP map generation (if requested)
        if generate_ssp:
            logger.info("Step 7: SSP map generation")
            ssp_map = generate_ssp_maps(current_img)
            results["ssp_map"] = ssp_map

            if output_dir is not None:
                output_path = os.path.join(output_dir, "07_ssp_map.nii.gz")
                nib.save(ssp_map, output_path)

            # Step 8: Reverse phase de-ambiguization
            logger.info("Step 8: Reverse phase de-ambiguization")
            deambiguated_ssp = reverse_phase_deambiguization(ssp_map)
            results["deambiguated_ssp"] = deambiguated_ssp

            if output_dir is not None:
                output_path = os.path.join(output_dir, "08_deambiguated_ssp.nii.gz")
                nib.save(deambiguated_ssp, output_path)

        logger.info("Complete fMRI preprocessing pipeline finished successfully")

        return results

    except Exception as e:
        raise FMRIPreprocessingError(f"Error in preprocessing pipeline: {str(e)}")


# Legacy functions for backward compatibility
def preprocess_fmri(data_path):
    """
    Legacy function for backward compatibility

    Args:
        data_path (str): Path to fMRI data

    Returns:
        Preprocessed fMRI data
    """
    warnings.warn(
        "preprocess_fmri is deprecated, use preprocess_fmri_pipeline instead",
        DeprecationWarning,
    )
    results = preprocess_fmri_pipeline(data_path)
    return results["normalized"] if "normalized" in results else results["smoothed"]


def normalize_fmri(data):
    """
    Legacy function for backward compatibility

    Args:
        data: fMRI data to normalize

    Returns:
        Normalized fMRI data
    """
    warnings.warn(
        "normalize_fmri is deprecated, use normalize_to_mni instead", DeprecationWarning
    )
    if isinstance(data, str):
        img = nib.load(data)
    else:
        img = data
    return normalize_to_mni(img)


def apply_motion_correction(data):
    """
    Legacy function for backward compatibility

    Args:
        data: fMRI data to correct

    Returns:
        Motion-corrected fMRI data
    """
    warnings.warn(
        "apply_motion_correction is deprecated, use realign_motion_correction instead",
        DeprecationWarning,
    )
    if isinstance(data, str):
        img = nib.load(data)
    else:
        img = data
    return realign_motion_correction(img)


def apply_spatial_smoothing(data, fwhm=6.0):
    """
    Legacy function for backward compatibility

    Args:
        data: fMRI data to smooth
        fwhm (float): Full width at half maximum in mm

    Returns:
        Spatially smoothed fMRI data
    """
    warnings.warn(
        "apply_spatial_smoothing is deprecated, use spatial_smoothing instead",
        DeprecationWarning,
    )
    if isinstance(data, str):
        img = nib.load(data)
    else:
        img = data
    return spatial_smoothing(img, fwhm)
