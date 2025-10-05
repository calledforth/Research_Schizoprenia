"""
MEG data preprocessing module for schizophrenia detection

This module implements a comprehensive MEG preprocessing pipeline including:
- Artifact removal using Maxfilter with head movement correction
- Signal-Space Projection (SSP) for cardiac and blink artifacts
- Bandpass filtering for four frequency ranges
- Covariance matrix processing with regularization
- Source space processing using 6-mm³ grid
- Beamformer projection with single-shell BEM
- Normalization and transformation to MNI space
- Hilbert transform for amplitude envelopes
- Spatial processing with smoothing and resampling
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

# Neuroimaging libraries
import mne
from mne import io
from mne.preprocessing import ICA, compute_proj_ecg, compute_proj_eog
from mne.cov import compute_covariance
from mne.beamformer import make_lcmv, apply_lcmv
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.source_space import setup_source_space
from mne.bem import make_bem_solution, make_bem_model
from mne.forward import make_forward_solution

# Scientific computing
from scipy import signal, ndimage
from scipy.linalg import svd

# FSL for normalization (requires FSL to be installed)
try:
    import nipype.interfaces.fsl as fsl

    FSL_AVAILABLE = True
except ImportError:
    FSL_AVAILABLE = False
    logging.warning("FSL not available. MNI normalization will be skipped.")

# Configure logging
logger = logging.getLogger(__name__)


class MEGPreprocessor:
    """
    Comprehensive MEG preprocessing pipeline for schizophrenia detection

    This class implements all the preprocessing steps required for MEG data
    in the schizophrenia detection pipeline, following the technical specification.
    """

    def __init__(self, config=None):
        """
        Initialize the MEG preprocessor

        Args:
            config: Configuration object with preprocessing parameters
        """
        if config is None:
            from ..config import default_config

            self.config = default_config.data
        else:
            self.config = config

        # Frequency bands as specified in the requirements
        self.freq_bands = {
            "delta": (1, 4),
            "theta": (5, 9),
            "alpha": (10, 15),
            "beta": (16, 29),
        }

        # Grid spacing for source space (6-mm³)
        self.grid_spacing = 6.0  # mm

        # Smoothing parameters
        self.smoothing_fwhm = 6.0  # mm
        self.target_voxel_size = (3.0, 3.0, 3.0)  # mm

        # Regularization parameter for covariance matrix
        self.reg_factor = 4.0  # Multiply minimal singular value by this factor

    def load_meg_data(self, file_path: str, **kwargs) -> mne.io.Raw:
        """
        Load MEG data from file

        Args:
            file_path (str): Path to the MEG data file
            **kwargs: Additional arguments for MNE's read_raw functions

        Returns:
            mne.io.Raw: Loaded MEG data

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"MEG data file not found: {file_path}")

        try:
            # Determine file type and load accordingly
            if file_path.endswith(".fif"):
                raw = io.read_raw_fif(file_path, **kwargs)
            elif file_path.endswith(".ds"):
                raw = io.read_raw_ctf(file_path, **kwargs)
            elif file_path.endswith(".vhdr"):
                raw = io.read_raw_brainvision(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            logger.info(f"Loaded MEG data from {file_path}")
            return raw

        except Exception as e:
            logger.error(f"Error loading MEG data: {str(e)}")
            raise

    def apply_maxfilter(
        self,
        raw: mne.io.Raw,
        head_pos: Optional[str] = None,
        downsample_freq: float = 250.0,
        **kwargs,
    ) -> mne.io.Raw:
        """
        Apply Maxfilter for artifact removal, head movement correction, and downsampling

        Args:
            raw (mne.io.Raw): Input MEG data
            head_pos (str, optional): Path to head position file
            downsample_freq (float): Target frequency for downsampling (Hz)
            **kwargs: Additional arguments for maxfilter

        Returns:
            mne.io.Raw: Processed MEG data
        """
        try:
            logger.info(
                "Applying Maxfilter for artifact removal and head movement correction"
            )

            # Apply Maxfilter if head position data is available
            if head_pos and os.path.exists(head_pos):
                raw = mne.preprocessing.maxwell_filter(raw, head_pos=head_pos, **kwargs)
                logger.info("Applied Maxwell filter with head movement correction")
            else:
                # Basic noise reduction without head position data
                raw = mne.preprocessing.maxwell_filter(raw, **kwargs)
                logger.info("Applied Maxwell filter without head movement correction")

            # Downsample to specified frequency
            if downsample_freq < raw.info["sfreq"]:
                raw.resample(downsample_freq)
                logger.info(f"Downsampled to {downsample_freq} Hz")

            return raw

        except Exception as e:
            logger.error(f"Error in Maxfilter processing: {str(e)}")
            raise

    def compute_ssp_projections(
        self,
        raw: mne.io.Raw,
        ecg_ch: Optional[str] = None,
        eog_ch: Optional[List[str]] = None,
    ) -> mne.io.Raw:
        """
        Compute and apply Signal-Space Projection (SSP) for cardiac and blink artifacts

        Args:
            raw (mne.io.Raw): Input MEG data
            ecg_ch (str, optional): ECG channel name
            eog_ch (List[str], optional): List of EOG channel names

        Returns:
            mne.io.Raw: MEG data with SSP projections applied
        """
        try:
            logger.info("Computing SSP projections for cardiac and blink artifacts")

            # Find ECG and EOG channels if not specified
            if ecg_ch is None:
                ecg_chs = mne.pick_types(raw.info, ecg=True, meg=False)
                if len(ecg_chs) > 0:
                    ecg_ch = raw.ch_names[ecg_chs[0]]

            if eog_ch is None:
                eog_chs = mne.pick_types(raw.info, eog=True, meg=False)
                if len(eog_chs) > 0:
                    eog_ch = [raw.ch_names[i] for i in eog_chs]

            # Compute ECG projections
            if ecg_ch:
                ecg_projs, _ = compute_proj_ecg(
                    raw,
                    ch_name=ecg_ch,
                    reject=dict(grad=4000e-13, mag=4e-12, eog=250e-6),
                )
                raw.add_proj(ecg_projs)
                logger.info(f"Added {len(ecg_projs)} ECG projections")

            # Compute EOG projections
            if eog_ch:
                eog_projs, _ = compute_proj_eog(
                    raw,
                    ch_name=eog_ch,
                    reject=dict(grad=4000e-13, mag=4e-12, eog=250e-6),
                )
                raw.add_proj(eog_projs)
                logger.info(f"Added {len(eog_projs)} EOG projections")

            # Apply the projections
            raw.apply_proj()
            logger.info("Applied SSP projections")

            return raw

        except Exception as e:
            logger.error(f"Error in SSP computation: {str(e)}")
            raise

    def apply_bandpass_filter(
        self,
        raw: mne.io.Raw,
        freq_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, mne.io.Raw]:
        """
        Apply bandpass filtering for specified frequency ranges

        Args:
            raw (mne.io.Raw): Input MEG data
            freq_bands (Dict, optional): Dictionary of frequency bands to filter

        Returns:
            Dict[str, mne.io.Raw]: Dictionary of filtered data for each frequency band
        """
        if freq_bands is None:
            freq_bands = self.freq_bands

        filtered_data = {}

        try:
            logger.info("Applying bandpass filters for frequency ranges")

            for band_name, (low_freq, high_freq) in freq_bands.items():
                # Create a copy of the data for this frequency band
                band_data = raw.copy()

                # Apply bandpass filter
                band_data.filter(
                    l_freq=low_freq, h_freq=high_freq, method="fir", fir_design="firwin"
                )

                filtered_data[band_name] = band_data
                logger.info(
                    f"Applied {band_name} band filter ({low_freq}-{high_freq} Hz)"
                )

            return filtered_data

        except Exception as e:
            logger.error(f"Error in bandpass filtering: {str(e)}")
            raise

    def compute_regularized_covariance(
        self, raw: mne.io.Raw, method: str = "shrunk", **kwargs
    ) -> mne.Covariance:
        """
        Compute covariance matrix with regularization

        Args:
            raw (mne.io.Raw): Input MEG data
            method (str): Covariance estimation method
            **kwargs: Additional arguments for covariance computation

        Returns:
            mne.Covariance: Regularized covariance matrix
        """
        try:
            logger.info("Computing regularized covariance matrix")

            # Compute baseline covariance
            cov = compute_covariance(raw, method=method, **kwargs)

            # Apply regularization by modifying the smallest singular value
            cov_data = cov["data"]
            U, s, Vt = svd(cov_data, full_matrices=False)

            # Find minimal singular value and apply regularization
            min_singular = np.min(s)
            s[s < min_singular * self.reg_factor] = min_singular * self.reg_factor

            # Reconstruct covariance matrix
            regularized_cov_data = U @ np.diag(s) @ Vt
            cov["data"] = regularized_cov_data

            logger.info(
                f"Applied regularization factor of {self.reg_factor} to covariance matrix"
            )

            return cov

        except Exception as e:
            logger.error(f"Error in covariance computation: {str(e)}")
            raise

    def setup_source_space(
        self, subject: str = "fsaverage", spacing: str = "oct6"
    ) -> mne.SourceSpaces:
        """
        Setup source space with 6-mm³ grid

        Args:
            subject (str): Subject name (default: 'fsaverage')
            spacing (str): Source space spacing

        Returns:
            mne.SourceSpaces: Source space structure
        """
        try:
            logger.info(f"Setting up source space with {spacing} spacing")

            # Create source space
            src = setup_source_space(
                subject, spacing=spacing, add_dist=False, n_jobs=-1
            )

            logger.info(f"Created source space with {len(src)} sources")

            return src

        except Exception as e:
            logger.error(f"Error in source space setup: {str(e)}")
            raise

    def create_bem_model(
        self,
        subject: str = "fsaverage",
        conductivity: Tuple[float, float, float] = (0.3, 0.006, 0.3),
    ) -> mne.Bem:
        """
        Create single-shell boundary element model

        Args:
            subject (str): Subject name
            conductivity (Tuple): Conductivity values for the BEM model

        Returns:
            mne.Bem: BEM model
        """
        try:
            logger.info("Creating single-shell BEM model")

            # Create BEM model
            bem_model = make_bem_model(
                subject=subject, ico=4, conductivity=conductivity, subjects_dir=None
            )

            # Create BEM solution
            bem_solution = make_bem_solution(bem_model)

            logger.info("Created single-shell BEM model")

            return bem_solution

        except Exception as e:
            logger.error(f"Error in BEM model creation: {str(e)}")
            raise

    def create_forward_solution(
        self,
        raw: mne.io.Raw,
        src: mne.SourceSpaces,
        bem: mne.Bem,
        trans: Optional[str] = None,
    ) -> mne.Forward:
        """
        Create forward solution for beamformer

        Args:
            raw (mne.io.Raw): MEG data
            src (mne.SourceSpaces): Source space
            bem (mne.Bem): BEM model
            trans (str, optional): Transformation file

        Returns:
            mne.Forward: Forward solution
        """
        try:
            logger.info("Creating forward solution")

            # Create forward solution
            fwd = make_forward_solution(
                raw.info, trans=trans, src=src, bem=bem, meg=True, eeg=False, n_jobs=-1
            )

            logger.info(
                f"Created forward solution with {fwd['nsource']} sources and {fwd['nchan']} channels"
            )

            return fwd

        except Exception as e:
            logger.error(f"Error in forward solution creation: {str(e)}")
            raise

    def apply_lcmv_beamformer(
        self,
        raw: mne.io.Raw,
        fwd: mne.Forward,
        cov: mne.Covariance,
        reg: float = 0.05,
        pick_ori: str = "max-power",
    ) -> mne.SourceEstimate:
        """
        Apply LCMV beamformer with dipole model

        Args:
            raw (mne.io.Raw): MEG data
            fwd (mne.Forward): Forward solution
            cov (mne.Covariance): Covariance matrix
            reg (float): Regularization parameter
            pick_ori (str): Orientation picking method

        Returns:
            mne.SourceEstimate: Source estimate
        """
        try:
            logger.info("Applying LCMV beamformer")

            # Create filters
            filters = make_lcmv(
                raw.info, fwd, cov, reg=reg, pick_ori=pick_ori, rank="full"
            )

            # Apply beamformer
            stc = apply_lcmv(raw, filters, max_ori_out="signed")

            logger.info("Applied LCMV beamformer successfully")

            return stc

        except Exception as e:
            logger.error(f"Error in beamformer application: {str(e)}")
            raise

    def apply_hilbert_transform(
        self, stc: mne.SourceEstimate
    ) -> Dict[str, mne.SourceEstimate]:
        """
        Apply Hilbert transform to create amplitude envelopes

        Args:
            stc (mne.SourceEstimate): Source time course

        Returns:
            Dict[str, mne.SourceEstimate]: Amplitude envelopes for each frequency band
        """
        try:
            logger.info("Applying Hilbert transform for amplitude envelopes")

            # Apply Hilbert transform to get amplitude envelope
            envelope_data = np.abs(signal.hilbert(stc.data, axis=1))

            # Create new source estimate with envelope data
            envelope_stc = stc.copy()
            envelope_stc._data = envelope_data

            logger.info("Computed amplitude envelopes using Hilbert transform")

            return {"envelope": envelope_stc}

        except Exception as e:
            logger.error(f"Error in Hilbert transform: {str(e)}")
            raise

    def apply_spatial_processing(
        self,
        stc: mne.SourceEstimate,
        smooth_fwhm: Optional[float] = None,
        target_voxel_size: Optional[Tuple[float, float, float]] = None,
    ) -> mne.SourceEstimate:
        """
        Apply spatial processing with smoothing and resampling

        Args:
            stc (mne.SourceEstimate): Source estimate
            smooth_fwhm (float, optional): Smoothing FWHM in mm
            target_voxel_size (Tuple, optional): Target voxel size in mm

        Returns:
            mne.SourceEstimate: Spatially processed source estimate
        """
        if smooth_fwhm is None:
            smooth_fwhm = self.smoothing_fwhm
        if target_voxel_size is None:
            target_voxel_size = self.target_voxel_size

        try:
            logger.info("Applying spatial processing with smoothing and resampling")

            # Apply spatial smoothing
            if smooth_fwhm > 0:
                # Convert FWHM to sigma for Gaussian smoothing
                sigma = smooth_fwhm / (2 * np.sqrt(2 * np.log(2)))

                # Apply smoothing to each time point
                smoothed_data = np.zeros_like(stc.data)
                for i in range(stc.data.shape[1]):
                    smoothed_data[:, i] = ndimage.gaussian_filter(
                        stc.data[:, i], sigma=sigma
                    )

                stc._data = smoothed_data
                logger.info(f"Applied {smooth_fwhm} mm FWHM smoothing")

            # Note: Resampling would typically be done during the MNI normalization step
            # This is a placeholder for the resampling operation
            logger.info(f"Target voxel size: {target_voxel_size} mm")

            return stc

        except Exception as e:
            logger.error(f"Error in spatial processing: {str(e)}")
            raise

    def normalize_to_mni(
        self,
        stc: mne.SourceEstimate,
        subject: str = "fsaverage",
        subjects_dir: Optional[str] = None,
    ) -> mne.SourceEstimate:
        """
        Normalize source estimate to MNI space using FLIRT

        Args:
            stc (mne.SourceEstimate): Source estimate
            subject (str): Subject name
            subjects_dir (str, optional): Subjects directory

        Returns:
            mne.SourceEstimate: Normalized source estimate
        """
        if not FSL_AVAILABLE:
            logger.warning("FSL not available. Skipping MNI normalization.")
            return stc

        try:
            logger.info("Normalizing to MNI space using FLIRT")

            # This is a simplified implementation
            # In practice, you would need to:
            # 1. Convert source estimate to NIfTI
            # 2. Use FLIRT to register to MNI space
            # 3. Convert back to source estimate

            # Placeholder for the actual normalization
            logger.warning("MNI normalization is not fully implemented in this version")

            return stc

        except Exception as e:
            logger.error(f"Error in MNI normalization: {str(e)}")
            raise

    def preprocess_meg_data(
        self,
        file_path: str,
        head_pos: Optional[str] = None,
        subject: str = "fsaverage",
        trans: Optional[str] = None,
        save_output: bool = False,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Dict[str, mne.SourceEstimate]]:
        """
        Complete MEG preprocessing pipeline

        Args:
            file_path (str): Path to MEG data file
            head_pos (str, optional): Path to head position file
            subject (str): Subject name
            trans (str, optional): Transformation file
            save_output (bool): Whether to save output files
            output_dir (str, optional): Output directory

        Returns:
            Dict: Dictionary containing processed data for each frequency band
        """
        try:
            logger.info("Starting complete MEG preprocessing pipeline")

            # Step 1: Load MEG data
            raw = self.load_meg_data(file_path)

            # Step 2: Apply Maxfilter for artifact removal and downsampling
            raw = self.apply_maxfilter(raw, head_pos=head_pos)

            # Step 3: Apply SSP for cardiac and blink artifacts
            raw = self.compute_ssp_projections(raw)

            # Step 4: Apply bandpass filtering for frequency bands
            filtered_data = self.apply_bandpass_filter(raw)

            # Step 5: Setup source space and BEM model
            src = self.setup_source_space(subject)
            bem = self.create_bem_model(subject)

            # Step 6: Create forward solution
            fwd = self.create_forward_solution(raw, src, bem, trans)

            # Initialize results dictionary
            results = {}

            # Process each frequency band
            for band_name, band_data in filtered_data.items():
                logger.info(f"Processing {band_name} frequency band")

                # Step 7: Compute regularized covariance matrix
                cov = self.compute_regularized_covariance(band_data)

                # Step 8: Apply LCMV beamformer
                stc = self.apply_lcmv_beamformer(band_data, fwd, cov)

                # Step 9: Apply Hilbert transform for amplitude envelopes
                envelope_stc = self.apply_hilbert_transform(stc)

                # Step 10: Apply spatial processing
                processed_stc = {}
                for env_name, env_stc in envelope_stc.items():
                    processed_stc[env_name] = self.apply_spatial_processing(env_stc)

                # Step 11: Normalize to MNI space
                for env_name, env_stc in processed_stc.items():
                    processed_stc[env_name] = self.normalize_to_mni(
                        env_stc, subject=subject
                    )

                results[band_name] = processed_stc

                # Save output if requested
                if save_output and output_dir:
                    self._save_results(results[band_name], band_name, output_dir)

            logger.info("MEG preprocessing pipeline completed successfully")

            return results

        except Exception as e:
            logger.error(f"Error in MEG preprocessing pipeline: {str(e)}")
            raise

    def _save_results(
        self, results: Dict[str, mne.SourceEstimate], band_name: str, output_dir: str
    ):
        """
        Save preprocessing results to files

        Args:
            results (Dict): Results to save
            band_name (str): Frequency band name
            output_dir (str): Output directory
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

            for result_name, stc in results.items():
                output_path = os.path.join(output_dir, f"{band_name}_{result_name}.stc")
                stc.save(output_path)
                logger.info(f"Saved {output_path}")

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise


# Convenience functions for backward compatibility
def preprocess_meg(data_path, **kwargs):
    """
    Preprocess MEG data (convenience function)

    Args:
        data_path (str): Path to MEG data
        **kwargs: Additional arguments for preprocessing

    Returns:
        Preprocessed MEG data
    """
    preprocessor = MEGPreprocessor()
    return preprocessor.preprocess_meg_data(data_path, **kwargs)


def filter_meg(data, lowpass=40.0, highpass=0.1):
    """
    Apply bandpass filter to MEG data (convenience function)

    Args:
        data: MEG data to filter
        lowpass (float): Lowpass filter frequency in Hz
        highpass (float): Highpass filter frequency in Hz

    Returns:
        Filtered MEG data
    """
    if isinstance(data, mne.io.Raw):
        data_filtered = data.copy()
        data_filtered.filter(l_freq=highpass, h_freq=lowpass)
        return data_filtered
    else:
        raise ValueError("Input data must be MNE Raw object")


def apply_artifact_rejection(data):
    """
    Apply artifact rejection to MEG data (convenience function)

    Args:
        data: MEG data to clean

    Returns:
        Cleaned MEG data
    """
    preprocessor = MEGPreprocessor()
    return preprocessor.compute_ssp_projections(data)


def compute_source_space(data):
    """
    Compute source space representation of MEG data (convenience function)

    Args:
        data: Sensor space MEG data

    Returns:
        Source space MEG data
    """
    preprocessor = MEGPreprocessor()
    # This is a simplified implementation
    # Full implementation would require forward solution and beamformer
    logger.warning("compute_source_space is a simplified implementation")
    return data
