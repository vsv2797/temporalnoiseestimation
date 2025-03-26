import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from miaplpy.objects.slcStack import slcStack
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt , lfilter
from spatz.change.phase_noise_time import temporallyUnwrapArcPhase , computeIfgsAndBaselines
from scipy.spatial import KDTree
import datetime

def load_slc_stack(file_path,geometry_file):
    """Loads SLC stack, temporal baselines (tbase), and perpendicular baselines (pbase) from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        #print("Available datasets in HDF5 file:", list(f.keys()))  # Debugging output
        
        # Ensure required datasets exist
        if 'slc' not in f or 'date' not in f or 'bperp' not in f:
            raise KeyError(f"Missing required datasets in {file_path}. Available datasets: {list(f.keys())}")

        # Load SLC stack
        slc = np.array(f['slc'])  # Shape: (num_ifgs, height, width)

        # Load acquisition dates and convert them to decimal years
        acquisition_dates = np.array(f['date'])  # Dates in YYYYMMDD format
        acquisition_dates = [datetime.datetime.strptime(str(d, "utf-8"), "%Y%m%d") for d in acquisition_dates]
        tbase = np.array([(date - acquisition_dates[0]).days for date in acquisition_dates]) / 365.25  # Convert to years

        # Load perpendicular baselines
        pbase = np.array(f['bperp'])  # Shape: (num_ifgs,)
        # Load `loc_inc` and `slant_range` from geometry file
    with h5py.File(geometry_file, 'r') as g:
        #print("Available datasets in geometry HDF5 file:", list(g.keys()))  # Debugging output

        if 'incidenceAngle' not in g or 'slantRangeDistance' not in g:
            raise KeyError(f"Missing required datasets in {geometry_file}. Available datasets: {list(g.keys())}")

        loc_inc = np.array(g['incidenceAngle'])
        loc_inc = np.deg2rad(loc_inc)  # Convert to radians

        slant_range = np.array(g['slantRangeDistance'])  # Extract slant range


    print("SLC Stack Loaded Successfully")
    #print(f"SLC Shape: {slc.shape}, Temporal Baselines: {tbase.shape}, Perpendicular Baselines: {pbase.shape}")
    
    return slc, tbase, pbase,loc_inc, slant_range
def compute_virtual_reference(slc_stack, first_order_mask, tcs_mask, num_nearest_neighbors=5):
    """Computes virtual reference pixels for interferometry using nearest neighbors."""
    slc_cand = slc_stack[:, first_order_mask]
    slc_p1 = slc_stack[:, tcs_mask]
    
    coord_xy_cand = np.array(np.where(first_order_mask)).T
    coord_xy_p1 = np.array(np.where(tcs_mask)).T
    
    searchtree = KDTree(coord_xy_p1)
    slc_virt_ref_list = []
    
    for idx in range(coord_xy_cand.shape[0]):
        _, idx_p1 = searchtree.query(coord_xy_cand[idx], k=num_nearest_neighbors + 1)
        idx_p1 = idx_p1[1:]  # Remove self-reference
        slc_virt_ref_list.append(slc_p1[:, idx_p1])
    
    slc_virt_ref = np.stack(slc_virt_ref_list, axis=1)
    if slc_virt_ref.ndim == 3:
        slc_virt_ref = slc_virt_ref[:, :, 0]  # Reduce dimensionality if needed
    
    return slc_cand, slc_virt_ref
import numpy as np

def compute_ifg_stack(slc_stack, tbase, pbase, slc_cand, slc_p1, idx_p1):
    """
    Computes the interferometric phase stack (IFG stack) from SLC stack using candidate and virtual reference pixels.

    Parameters:
    - slc_stack: (num_images, height, width) complex SLC data.
    - tbase: Temporal baselines.
    - pbase: Perpendicular baselines.
    - slc_cand: (num_images, num_cand) SLC candidate pixels.
    - slc_p1: (num_images, num_tcs) SLC pixels for potential virtual reference.
    - idx_p1: (num_cand, num_neighbors) Indices of the nearest TCS pixels.

    Returns:
    - ifg_stack: (num_ifgs, height, width) Interferometric phase stack.
    - tbase_ifg: Temporal baselines of interferograms.
    - pbase_ifg: Perpendicular baselines of interferograms.
    """
    
    num_images, height, width = slc_stack.shape

    # Ensure idx_p1 is a valid integer array
    idx_p1 = np.array(idx_p1, dtype=int)  # Fix: Ensure integer type for indexing

    # Ensure each candidate pixel gets ONE valid virtual reference
    num_cand = slc_cand.shape[1]  # Number of candidate pixels
    slc_virt_ref = np.zeros((num_images, num_cand), dtype=complex)  # Allocate array

    for i in range(num_cand):  # Loop over each candidate pixel
        valid_indices = idx_p1[i]  # Get nearest neighbors for candidate i

        if len(valid_indices) == 0:
            raise ValueError(f"No valid TCS neighbors found for candidate pixel {i}")

        # Compute virtual reference as the mean of nearest TCS pixels
        slc_virt_ref[:, i] = np.mean(slc_p1[:, valid_indices], axis=1)  

    # Ensure correct candidate count before concatenation
    if slc_virt_ref.shape[1] != num_cand:
        raise ValueError(
            f"Mismatch in candidate size: slc_cand has {num_cand}, but slc_virt_ref has {slc_virt_ref.shape[1]}"
        )

    # Concatenate candidate SLC pixels with virtual reference
    slc_all = np.concatenate((slc_cand, slc_virt_ref), axis=1)

    # Choose a reference index (middle image)
    ref_idx = num_images // 2  

    # Compute IFG stack
    ifg_stack, tbase_ifg, pbase_ifg = computeIfgsAndBaselines(
        slc_all=slc_all, tbase=tbase, pbase=pbase, ref_idx=ref_idx
    )

    # Reshape IFG stack back to spatial dimensions
    ifg_stack = ifg_stack.reshape(ifg_stack.shape[0], height, width)

    return ifg_stack, tbase_ifg, pbase_ifg
def compute_adi(slc_stack):
    """Computes ADI from the SLC stack."""
    amp_stack = np.abs(slc_stack)
    mean_amp = np.mean(amp_stack, axis=0)
    std_amp = np.std(amp_stack, axis=0)
    mean_amplitude = mean_amp / np.max(mean_amp)
    adi = std_amp / mean_amp
    return adi , mean_amplitude

def select_points(adi, ADI_THR_PS, ADI_THR_TCS):
    """Selects first-order and TCS points based on ADI thresholds."""
    first_order_mask = adi < ADI_THR_PS
    tcs_mask = (adi >= ADI_THR_PS) & (adi < ADI_THR_TCS)
    return first_order_mask, tcs_mask

def select_reference_pixel(mask):
    """Selects a valid reference pixel within the bounds of the dataset."""
    indices = np.argwhere(mask)  # Get (row, col) indices of valid pixels
    if len(indices) == 0:
        raise ValueError("No valid first-order pixels found!")
    ref_pixel = indices[0]  # Select the first valid pixel
    return ref_pixel[0], ref_pixel[1]  # (row, col)

def compute_arc_phase(ifg_stack, ref_row, ref_col):
    """Computes arc phases relative to a reference index, ensuring proper phase computation and unwrapping."""
    num_slc, num_row, num_col = ifg_stack.shape
    reference = ifg_stack[:, ref_row, ref_col]
    arc_phases = np.angle(ifg_stack.reshape(ifg_stack.shape[0], -1) * np.conjugate(reference[:, np.newaxis])) # Correct computation
    arc_phases = arc_phases.reshape(num_slc, num_row, num_col)

    # / reference[:, np.newaxis, np.newaxis]
    #arc_phases = np.unwrap(arc_phases, axis=0)  # Phase unwrapping to prevent discontinuities
    return arc_phases

import numpy as np

def compute_ifg_coherence_network(slc_stack, tbase, pbase, coherence_threshold=0.6):
    """
    Computes an interferometric phase stack (IFG stack), temporal coherence map (γ), 
    and an IFG Star Network using `computeIfgsAndBaselines`.

    Parameters:
    - slc_stack: (num_acquisitions, height, width) complex SLC data.
    - tbase: (num_acquisitions,) Temporal baselines.
    - pbase: (num_acquisitions,) Perpendicular baselines.
    - coherence_threshold: Minimum coherence for valid interferograms.

    Returns:
    - gamma_map: (height, width) Temporal coherence values (0-1).
    - ifg_stack: (num_ifgs, height, width) Interferometric phase stack.
    - valid_ifg_pairs: List of selected (master, slave) IFGs forming the star network.
    - master_idx: Index of the chosen master SLC.
    - tbase_ifg: Temporal baselines of interferograms.
    - pbase_ifg: Perpendicular baselines of interferograms.
    """
    master_idx = slc_stack.shape[0] // 2  

    # Reshape SLC stack for processing (num_images, num_pixels)
    slc_all = slc_stack.reshape(slc_stack.shape[0], -1)
    # Compute IFG stack and baselines
    ifg_stack, tbase_ifg, pbase_ifg = computeIfgsAndBaselines(
        slc_all=slc_all,
        pbase=pbase,
        tbase=tbase,
        ref_idx=master_idx
    )
    ifg_stack = ifg_stack.reshape(ifg_stack.shape[0], slc_stack.shape[1], slc_stack.shape[2])
    # Compute temporal coherence (γ) correctly as in phase_noise_time.py
    gamma_map = np.abs(np.mean(np.exp(1j * ifg_stack), axis=0))
    # Create IFG Star Network (Filter by coherence)
    valid_ifg_pairs = []
    for i in range(ifg_stack.shape[0]):
        mean_coherence = np.mean(gamma_map)  # Compute average coherence
        if mean_coherence > coherence_threshold:
            valid_ifg_pairs.append((master_idx, i))
    return gamma_map, ifg_stack, valid_ifg_pairs, master_idx, tbase_ifg, pbase_ifg

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Apply a Butterworth low-pass filter to smooth phase series.
    
    Parameters:
    - data (np.ndarray): 1D array of unwrapped phase values.
    - cutoff (float): Cutoff frequency for the filter (Hz).
    - fs (float): Sampling frequency (assumed time intervals).
    - order (int): Order of the Butterworth filter.

    Returns:
    - np.ndarray: Smoothed phase series.
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalize cutoff
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    n_samples = len(data)
    pad_length = max(3 * order, 15)  # Required pad length for filtfilt
    if n_samples > pad_length:  
        smoothed_data = filtfilt(b, a, data)  # Use filtfilt if enough data
        #print('filtfilt is being used')
    elif n_samples > order:  
        smoothed_data = lfilter(b, a, data)  # Use lfilter if slightly short
       # print('lfilter is being used')
    else:
        print(f"⚠ Warning: Skipping Butterworth filter (data too short: {n_samples} samples)")
        smoothed_data = data  # Return raw data if not enough samples

    return smoothed_data
def sav_golay_smooth(arc_phase, window_length, polyorder):
    """
    Apply a Savitzky-Golay filter to smooth the phase series.

    Parameters:
    - phase_series (np.ndarray): 1D array of unwrapped phase values.
    - window_length (int): The length of the filter window (must be odd).
    - polyorder (int): Polynomial order for Savitzky-Golay filter.

    Returns:
    - np.ndarray: Smoothed phase series.
    """
    n_samples = len(arc_phase)  # Get length of phase series
    # Ensure window_length does not exceed available data
    adjusted_window = min(window_length, n_samples)
    
    # Ensure window_length is at least 3 (minimum for smoothing)
    if adjusted_window < 3:
        adjusted_window = 3  
    # Ensure window_length is odd (required by savgol_filter)
    if adjusted_window % 2 == 0:
        adjusted_window -= 1  
    # Ensure window_length is greater than polyorder
    if adjusted_window <= polyorder:
        adjusted_window = polyorder + 1  
    # Apply smoothing only if valid window size is possible
    if adjusted_window >= 3 and n_samples >= adjusted_window:
        return savgol_filter(arc_phase, adjusted_window, polyorder)
    return arc_phase  # Return original if filtering is not possible
