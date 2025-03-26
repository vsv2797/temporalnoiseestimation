import numpy as np

def compute_temporal_coherence(ifg_stack):
    """
    Computes temporal coherence for each pixel across all interferograms.

    Parameters:
    - ifg_stack: (num_ifgs, height, width) Interferometric phase stack.

    Returns:
    - coherence_map: (height, width) Coherence values (0-1).
    """
    num_ifgs = ifg_stack.shape[0]
    mean_phase = np.mean(ifg_stack, axis=0)
    phase_fluctuations = ifg_stack - mean_phase  # Remove mean trend

    # Compute coherence as 1 - (standard deviation normalized by π)
    coherence_map = 1 - (np.std(phase_fluctuations, axis=0) / np.pi)
    coherence_map = np.clip(coherence_map, 0, 1)  # Ensure values between 0 and 1
    return coherence_map


def compute_adaptive_thresholds(coherence_map, noise_map, percentile=70):
    """
    Computes adaptive noise and coherence thresholds based on percentiles.

    Parameters:
    - coherence_map: Temporal coherence values.
    - noise_map: Phase noise values.
    - percentile: Percentile value to determine dynamic threshold.

    Returns:
    - noise_threshold: Adaptive phase noise threshold.
    - coherence_threshold: Adaptive coherence threshold.
    """
    noise_threshold = np.percentile(noise_map, percentile)  # Selects the lower 70% of phase noise
    coherence_threshold = np.percentile(coherence_map, 100 - percentile)  # Selects top 30% coherence values
    return noise_threshold, coherence_threshold

def estimate_phase_noise(arc_phases):
    """
    Estimates phase noise for each point.

    Parameters:
    - arc_phases: (num_ifgs, height, width) Arc phase data.

    Returns:
    - noise_map: (height, width) Phase noise estimation.
    """
    noise_map = np.std(arc_phases, axis=0)  # Standard deviation over time
    return noise_map
def filter_noisy_points(arc_phases, coherence_map, tcs_mask, first_order_mask):
    """
    Filters points based on noise level and coherence dynamically.

    Parameters:
    - arc_phases: (num_ifgs, height, width) Arc phase data.
    - coherence_map: (height, width) Temporal coherence values.
    - tcs_mask: Boolean mask of TCS points.
    - first_order_mask: Boolean mask of first-order points.

    Returns:
    - filtered_first_order_mask: Boolean mask of reliable first-order points.
    - filtered_tcs_mask: Boolean mask of reliable TCS points.
    """
    noise_map = estimate_phase_noise(arc_phases)

    # Compute adaptive thresholds
    noise_threshold, coherence_threshold = compute_adaptive_thresholds(coherence_map, noise_map)

    # Allow more flexibility for TCS points (lower coherence is fine)
    filtered_first_order_mask = (coherence_map >= coherence_threshold) & (noise_map <= noise_threshold)
    filtered_tcs_mask = (coherence_map >= (coherence_threshold * 0.8))  # Allow lower coherence for TCS

    return filtered_first_order_mask & first_order_mask, filtered_tcs_mask & tcs_mask
