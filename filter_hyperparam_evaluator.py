
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from func import load_slc_stack, compute_adi, select_points, compute_virtual_reference
import pandas as pd

def butter_lowpass_filter(data, cutoff, order):
    fs = 0.0667
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    z = np.exp(1j * data)
    if len(data) >= max(3 * order, 15):
        smoothed_data = filtfilt(b, a, z)
    else:
        smoothed_data = z
    return np.angle(smoothed_data)

def sav_golay_smooth(data, window_length, polyorder):
    adjusted_window = min(window_length, len(data))
    if adjusted_window < 3:
        adjusted_window = 3
    if adjusted_window % 2 == 0:
        adjusted_window -= 1
    if adjusted_window <= polyorder:
        adjusted_window = polyorder + 1
    if adjusted_window > len(data):
        return data
    return savgol_filter(data, adjusted_window, polyorder)

def computeIfgsAndBaselines(slc_all, tbase, pbase, ref_idx):
    ifgs_all = slc_all * np.conjugate(slc_all[ref_idx, :])
    tbase_ifg = tbase - tbase[ref_idx]
    pbase_ifg = pbase - pbase[ref_idx]
    ifgs_all = np.delete(ifgs_all, ref_idx, axis=0)
    tbase_ifg = np.delete(tbase_ifg, ref_idx, axis=0)
    pbase_ifg = np.delete(pbase_ifg, ref_idx, axis=0)
    return ifgs_all, tbase_ifg, pbase_ifg

# Load data
slc_stack_path = "/home/vavilla/sarvey/spatz/data/inputs/slcStack.h5"
geometry_file_path = "/home/vavilla/sarvey/spatz/data/inputs/geometryRadar.h5"
slc_stack, tbase, pbase, loc_inc, slant_range = load_slc_stack(slc_stack_path, geometry_file_path)
adi, _ = compute_adi(slc_stack)
first_order_mask, tcs_mask = select_points(adi, 0.3, 0.4)
slc_cand, slc_virt_ref = compute_virtual_reference(slc_stack, first_order_mask, tcs_mask, num_nearest_neighbors=5)

# Sample arc
idx = np.random.randint(0,slc_cand.shape[1])
slc1 = slc_cand[:, idx]
slc2 = slc_virt_ref[:, idx, np.newaxis]
slc_all = np.concatenate((slc1[:, None], slc2), axis=1)
ifgs, tbase_ifg, _ = computeIfgsAndBaselines(slc_all, tbase, pbase, ref_idx=slc1.shape[0] // 2)
arc_phase = np.angle(np.mean(ifgs[:, 1:], axis=1) * np.conjugate(ifgs[:, 0]))

def evaluate_residual(arc, smoothed):
    residual = arc - smoothed
    residual = np.angle(np.exp(1j * residual) * np.conj(np.mean(np.exp(1j * residual))))
    coh = np.abs(np.mean(np.exp(1j * residual)))
    var = np.var(residual)
    score = coh / var if var > 0 else 0
    return coh, var, score

# Test filters
bw_results = []
cutoffs = [0.0001, 0.001, 0.005]
orders = [2, 3, 4, 5]

for cutoff in cutoffs:
    for order in orders:
        try:
            smoothed = butter_lowpass_filter(arc_phase, cutoff, order)
            coh, var, score = evaluate_residual(arc_phase, smoothed)
            bw_results.append((cutoff, order, coh, var, score))
        except Exception as e:
            bw_results.append((cutoff, order, 0, np.inf, 0))

sg_results = []
window_lengths = [5, 7, 11]
poly_orders = [2, 3, 4, 5]

for win in window_lengths:
    for poly in poly_orders:
        try:
            smoothed = sav_golay_smooth(arc_phase, win, poly)
            coh, var, score = evaluate_residual(arc_phase, smoothed)
            sg_results.append((win, poly, coh, var, score))
        except Exception as e:
            sg_results.append((win, poly, 0, np.inf, 0))

# Save results
bw_df = pd.DataFrame(bw_results, columns=["cutoff", "order", "coherence", "residual_var", "score"])
sg_df = pd.DataFrame(sg_results, columns=["window_length", "poly_order", "coherence", "residual_var", "score"])

bw_df.to_csv("bw_filter_eval.csv", index=False)
sg_df.to_csv("sg_filter_eval.csv", index=False)

print("Evaluation complete. Saved: bw_filter_eval.csv and sg_filter_eval.csv")
