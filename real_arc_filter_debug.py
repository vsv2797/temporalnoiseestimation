
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, savgol_filter
from func import load_slc_stack, compute_adi, select_points, compute_virtual_reference

def butter_lowpass_filter(data, cutoff, order):
    fs = 0.072
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

# Get a sample arc
idx = np.random.randint(0,slc_cand.shape[1])
slc1 = slc_cand[:, idx]
slc2 = slc_virt_ref[:, idx, np.newaxis]
slc_all = np.concatenate((slc1[:, None], slc2), axis=1)
ifgs, tbase_ifg, _ = computeIfgsAndBaselines(slc_all, tbase, pbase, ref_idx=slc1.shape[0] // 2)
arc_phase = np.angle(np.mean(ifgs[:, 1:], axis=1) * np.conjugate(ifgs[:, 0]))

# Sweep parameters
cutoffs = [0.0001,0.005, 0.001]
orders = [2, 3, 4, 5]
window_lengths = [5, 7, 11]
poly_orders = [2, 3,4,5]

# Butterworth plots
fig, axs = plt.subplots(len(cutoffs), len(orders), figsize=(14, 8))
for i, cutoff in enumerate(cutoffs):
    for j, order in enumerate(orders):
        smoothed = butter_lowpass_filter(arc_phase, cutoff=cutoff, order=order)
        res = arc_phase - smoothed
        ax = axs[i, j]
        ax.plot(arc_phase, '.', label='Observed')
        ax.plot(smoothed, '.', label='Smoothed')
        ax.plot(res, '.', label='Residual')
        ax.set_title(f"BW cutoff={cutoff}, order={order}")
        ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig("butterworth_sweep.png")

# Sav-Golay plots
fig, axs = plt.subplots(len(window_lengths), len(poly_orders), figsize=(14, 8))
for i, win in enumerate(window_lengths):
    for j, poly in enumerate(poly_orders):
        smoothed = sav_golay_smooth(arc_phase, win, poly)
        res = arc_phase - smoothed
        ax = axs[i, j]
        ax.plot(arc_phase, '.', label='Observed')
        ax.plot(smoothed, '', label='Smoothed')
        ax.plot(res, '.', label='Residual')
        ax.set_title(f"SG window={win}, poly={poly}")
        ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig("savgolay_sweep.png")
