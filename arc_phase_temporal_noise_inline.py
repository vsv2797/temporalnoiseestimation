
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.spatial import KDTree
from scipy.signal import butter, filtfilt, savgol_filter
from spatz.change.likelihood import logLikelihoodInterferometricPhase
from miaplpy.objects.slcStack import slcStack
from spatz.utils.save_plot import save_and_close_plot
from func import load_slc_stack, compute_adi, select_points, compute_virtual_reference
import os
import h5py
from tqdm import tqdm

def computeIfgsAndBaselines(slc_all, tbase, pbase, ref_idx):
    ifgs_all = slc_all * np.conjugate(slc_all[ref_idx, :])
    tbase_ifg = tbase - tbase[ref_idx]
    pbase_ifg = pbase - pbase[ref_idx]
    ifgs_all = np.delete(ifgs_all, ref_idx, axis=0)
    tbase_ifg = np.delete(tbase_ifg, ref_idx, axis=0)
    pbase_ifg = np.delete(pbase_ifg, ref_idx, axis=0)
    return ifgs_all, tbase_ifg, pbase_ifg

def butter_lowpass_filter(data, cutoff, tbase, order):
    fs = 0.0667  # hardcoded sampling frequency
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
    return savgol_filter(data, adjusted_window, polyorder) if len(data) >= adjusted_window else data

def temporallyUnwrapArcPhase(arc_phase, tbase_ifg,show_plots:False):
    smoothed_bw = butter_lowpass_filter(arc_phase, cutoff=0.005, tbase=tbase_ifg, order=2)
    smoothed_sg = sav_golay_smooth(arc_phase, window_length=11, polyorder=2)
    res_bw = arc_phase - smoothed_bw
    res_sg = arc_phase - smoothed_sg
    coh_bw = np.abs(np.mean(np.exp(1j * res_bw)))
    coh_sg = np.abs(np.mean(np.exp(1j * res_sg)))
    if show_plots:
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2)
        ax0 = fig.add_subplot(gs[0, 0])
        #ax0.plot(ifg_res, '.', label="Residual phase")
        ax0.set_title('Butterworth filtered arc phase (freq- domain)')
        ax0.plot(smoothed_bw, '.', label="smoothed phase")
        ax0.plot(arc_phase, '.', label="Observed phase")
        ax0.plot(res_bw, '.', label="Smoothed Residual phase")
        plt.legend()
        plt.xlabel("Interferogram index")
        plt.ylabel("Phase")
        ax0 = fig.add_subplot(gs[1, 0])
        #ax0.plot(ifg_res, '.', label="Residual phase")
        ax0.set_title('sav-golay filtered arc phase (time - domain)')
        ax0.plot(smoothed_sg, '.', label="smoothed phase")
        ax0.plot(arc_phase, '.', label="Observed phase")
        ax0.plot(res_sg, '.', label="Smoothed Residual phase")
        plt.legend()
        plt.xlabel("Interferogram index")
        plt.ylabel("Phase")
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.hist(res_bw.flatten(), bins=200, color='k')
        plt.xlabel("Phase noise (Butter Worth)")
        plt.ylabel("Number of pixels")
        plt.subplots_adjust(wspace=0.3)
        ax1 = fig.add_subplot(gs[1, 1])
        ax1.hist(res_sg.flatten(), bins=200, color='k')
        plt.xlabel("Phase noise(Sav-golay)")
        plt.ylabel("Number of pixels")
        plt.subplots_adjust(wspace=0.3)
        save_and_close_plot("/home/vavilla/sarvey/spatz2/sai.vavilla/data/outputs_tcs/1")

    return coh_bw, res_bw, coh_sg, res_sg

def singleCD(slc_cand, slc_virt_ref, tbase, pbase, loc_inc, slant_range, wavelength, demerr_bound, velocity_bound, num_samples):
    num_images = slc_cand.shape[0]
    slc_all = np.concatenate((slc_cand.reshape(-1, 1), slc_virt_ref), axis=1)
    mask_time = np.zeros(num_images, dtype=bool)
    score_bw = np.full(num_images - 1, np.nan)
    score_sg = np.full(num_images - 1, np.nan)

    ifgs_all, tbase_ifg, pbase_ifg = computeIfgsAndBaselines(slc_all, tbase, pbase, ref_idx=num_images // 2)
    ifgs_virt_ref = np.mean(ifgs_all[:, 1:], axis=1)
    arc_phase = np.angle(ifgs_virt_ref * np.conjugate(ifgs_all[:, 0]))
    coh0, noise0, coh_sg0, noise_sg0 = temporallyUnwrapArcPhase(arc_phase, tbase_ifg,show_plots=False)
    noise0 = np.angle(np.exp(1j * noise0) * np.conjugate(np.mean(np.exp(1j * noise0))))
    noise_sg0 = np.angle(np.exp(1j * noise_sg0) * np.conjugate(np.mean(np.exp(1j * noise_sg0))))
    p0 = logLikelihoodInterferometricPhase(coh0, noise0, noise0)
    p0_sg = logLikelihoodInterferometricPhase(coh_sg0, noise_sg0, noise_sg0)

    for step in range(10, num_images - 11):
        mask_time[:] = False
        mask_time[:step+1] = True

        ifgs_1, tbase_1, _ = computeIfgsAndBaselines(slc_all[mask_time], tbase[mask_time], pbase[mask_time], ref_idx=-1)
        ifgs_2, tbase_2, _ = computeIfgsAndBaselines(slc_all[~mask_time], tbase[~mask_time], pbase[~mask_time], ref_idx=0)
        arc_1 = np.angle(np.mean(ifgs_1[:, 1:], axis=1) * np.conjugate(ifgs_1[:, 0]))
        arc_2 = np.angle(np.mean(ifgs_2[:, 1:], axis=1) * np.conjugate(ifgs_2[:, 0]))
        coh1, res1, coh_sg1, res_sg1 = temporallyUnwrapArcPhase(arc_1, tbase_1,show_plots=False)
        coh2, res2, coh_sg2, res_sg2 = temporallyUnwrapArcPhase(arc_2, tbase_2,show_plots=False)
        res1, res2 = [np.angle(np.exp(1j*r) * np.conjugate(np.mean(np.exp(1j*r)))) for r in (res1, res2)]
        res_sg1, res_sg2 = [np.angle(np.exp(1j*r) * np.conjugate(np.mean(np.exp(1j*r)))) for r in (res_sg1, res_sg2)]

        p1 = logLikelihoodInterferometricPhase(coh1, res1, 0)
        p2 = logLikelihoodInterferometricPhase(coh2, res2, 0)
        p1_sg = logLikelihoodInterferometricPhase(coh_sg1, res_sg1, 0)
        p2_sg = logLikelihoodInterferometricPhase(coh_sg2, res_sg2, 0)

        score_bw[step] = (p0 / len(arc_phase)) - ((p1 + p2) / (len(res1) + len(res2)))
        score_sg[step] = (p0_sg / len(arc_phase)) - ((p1_sg + p2_sg) / (len(res_sg1) + len(res_sg2)))

    return score_bw, score_sg, max(coh1, coh2), max(coh_sg1, coh_sg2), coh0, coh_sg0

# ==== MAIN EXECUTION ====
slc_stack_path = "/home/vavilla/sarvey/spatz/data/inputs/slcStack.h5"
geometry_file_path = os.path.join(os.path.dirname(slc_stack_path), "geometryRadar.h5")
slc_stack, tbase, pbase, loc_inc, slant_range = load_slc_stack(slc_stack_path, geometry_file_path)
ADI_THR_PS = 0.3
ADI_THR_TCS = 0.4
adi, _ = compute_adi(slc_stack)
first_order_mask, tcs_mask = select_points(adi, ADI_THR_PS, ADI_THR_TCS)
slc_cand, slc_virt_ref = compute_virtual_reference(slc_stack, first_order_mask, tcs_mask, num_nearest_neighbors=5)
loc_inc_cand = loc_inc[first_order_mask]
slant_range_cand = slant_range[first_order_mask]

results = []

for idx in tqdm(range(slc_cand.shape[1]), desc="Processing TCS"):
    score_bw, score_sg, max_bw, max_sg, init_bw, init_sg = singleCD(
        slc_cand[:, idx],
        slc_virt_ref[:, idx, np.newaxis],
        tbase, pbase,
        float(loc_inc_cand[idx]),
        float(slant_range_cand[idx]),
        wavelength=0.055,
        demerr_bound=20,
        velocity_bound=5,
        num_samples=100
    )
    results.append((idx, np.nanargmin(score_bw), max_bw, init_bw))

# Create change map
change_map = np.full(slc_stack.shape[1:], np.nan)
coherence_map = np.full(slc_stack.shape[1:], np.nan)
coords = np.argwhere(first_order_mask)
for (i, change_idx, max_coh, _) in results:
    row, col = coords[i]
    change_map[row, col] = change_idx
    coherence_map[row, col] = max_coh
with h5py.File("change_maps_output.h5",'w') as f:
    f.create_dataset("change_index", data=change_map)
    f.create_dataset('coherence', data=coherence_map)
print("Saved the change map and coherence map outputs")
