import numpy as np
import matplotlib.pyplot as plt
import os
from miaplpy.objects.slcStack import slcStack
from spatz.change.phase_noise_time import  computeIfgsAndBaselines
from Interactive_plot import InteractiveArcPlot
from scipy.spatial import KDTree
import cmcrameri as cmc
from func import load_slc_stack , compute_adi ,select_points , select_reference_pixel , compute_arc_phase, plot_arc_phase, plot_comparison , butter_lowpass_filter ,sav_golay_smooth 

# Set the path to your SLC stack file
slc_stack_path  = "/home/vsv2797/sarvey/spatz2/sai.vavilla/data/inputs/slcStack.h5"
slc_stack_obj = slcStack(slc_stack_path)  # Initialize the slcStack object
print("slcStack metadata:", slc_stack_obj.__dict__)
geometry_file_path = os.path.join(os.path.dirname(slc_stack_path), "geometryRadar.h5")

# ADI threshold values
ADI_THR_PS = 0.25  # Threshold for first-order PS
ADI_THR_TCS = 0.3   # Threshold for TCS
print("Loading SLC stack...")
slc_stack , tbase, pbase,loc_inc, slant_range = load_slc_stack(slc_stack_path , geometry_file_path)
valid_mask = np.abs(slc_stack) > 0  # Mask valid (nonzero) pixels
slc_stack = slc_stack * valid_mask  # Keep only valid values
print("Computing ADI...")
adi , mean_amp = compute_adi(slc_stack)
print("Selecting first-order and TCS points...")
first_order_mask, tcs_mask = select_points(adi, ADI_THR_PS,ADI_THR_TCS)
# Extract SLC candidate pixels
slc_cand = slc_stack[:, first_order_mask]
# Extract additional subset of pixels (if `mask_p1` is defined)
mask_p1 = tcs_mask  # Use TCS mask for second subset (modify as needed)
coord_xy_p1 = np.array(np.where(mask_p1)).T
slc_p1 = slc_stack[:, mask_p1]
# Get the number of candidate pixels
num_cand = slc_cand.shape[1]
mask_cand = first_order_mask  # Use the first-order points as candidates
coord_xy_cand = np.array(np.where(mask_cand)).T
print(f"Number of candidate pixels: {num_cand}")
num_images = slc_cand.shape[0]
# Filter loc_inc and slant_range using mask_cand
loc_inc = loc_inc[mask_cand]
slant_range = slant_range[mask_cand]
searchtree = KDTree(coord_xy_p1)
num_nearest_neighbors = 5  # Adjust based on dataset

# Find nearest TCS points for each candidate pixel
slc_virt_ref_list = []
for idx in range(coord_xy_cand.shape[0]):
    _, idx_p1 = searchtree.query(coord_xy_cand[idx], k=num_nearest_neighbors + 1)
    idx_p1 = idx_p1[1:]  # Remove self-reference
    slc_virt_ref_list.append(slc_p1[:, idx_p1])

# Stack virtual references along second dimension
slc_virt_ref = np.stack(slc_virt_ref_list, axis=1)
if slc_virt_ref.ndim == 3:
    slc_virt_ref = slc_virt_ref[:, :, 0]

# Combine candidate pixels with virtual reference
slc_all = np.concatenate((slc_cand, slc_virt_ref), axis=1)

ifgs_all, tbase_ifg, pbase_ifg = computeIfgsAndBaselines(
    slc_all=slc_all,
    pbase=pbase,
    tbase=tbase,
    ref_idx=slc_all.shape[0] // 2  # take arbitrary image in the middle of time series
)
# Execute pipeline
if __name__ == "__main__":
    if not os.path.exists(slc_stack_path):
        raise FileNotFoundError(f"File not found: {slc_stack_path}")
    ref_row, ref_col = select_reference_pixel(first_order_mask)# Select first stable scatterer as reference
    print("Computing arc phases...")
    arc_phases = compute_arc_phase(slc_stack, ref_row, ref_col)  # Update function call
    print("smoothing data using butterworth filter")
    smoothed_data = butter_lowpass_filter(arc_phases,cutoff=0.005,fs=0.0667,order=4)
    smoothed_data2 = sav_golay_smooth(arc_phases,window_length=11,polyorder=3)
    first_order_mask = adi <= 0.3
    second_order_mask = (adi > 0.3) & (adi <= 0.4)
    print("Launching Interactive Arc Selection...")
    interactive_plot = InteractiveArcPlot(mean_amp, smoothed_data, first_order_mask, second_order_mask)
    plt.show() 
    used_ifgs = np.any(arc_phases != 0, axis=(1, 2))[:-1]  # Reduce to shape (434,)
    print("Visualizing arc phases...")
    #plot_arc_phase(arc_phases,first_order_mask)
    #plot_arc_phase(arc_phases=smoothed_data, first_order_mask=first_order_mask)
    #plot_arc_phase(arc_phases=smoothed_data2,first_order_mask=first_order_mask)
    #print("Visualizing comparison of Butterworth and Savitzky-Golay filters...")
    #plot_comparison(arc_phases, smoothed_data, smoothed_data2, first_order_mask)

