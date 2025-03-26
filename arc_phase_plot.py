import numpy as np
import matplotlib.pyplot as plt
import os
from miaplpy.objects.slcStack import slcStack
from spatz.change.phase_noise_time import  computeIfgsAndBaselines
from Interactive_plot import InteractiveArcPlot , InteractiveIFGPlot
from scipy.spatial import KDTree
import cmcrameri as cmc
from func import load_slc_stack , compute_adi ,select_points , select_reference_pixel , compute_arc_phase,  butter_lowpass_filter ,sav_golay_smooth
from temporal_noise_estimation import compute_temporal_coherence, estimate_phase_noise,filter_noisy_points
from spatz.change.phase_noise_time import singleCD
from plot import plot_arc_phase, plot_comparison
from func import compute_ifg_coherence_network , compute_ifg_stack
import networkx as nx
from plot import plot_point_network , plot_filtered_arc_phase
from Interactive_plot import InteractiveFilteredArcPlot
# Set the path to your SLC stack file
slc_stack_path  = "/home/vavilla/sarvey/spatz/data/inputs/slcStack.h5"
slc_stack_obj = slcStack(slc_stack_path)  # Initialize the slcStack object
print("slcStack metadata:", slc_stack_obj.__dict__)
geometry_file_path = os.path.join(os.path.dirname(slc_stack_path), "geometryRadar.h5")

# ADI threshold values
ADI_THR_PS = 0.4  # Threshold for first-order PS
ADI_THR_TCS = 0.5   # Threshold for TCS
print("Loading SLC stack...")
slc_stack , tbase, pbase,loc_inc, slant_range = load_slc_stack(slc_stack_path , geometry_file_path)
valid_mask = np.abs(slc_stack) > 0  # Mask valid (nonzero) pixels
slc_stack = slc_stack * valid_mask  # Keep only valid values
print("Computing ADI...")
adi , mean_amp = compute_adi(slc_stack)

# point selection
print("Selecting first-order and TCS points...")
first_order_mask, tcs_mask = select_points(adi, ADI_THR_PS,ADI_THR_TCS)
slc_cand = slc_stack[:, first_order_mask]
mask_p1 = tcs_mask
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
slc_all = np.concatenate((slc_cand, slc_virt_ref), axis=1)


#ifg_stack, tbase_ifg, pbase_ifg = compute_ifg_stack(slc_stack, tbase, pbase, slc_cand, slc_p1,slc_virt_ref)
ifg_stack, tbase_ifg, pbase_ifg = computeIfgsAndBaselines(
    slc_all=slc_all, pbase=pbase, tbase=tbase, ref_idx=slc_all.shape[0] // 2
)
#print("IFG stack shape:", ifg_stack.shape)
searchtree = KDTree(coord_xy_p1)  # PS points
point_network = nx.Graph()
for i, coord in enumerate(coord_xy_p1):
    point_network.add_node(i, pos=tuple(coord))

for i, coord in enumerate(coord_xy_cand):
    _, nearest_ps_idx = searchtree.query(coord, k=1)  # Find the closest PS
    tcs_index = len(coord_xy_p1) + i  # Unique index for TCS
    point_network.add_node(tcs_index, pos=tuple(coord))  # Add TCS node
    point_network.add_edge(tcs_index, nearest_ps_idx)# Extract positions of first-order and TCS points

if __name__ == "__main__":
    ref_row, ref_col = select_reference_pixel(first_order_mask)# Select first stable scatterer as reference
    gamma_map, ifg_stack, valid_ifg_pairs, master_idx, tbase_ifg, pbase_ifg = compute_ifg_coherence_network(
     slc_stack, tbase, pbase)
    arc_phases = compute_arc_phase(ifg_stack, ref_row, ref_col)  # Update function call
    smoothed_data = butter_lowpass_filter(arc_phases,cutoff=0.005,fs=0.0667,order=4)
    smoothed_data2 = sav_golay_smooth(arc_phases,window_length=11,polyorder=3)
    res = smoothed_data - arc_phases
    first_order_mask = adi <= 0.3
    second_order_mask = (adi > 0.3) & (adi <= 0.4)
    # #print("Visualizing arc phases...")
    #plot_arc_phase(arc_phases,first_order_mask)
    #plot_arc_phase(arc_phases=smoothed_data, first_order_mask=first_order_mask)
    #plot_arc_phase(arc_phases=smoothed_data2,first_order_mask=first_order_mask)
    #print("Visualizing comparison of Butterworth and Savitzky-Golay filters...")
    #plot_comparison(arc_phases, smoothed_data, smoothed_data2, first_order_mask)
    # print("Computing IFG Stack, Temporal Coherence, and Network...")
    # gamma_map, ifg_stack, valid_ifg_pairs, master_idx, tbase_ifg, pbase_ifg = compute_ifg_coherence_network(
    #     slc_stack, tbase, pbase
    # )
    interactiv_plot = InteractiveIFGPlot(ifg_stack, valid_ifg_pairs, tbase_ifg, pbase_ifg, master_idx)
    plt.show()

    # print(f"Selected {len(valid_ifg_pairs)} IFGs for phase estimation.")

    # print("Estimating Temporal Coherence...")
    coherence_map = compute_temporal_coherence(smoothed_data)
    # coherence_map_2 = compute_temporal_coherence(smoothed_data2)
    # print("Filtering Noisy Points with Adaptive Thresholds...")
    filtered_first_order_mask, filtered_tcs_mask = filter_noisy_points(smoothed_data, coherence_map, tcs_mask, first_order_mask)
    # print("Launching Interactive Arc Selection...")
    amp_img = abs(slc_stack[10])
    interactive_plot = InteractiveArcPlot(amp_img, res, filtered_first_order_mask, filtered_tcs_mask)
    plt.show()  # Launch the interactive visualization
    #interactive_plot = InteractiveFilteredArcPlot(
    #mean_amplitude=mean_amp,
    #arc_phases=arc_phases,
    #butterworth_phases=smoothed_data,
    #savgol_phases=smoothed_data2,
    #first_order_mask=first_order_mask,
    #second_order_mask=second_order_mask
    #)
    #plt.show()
    # plt.figure(figsize=(8, 6))
    # plt.imshow(coherence_map, cmap="jet", interpolation="nearest")
    # plt.colorbar(label="Coherence")
    # plt.title("Temporal Coherence Map")
    # plt.xlabel("Range (pixels)")
    # plt.ylabel("Azimuth (pixels)")
    # plt.show()

    network_pairs = [(coord_xy_p1[p1], coord_xy_cand[p2 - len(coord_xy_p1)]) for p1, p2 in point_network.edges]
    #plot_point_network(mean_amp, coord_xy_p1, coord_xy_cand, network_pairs)
    #plt.figure(figsize=(8, 6))
    #plt.imshow(mean_amp, cmap="gray", interpolation="nearest")
    #plt.imshow(coherence_map, cmap="jet", alpha=0.6)  # Overlay with transparency
    #plt.colorbar(label="Coherence")
    #plt.title("Coherence Map Overlaid on Amplitude Image")
    #plt.xlabel("Range (pixels)")
    #plt.ylabel("Azimuth (pixels)")
    #plt.show()

    # plt.figure(figsize=(8, 5))
    # plt.hist(coherence_map.flatten(), bins=50, color="blue", alpha=0.7)
    # plt.axvline(x=np.mean(coherence_map), color="red", linestyle="--", label="Mean Coherence")
    # plt.title("Coherence Value Distribution")
    # plt.xlabel("Coherence")
    # plt.ylabel("Frequency")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
