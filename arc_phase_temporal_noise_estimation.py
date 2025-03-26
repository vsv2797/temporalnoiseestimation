
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from miaplpy.objects.slcStack import slcStack
from func import load_slc_stack, compute_adi, select_points, compute_virtual_reference
from spatz.change.phase_noise_time import singleCD
import os

# Load SLC stack and metadata
slc_stack_path = "/home/vavilla/sarvey/spatz/data/inputs/slcStack.h5"
geometry_file_path = os.path.join(os.path.dirname(slc_stack_path), "geometryRadar.h5")
slc_stack, tbase, pbase, loc_inc, slant_range = load_slc_stack(slc_stack_path, geometry_file_path)

# ADI thresholds
ADI_THR_PS = 0.3
ADI_THR_TCS = 0.4

# Select points
adi, _ = compute_adi(slc_stack)
first_order_mask, tcs_mask = select_points(adi, ADI_THR_PS, ADI_THR_TCS)

# Compute virtual references
slc_cand, slc_virt_ref = compute_virtual_reference(slc_stack, first_order_mask, tcs_mask, num_nearest_neighbors=5)

# Filter geometry data
loc_inc_cand = loc_inc[first_order_mask]
slant_range_cand = slant_range[first_order_mask]

# Loop through points and compute temporal noise
results = []
for idx in range(slc_cand.shape[1]):
    score_bw, score_sg, max_coh_bw, max_coh_sg, coh_init_bw, coh_init_sg = singleCD(
        slc_cand=slc_cand[:, idx],
        slc_virt_ref=slc_virt_ref[:, idx, np.newaxis],
        tbase=tbase,
        pbase=pbase,
        loc_inc=float(loc_inc_cand[idx]),
        slant_range=float(slant_range_cand[idx]),
        wavelength=0.055,
        demerr_bound=20,
        velocity_bound=5,
        num_samples=100,
        show_plots=False
    )
    change_idx = np.nanargmin(score_bw)
    results.append((idx, change_idx, max_coh_bw, coh_init_bw))

# Convert results to maps
change_map = np.full(slc_stack.shape[1:], np.nan)
coherence_map = np.full(slc_stack.shape[1:], np.nan)
coords = np.argwhere(first_order_mask)
for (i, change_idx, max_coh, _) in results:
    row, col = coords[i]
    change_map[row, col] = change_idx
    coherence_map[row, col] = max_coh

# Save maps as .npy
np.save("change_map.npy", change_map)
np.save("coherence_map.npy", coherence_map)
print("Saved change and coherence maps.")
