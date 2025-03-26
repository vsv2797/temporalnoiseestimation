import numpy as np
import matplotlib.pyplot as plt

# Visualize Arc Phase Time Series in Subplots
def plot_arc_phase(arc_phases, first_order_mask):
    """Plots arc phase time series for selected first-order points with subplots."""
    num_ifgs, rows, cols = arc_phases.shape
    num_subplots = 10  # Divide IFG index into 4 parts
    ifg_per_subplot = num_ifgs // num_subplots

    plt.figure(figsize=(14, 8))

    selected_indices = np.argwhere(first_order_mask)[:15]  # Pick first 10 valid points
    row_indices, col_indices = selected_indices[:, 0], selected_indices[:, 1]

    for i in range(num_subplots):
        ax = plt.subplot(4, 4, i + 1)  # 2x2 grid
        start_ifg = i * ifg_per_subplot
        end_ifg = start_ifg + ifg_per_subplot

        for row, col in zip(row_indices, col_indices):
            ax.plot(range(start_ifg, end_ifg), arc_phases[start_ifg:end_ifg, row, col], alpha=0.7, label=f'Pixel ({row}, {col})')

        ax.set_xlabel("Interferogram Index")
        ax.set_ylabel("Phase (radians)")
        ax.set_title(f"Arc Phase (IFG {start_ifg} - {end_ifg})")
        # ax.legend(fontsize='small', loc='upper right')

    plt.tight_layout()
    #plt.show()

    selected_indices = np.argwhere(first_order_mask)[:10]  # Pick first 10 valid points
    row_indices, col_indices = selected_indices[:, 0], selected_indices[:, 1]

    for row, col in zip(row_indices, col_indices):
        plt.figure()
        plt.plot(arc_phases[:, row, col], '.')
        plt.ylim([-np.pi*4, np.pi*4])
    plt.show()
def plot_comparison(arc_phases, smoothed_butter, smoothed_savgol, first_order_mask):
    """Plots separate time series for each selected first-order pixel, comparing original, Butterworth, and Savitzky-Golay filters."""
    num_ifgs, rows, cols = arc_phases.shape
    selected_indices = np.argwhere(first_order_mask)[:5]  # Pick first 5 valid points
    row_indices, col_indices = selected_indices[:, 0], selected_indices[:, 1]

    for row, col in zip(row_indices, col_indices):
        plt.figure(figsize=(8, 5))
        plt.plot(arc_phases[:, row, col], '.', markersize=4, alpha=0.7, label="Original")
        plt.plot(smoothed_butter[:, row, col], '.', linewidth=1.5, label="Butterworth Filter")
        plt.plot(smoothed_savgol[:, row, col], '.', linewidth=1.5, label="Savitzky-Golay Filter")

        plt.xlabel("Interferogram Index")
        plt.ylabel("Phase (radians)")
        plt.title(f"Pixel ({row}, {col}) - Arc Phase Comparison")
        plt.ylim([-np.pi, np.pi])  # Set consistent y-limits
        plt.legend(fontsize="small")
        plt.grid(True)
    plt.show()

def plot_point_network(mean_amplitude, coord_xy_p1, coord_xy_cand, network_pairs):
    """
    Plots the amplitude image with overlaid PS (red) and TCS (blue) points, along with arcs between them.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(mean_amplitude, cmap="gray", interpolation="nearest")
    ax.set_title("Point Network Overlaid on Amplitude Image")
    ax.set_xlabel("Range")
    ax.set_ylabel("Azimuth")
    ax.scatter(coord_xy_p1[:, 1], coord_xy_p1[:, 0], color='red', marker='o', s=4, label="First Order (PS)")
    ax.scatter(coord_xy_cand[:, 1], coord_xy_cand[:, 0], color='blue', marker='.', s=2, label="TCS Candidates")
    for p1, p2 in network_pairs:
        ax.plot([p1[1], p2[1]], [p1[0], p2[0]], 'cyan', linestyle='-', alpha=0.7)
    ax.legend()
    plt.show()
def plot_filtered_arc_phase(mean_amplitude, arc_phases, smoothed_phases, first_order_mask, second_order_mask):
    """
    Plots an amplitude image with first-order and second-order points on the left.
    Displays two plots on the right showing Savitzky-Golay and Butterworth filtered arc phases with residuals.
    """
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Left Panel: Amplitude Image with Scatter Points
    ax[0].imshow(mean_amplitude, cmap="gray", interpolation="nearest")
    ax[0].set_title("Amplitude Image with First & Second Order Points")

    # Highlight first-order points
    first_order_coords = np.array(np.where(first_order_mask)).T
    ax[0].scatter(first_order_coords[:, 1], first_order_coords[:, 0],
                   color='red', marker='D', s=2, label="First Order Points")

    # Highlight second-order points
    second_order_coords = np.array(np.where(second_order_mask)).T
    ax[0].scatter(second_order_coords[:, 1], second_order_coords[:, 0],
                   color='blue', marker='.', s=2, label="Second Order Points")
    ax[0].legend()

    # Extract and plot for all selected points
    for row, col in first_order_coords:
        arc_phase_series = arc_phases[:, row, col]
        smoothed_series = smoothed_phases[:, row, col]
        residual_series = arc_phase_series - smoothed_series

        # Plot Savitzky-Golay output
        ax[1].plot(arc_phase_series, '.', markersize=4, alpha=0.7, label=f"Observed Phase ({row},{col})")
        ax[1].plot(smoothed_series, '-', linewidth=1.5, label=f"Smoothed ({row},{col})")
        ax[1].plot(residual_series, '--', linewidth=1, label=f"Residual ({row},{col})")

    ax[1].set_title("Filtered Arc Phase")
    ax[1].set_xlabel("Interferogram Index")
    ax[1].set_ylabel("Phase (radians)")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

