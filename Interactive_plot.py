import numpy as np
import matplotlib.pyplot as plt

class InteractiveArcPlot:
    """
    Interactive tool to manually draw arcs between two points on the amplitude image
    and visualize the corresponding arc phase time series.
    """
    def __init__(self, mean_amplitude, arc_phases, first_order_mask, tcs_mask):
        self.mean_amplitude = mean_amplitude
        self.arc_phases = arc_phases  # Shape: (num_ifgs, height, width)
        self.first_order_mask = first_order_mask
        self.tcs_mask = tcs_mask  # Ensure TCS points are included

        self.fig, self.ax = plt.subplots(1, 2, figsize=(16, 8))
        self.selected_points = []  # Stores clicked points (row, col)

        self.plot_amplitude_image()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def plot_amplitude_image(self):
        """Plots the amplitude image and overlays first-order and TCS points."""
        self.ax[0].imshow(self.mean_amplitude, cmap="cmc.grayC_r", interpolation="nearest")
        self.ax[0].set_title(" Amplitude image with first order and second order points")
        self.ax[0].set_xlabel("Range")
        self.ax[0].set_ylabel("Azimuth")

        # Highlight first-order points
        first_order_coords = np.array(np.where(self.first_order_mask)).T
        self.ax[0].scatter(first_order_coords[:, 1], first_order_coords[:, 0],
                            color='red', marker='D', s=2, label="First Order Points")

        # Highlight TCS points (Fixed: Ensure they appear)
        tcs_coords = np.array(np.where(self.tcs_mask)).T
        self.ax[0].scatter(tcs_coords[:, 1], tcs_coords[:, 0],
                            color='blue', marker='.', s=2, label="TCS Points")

        self.ax[0].legend()

    def on_click(self, event):
        """Handles click events to select two points, restricted to first-order and second-order points."""
        if event.inaxes != self.ax[0]:  # Ensure clicks are within the amplitude image
            return

        row, col = int(event.ydata), int(event.xdata)

        # Check if the selected point is in the first-order or second-order mask
        if not (self.first_order_mask[row, col] or self.tcs_mask[row, col]):
            print(f"⚠️ Invalid selection at ({row}, {col}). Only select first-order or second-order points.")
            return  # Ignore the selection if it's not in first-order or second-order points

        if len(self.selected_points) == 2:  # Reset points if a new selection starts
            self.selected_points = []
            self.ax[0].clear()  # Clear previous selections
            self.plot_amplitude_image()  # Redraw the amplitude image with first-order and TCS points

        self.selected_points.append((row, col))

        # Mark selected point
        self.ax[0].scatter(col, row, color='cyan', marker='.', s=40)
        self.fig.canvas.draw()

        # If two points are selected, draw the arc and plot phase
        if len(self.selected_points) == 2:
            self.plot_arc()
            self.plot_arc_phase()

    def plot_arc(self):
        """Draws a line (arc) between the two selected points."""
        (row1, col1), (row2, col2) = self.selected_points
        self.ax[0].plot([col1, col2], [row1, row2], color='gray', linewidth=2, label="Arc")
        self.fig.canvas.draw()

    def plot_arc_phase(self):
        """Extracts and plots the arc phase time series between the two selected points."""
        (row1, col1), (row2, col2) = self.selected_points

        # Extract arc phase time series between the selected points
        arc_phase_series = self.arc_phases[:, row1, col1] - self.arc_phases[:, row2, col2]

        self.ax[1].clear()
        self.ax[1].scatter(np.arange(len(arc_phase_series)), arc_phase_series,
                        color='blue', marker='o', s=10)  # Use scatter plot (points only)
        self.ax[1].set_title(f"Arc Phase Between ({row1},{col1}) and ({row2},{col2})")
        self.ax[1].set_xlabel("Time (Interferograms)")
        self.ax[1].set_ylabel("Phase (radians)")
        self.ax[1].grid(True)

        self.fig.canvas.draw()

class InteractiveFilteredArcPlot:
    """
    Interactive tool to manually draw arcs between two points on the amplitude image
    and visualize the corresponding arc phase time series with precomputed Butterworth and Savitzky-Golay filters.
    """
    def __init__(self, mean_amplitude, arc_phases, butterworth_phases, savgol_phases, first_order_mask, second_order_mask):
        self.mean_amplitude = mean_amplitude
        self.arc_phases = arc_phases
        self.butterworth_phases = butterworth_phases
        self.savgol_phases = savgol_phases
        self.first_order_mask = first_order_mask
        self.second_order_mask = second_order_mask

        self.selected_points = []  # Stores clicked points (row, col)

        self.fig, self.ax = plt.subplots(2, 2, figsize=(18, 10))
        self.ax = self.ax.flatten()  # Flatten to easily index subplots
        self.plot_amplitude_image()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def plot_amplitude_image(self):
        """Plots the amplitude image and overlays first-order and second-order points."""
        self.ax[0].imshow(self.mean_amplitude, cmap="gray", interpolation="nearest")
        self.ax[0].set_title("Amplitude Image with First & Second Order Points")
        self.ax[0].set_xlabel("Range")
        self.ax[0].set_ylabel("Azimuth")

        first_order_coords = np.array(np.where(self.first_order_mask)).T
        self.ax[0].scatter(first_order_coords[:, 1], first_order_coords[:, 0],
                            color='red', marker='D', s=2, label="First Order Points")

        second_order_coords = np.array(np.where(self.second_order_mask)).T
        self.ax[0].scatter(second_order_coords[:, 1], second_order_coords[:, 0],
                            color='blue', marker='.', s=2, label="Second Order Points")
        self.ax[0].legend()

    def on_click(self, event):
        """Handles click events to select two points for arc phase visualization."""
        if event.inaxes != self.ax[0]:  # Ensure clicks are within the amplitude image
            return

        row, col = int(event.ydata), int(event.xdata)

        if not (self.first_order_mask[row, col] or self.second_order_mask[row, col]):
            print(f"⚠️ Invalid selection at ({row}, {col}). Only select first-order or second-order points.")
            return

        if len(self.selected_points) == 2:  # Reset selection if a new arc starts
            self.selected_points = []
            self.ax[0].clear()  # Clear previous selections
            self.plot_amplitude_image()  # Redraw the amplitude image

        self.selected_points.append((row, col))
        self.ax[0].scatter(col, row, color='cyan', marker='o', s=40)
        self.fig.canvas.draw()

        if len(self.selected_points) == 2:
            self.plot_arc()
            self.plot_filtered_arc_phase()

    def plot_arc(self):
        """Draws a line (arc) between the selected points."""
        (row1, col1), (row2, col2) = self.selected_points
        self.ax[0].plot([col1, col2], [row1, row2], color='gray', linewidth=2, label="Arc")
        self.fig.canvas.draw()

    def plot_filtered_arc_phase(self):
        """Plots the arc phase time series with precomputed Butterworth and Savitzky-Golay filters."""
        (row1, col1), (row2, col2) = self.selected_points
        arc_phase_series = self.arc_phases[:, row1, col1] - self.arc_phases[:, row2, col2]
        butterworth_series = self.butterworth_phases[:, row1, col1] - self.butterworth_phases[:, row2, col2]
        savgol_series = self.savgol_phases[:, row1, col1] - self.savgol_phases[:, row2, col2]

        # Compute residuals
        sg_residuals = arc_phase_series - savgol_series
        butter_residuals = arc_phase_series - butterworth_series

        # Savitzky-Golay Filtered Phase Plot
        self.ax[1].clear()
        self.ax[1].plot(arc_phase_series, '.', markersize=4, alpha=0.7, label="Observed Phase")
        self.ax[1].plot(savgol_series, '.', label="Savitzky-Golay Filtered")
        self.ax[1].set_title("Savitzky-Golay Filtered Phase")
        self.ax[1].set_xlabel("Interferogram Index")
        self.ax[1].set_ylabel("Phase (radians)")
        self.ax[1].legend()
        self.ax[1].grid(True)

        # Butterworth Filtered Phase Plot
        self.ax[2].clear()
        self.ax[2].plot(arc_phase_series, '.', markersize=4, alpha=0.7, label="Observed Phase")
        self.ax[2].plot(butterworth_series, '.', label="Butterworth Filtered")
        self.ax[2].set_title("Butterworth Filtered Phase")
        self.ax[2].set_xlabel("Interferogram Index")
        self.ax[2].set_ylabel("Phase (radians)")
        self.ax[2].legend()
        self.ax[2].grid(True)

        # Residuals Plot (Savitzky-Golay vs Butterworth)
        self.ax[3].clear()
        self.ax[3].plot(sg_residuals, '.', linewidth=1, alpha=0.7, label="Residuals (SG)")
        self.ax[3].plot(butter_residuals, '.', linewidth=1, alpha=0.7, label="Residuals (Butterworth)")
        self.ax[3].set_title("Residuals Comparison")
        self.ax[3].set_xlabel("Interferogram Index")
        self.ax[3].set_ylabel("Residual Phase (radians)")
        self.ax[3].legend()
        self.ax[3].grid(True)

        self.fig.canvas.draw()
class InteractiveIFGPlot:
    """
    Interactive visualization of the IFG Star Network.
    Clicking on a point in the network will display the corresponding IFG.
    """
    def __init__(self, ifg_stack, valid_ifg_pairs, tbase_ifg, pbase_ifg, master_idx):
        self.ifg_stack = ifg_stack  # Interferometric phase stack (num_ifgs, height, width)
        self.valid_ifg_pairs = valid_ifg_pairs  # List of (master, slave) IFGs
        self.tbase_ifg = tbase_ifg  # Temporal baselines
        self.pbase_ifg = pbase_ifg  # Perpendicular baselines
        self.master_idx = master_idx  # Master image index
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 6))  # Two subplots

        self.plot_ifg_network()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)  # Enable clicks

    def plot_ifg_network(self):
        """Plots the IFG Star Network."""
        self.ax[0].clear()
        ifg_indices = np.arange(len(self.tbase_ifg))  # X-axis as IFG indices

        # Plot all IFGs in the network
        for idx, (master, slave) in enumerate(self.valid_ifg_pairs):
            self.ax[0].plot([ifg_indices[master], ifg_indices[slave]],
                             [self.pbase_ifg[master], self.pbase_ifg[slave]],
                             'bo-', alpha=0.5, markersize=5)
        # Highlight the Master SLC
        self.ax[0].scatter(ifg_indices[self.master_idx], self.pbase_ifg[self.master_idx],
                           color='red', marker='*', s=150, label="Master SLC")
        self.ax[0].set_xlabel("IFG Index")
        self.ax[0].set_ylabel("Perpendicular Baseline (Meters)")
        self.ax[0].set_title("IFG Star Network")
        self.ax[0].grid(True)
        self.ax[0].legend()

    def on_click(self, event):
        """Handles click events to select an IFG from the network and display its image."""
        if event.inaxes != self.ax[0]:  # Ensure click is in the IFG network plot
            return
        ifg_indices = np.arange(len(self.valid_ifg_pairs))  # IFG indices for selection
        x_clicked = int(round(event.xdata))
        # Ensure valid IFG selection
        if x_clicked < 0 or x_clicked >= len(ifg_indices):
            print("Invalid IFG selection!")
            return
        selected_ifg_idx = ifg_indices[x_clicked]  # Map to valid IFG index
        print(f"Selected IFG Index: {selected_ifg_idx}")
        self.ax[1].cla()  # Clear IFG panel
        self.fig.delaxes(self.ax[1])  # Remove the subplot entirely
        self.ax[1] = self.fig.add_subplot(122)  # Recreate the subplot
        self.plot_selected_ifg(selected_ifg_idx)

    def plot_selected_ifg(self,ifg_idx):
        """Plots the phase of the selected IFG while ensuring only one colorbar is displayed."""
        self.ax[1].clear()  # Clear the previous plot
        phase_ifg = np.angle(self.ifg_stack[ifg_idx])  # Convert complex to phase
        im = self.ax[1].imshow(phase_ifg, cmap="jet", aspect="auto")
        self.ax[1].set_title(f"Selected IFG (Index {ifg_idx} )")
        self.ax[1].set_xlabel("Range")
        self.ax[1].set_ylabel("Azimuth")
        # Remove the previous colorbar properly before adding a new one
        # if hasattr(self, 'cbar') and self.cbar is not None:
        #     try:
        #         self.cbar.remove()
        #     except Exception as e:
        #         print(f"Warning: Colorbar removal failed: {e}")
        #    self.cbar = None  # Reset the colorbar reference
        # Create a new colorbar and keep a reference
        self.cbar = self.fig.colorbar(im, ax=self.ax[1], orientation="vertical", label="Phase (radians)")
        self.fig.canvas.draw()






