import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn import svm
from sklearn.datasets import make_blobs


# --- 2. SVM Visualization Application Class ---

class SVMVisualizer:
    # Use numpy to generate 10 steps logarithmically spaced from 10^-2 (0.01) to 10^2 (100)
    C_STEPS = np.logspace(-2, 2, 10).tolist()

    STEP_DESCRIPTIONS = [
        "Step 1 (C={:.4f}): Very Low C. Extremely wide, soft margin. Prioritizes generalization, allowing many slack variables (misclassified or in-margin points).",
        "Step 2 (C={:.4f}): Low C. Wide margin, accepting some errors to find a better generalized boundary.",
        "Step 3 (C={:.4f}): Gradually increasing the penalty for errors, leading to a tighter fit.",
        "Step 4 (C={:.4f}): The hyperplane starts shifting to reduce margin violations.",
        "Step 5 (C={:.4f}): Moderate C. Approaching the optimal balance between margin width and classification error.",
        "Step 6 (C={:.4f}): Tighter fit to the closest points, reducing the margin slightly.",
        "Step 7 (C={:.4f}): Margin size continues to decrease as the cost of misclassification rises.",
        "Step 8 (C={:.4f}): High C. Margin is getting narrow, severely penalizing slack variables.",
        "Step 9 (C={:.4f}): Very High C. Hard margin behavior is visible. The hyperplane is strictly determined by the few closest points.",
        "Step 10 (C={:.4f}): Max C. Strict, narrow margin. Classifier is highly sensitive to the current data distribution, risking overfitting."
    ]
    current_step_index = 0

    def __init__(self, master):
        self.master = master
        master.title("Step-by-Step SVM Hyperplane Demo")
        master.geometry("850x750")  # Slightly larger window

        # --- Data Initialization ---
        self.X = None
        self.y = None
        self.generate_new_data(seed=6)  # Initial data generation

        # --- Variables ---
        self.C = tk.DoubleVar(value=self.C_STEPS[self.current_step_index])
        self.step_label_text = tk.StringVar()

        # --- Matplotlib Figure Setup ---
        self.fig, self.ax = plt.subplots(figsize=(7, 6))  # Larger plot size
        self.fig.patch.set_facecolor('#f0f0f0')  # Match tkinter background
        self.ax.set_title("SVM Hyperplane Visualization")
        self.ax.set_xlabel("Feature X1")
        self.ax.set_ylabel("Feature X2")

        # --- Tkinter Canvas Setup ---
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=10, pady=10)

        # --- Toolbar (optional, but useful for zoom/pan) ---
        toolbar = NavigationToolbar2Tk(self.canvas_widget, master)
        toolbar.update()

        # --- Control Panel (Frame) ---
        control_frame = ttk.LabelFrame(master, text="Demonstration Controls", padding="10")
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # --- Step Control ---
        step_control_frame = ttk.Frame(control_frame)
        step_control_frame.grid(row=0, column=0, sticky='w', padx=5, pady=5)

        ttk.Label(step_control_frame, textvariable=self.step_label_text, font=("TkDefaultFont", 10, "bold")).pack(
            side=tk.LEFT, padx=10)

        self.next_button = ttk.Button(step_control_frame, text="Next Step \u279D", command=self.next_step)
        self.next_button.pack(side=tk.LEFT, padx=10)

        self.reset_button = ttk.Button(step_control_frame, text="Reset & New Data", command=self.reset_demo)
        self.reset_button.pack(side=tk.LEFT, padx=10)

        # --- Info Label (Expanded) ---
        self.info_label = ttk.Label(control_frame, text="", wraplength=700, foreground='black')
        self.info_label.grid(row=1, column=0, columnspan=3, pady=5, sticky='w')

        # --- Initial Model Training and Drawing ---
        self.update_visualization()

    def generate_new_data(self, seed=None):
        """Generates new randomized data and stores it in self.X and self.y."""
        # Use a random seed if none is provided for randomization
        self.random_seed = seed if seed is not None else np.random.randint(0, 10000)
        # Generate new data with the chosen seed
        self.X, self.y = make_blobs(n_samples=50, centers=2, random_state=self.random_seed, cluster_std=0.8)

    def update_visualization(self):
        """Trains the SVM model and updates the plot based on the current C parameter."""

        C = self.C.get()
        total_steps = len(self.C_STEPS)

        # Update labels (format C value into description)
        current_step = self.current_step_index + 1
        description_template = self.STEP_DESCRIPTIONS[self.current_step_index]
        formatted_description = description_template.format(C)  # Format C into the description

        self.step_label_text.set(
            f"Step {current_step}/{total_steps} | C = {C:.4f}")  # Show 4 decimals for logarithmic steps
        self.info_label.config(text=formatted_description)

        # --- 3. Train SVM Model ---
        # We use a Linear Kernel to easily visualize the straight line
        clf = svm.SVC(kernel='linear', C=C)
        clf.fit(self.X, self.y)  # Use instance data self.X and self.y

        # --- 4. Draw Plot ---
        self.ax.clear()

        # Scatter plot of the data points
        self.ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=30, cmap=plt.cm.RdYlBu, zorder=10)

        # Highlight Support Vectors
        self.ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                        s=150, facecolors='none', edgecolors='gold', linewidth=2, label="Support Vectors", zorder=11)

        # --- 5. Draw Hyperplane and Margins ---

        # Get the hyperplane equation coefficients: w_0*x + w_1*y + b = 0
        w = clf.coef_[0]
        b = clf.intercept_[0]

        # Function to calculate the y-coordinate for a given x on the plane/margin
        def line(x):
            # Check for near-vertical lines to prevent division by zero
            if np.abs(w[1]) < 1e-6:
                return np.full_like(x, np.inf)
            return (-w[0] * x - b) / w[1]

        # Calculate coordinates for the boundaries
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1

        xx = np.linspace(x_min, x_max, 100)
        yy = line(xx)

        # Hyperplane (Decision Boundary): w*x + b = 0
        self.ax.plot(xx, yy, 'k-', linewidth=3, label="Hyperplane")

        # Margin boundaries: w*x + b = +1 and w*x + b = -1
        # Line for w*x + b = 1 (Upper Margin)
        yy_upper_margin = (-w[0] * xx - b + 1) / w[1]

        # Line for w*x + b = -1 (Lower Margin)
        yy_lower_margin = (-w[0] * xx - b - 1) / w[1]

        self.ax.plot(xx, yy_upper_margin, 'k--', linewidth=1, alpha=0.7, label="Margin Boundaries")
        self.ax.plot(xx, yy_lower_margin, 'k--', linewidth=1, alpha=0.7)

        # Fill the margin area
        self.ax.fill_between(xx, yy_lower_margin, yy_upper_margin, color='#cccccc', alpha=0.3)

        # Final plot settings
        self.ax.set_title(f"SVM Hyperplane (C={C:.4f})")
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(self.X[:, 1].min() - 1, self.X[:, 1].max() + 1)

        # Redraw the canvas
        self.canvas_widget.draw()

        # Disable/Enable Next Step button
        if self.current_step_index >= total_steps - 1:
            self.next_button.config(state=tk.DISABLED)
        else:
            self.next_button.config(state=tk.NORMAL)

    def next_step(self):
        """Advances to the next predefined C value."""
        if self.current_step_index < len(self.C_STEPS) - 1:
            self.current_step_index += 1
            self.C.set(self.C_STEPS[self.current_step_index])
            self.update_visualization()

    def reset_demo(self):
        """Resets the visualization to the first C value and randomizes the data."""
        self.current_step_index = 0
        self.generate_new_data()  # Generate new random data
        self.C.set(self.C_STEPS[self.current_step_index])
        self.update_visualization()


# --- 6. Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = SVMVisualizer(root)
    root.mainloop()