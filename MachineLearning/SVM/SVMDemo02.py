import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn import svm
from sklearn.datasets import make_blobs


# --- SVM Visualization Application Class ---

class SVMVisualizer:
    # Use numpy to generate 10 steps logarithmically spaced from 10^-2 (0.01) to 10^2 (100)
    C_STEPS = np.logspace(-2, 2, 10).tolist()

    STEP_DESCRIPTIONS = [
        "Step 1 (C={:.4f}): Very Low C. Extremely wide, soft margin. Prioritizes generalization, allowing many slack variables.",
        "Step 2 (C={:.4f}): Low C. Wide margin, accepting some errors to find a better generalized boundary.",
        "Step 3 (C={:.4f}): Gradually increasing the penalty for errors, leading to a tighter fit.",
        "Step 4 (C={:.4f}): The hyperplane starts shifting to reduce margin violations.",
        "Step 5 (C={:.4f}): Moderate C. Approaching the optimal balance between margin width and classification error.",
        "Step 6 (C={:.4f}): Tighter fit to the closest points, reducing the margin slightly.",
        "Step 7 (C={:.4f}): Margin size continues to decrease as the cost of misclassification rises.",
        "Step 8 (C={:.4f}): High C. Margin is getting narrow, severely penalizing slack variables.",
        "Step 9 (C={:.4f}): Very High C. Hard margin behavior is visible. The hyperplane is strictly determined by the few closest points.",
        "Step 10 (C={:.4f}): Max C. Strict, narrow margin. Classifier is highly sensitive, risking overfitting."
    ]
    current_step_index = 0

    def __init__(self, master):
        self.master = master
        master.title("SVM Hyperplane Step-by-Step")
        master.geometry("1100x750")

        # Configure styles
        style = ttk.Style()
        style.configure('Side.TFrame', background='#e1e1e1')
        style.configure('Header.TLabel', font=("Arial", 12, "bold"))

        # --- Data Initialization ---
        self.X = None
        self.y = None
        self.generate_new_data(seed=6)

        # --- Variables ---
        self.C = tk.DoubleVar(value=self.C_STEPS[self.current_step_index])
        self.step_label_text = tk.StringVar()

        # --- Main Layout ---
        main_container = ttk.Frame(master)
        main_container.pack(fill=tk.BOTH, expand=True)

        # 1. Plot Area (Left)
        self.plot_frame = ttk.Frame(main_container, padding=10)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        self.fig.patch.set_facecolor('#f0f0f0')

        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Navigation Toolbar
        self.toolbar_frame = ttk.Frame(self.plot_frame)
        self.toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas_widget, self.toolbar_frame)
        toolbar.update()

        # 2. Control Side Pane (Right)
        self.side_pane = ttk.Frame(main_container, width=300, relief="flat", padding=20)
        self.side_pane.pack(side=tk.RIGHT, fill=tk.Y)
        self.side_pane.pack_propagate(False)  # Keep fixed width

        # Pane Contents
        ttk.Label(self.side_pane, text="Hyperplane Control", style='Header.TLabel').pack(anchor='w', pady=(0, 20))

        # Current Status
        ttk.Label(self.side_pane, textvariable=self.step_label_text, font=("Arial", 10, "bold")).pack(anchor='w')

        # Progress Bar
        self.progress = ttk.Progressbar(self.side_pane, orient=tk.HORIZONTAL, length=260, mode='determinate',
                                        maximum=10)
        self.progress.pack(pady=15)

        # Navigation Buttons
        self.next_button = ttk.Button(self.side_pane, text="Next Step \u279D", command=self.next_step)
        self.next_button.pack(fill=tk.X, pady=5)

        self.reset_button = ttk.Button(self.side_pane, text="\u21BA Reset & New Data", command=self.reset_demo)
        self.reset_button.pack(fill=tk.X, pady=5)

        ttk.Separator(self.side_pane, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)

        # Description Area
        ttk.Label(self.side_pane, text="Statistical Insight:", style='Header.TLabel').pack(anchor='w', pady=(0, 10))
        self.info_label = ttk.Label(self.side_pane, text="", wraplength=260, font=("Arial", 10), justify=tk.LEFT)
        self.info_label.pack(anchor='nw', fill=tk.BOTH, expand=True)

        # --- Initial Update ---
        self.update_visualization()

    def generate_new_data(self, seed=None):
        self.random_seed = seed if seed is not None else np.random.randint(0, 10000)
        self.X, self.y = make_blobs(n_samples=50, centers=2, random_state=self.random_seed, cluster_std=0.8)

    def update_visualization(self):
        C_val = self.C.get()
        total_steps = len(self.C_STEPS)
        current_step = self.current_step_index + 1

        # Update UI Text
        description_template = self.STEP_DESCRIPTIONS[self.current_step_index]
        self.step_label_text.set(f"Step {current_step}/{total_steps} | C = {C_val:.4f}")
        self.info_label.config(text=description_template.format(C_val))
        self.progress['value'] = current_step

        # Train SVM
        clf = svm.SVC(kernel='linear', C=C_val)
        clf.fit(self.X, self.y)

        # Plotting
        self.ax.clear()
        self.ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=30, cmap=plt.cm.RdYlBu, zorder=10)

        # Support Vectors
        self.ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                        s=150, facecolors='none', edgecolors='gold', linewidth=2, label="Support Vectors", zorder=11)

        # Hyperplane logic
        w = clf.coef_[0]
        b = clf.intercept_[0]

        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        xx = np.linspace(x_min, x_max, 100)

        # Calculate decision boundary and margins
        # w0*x + w1*y + b = 0  =>  y = (-w0*x - b) / w1
        yy = (-w[0] * xx - b) / w[1]
        yy_up = (-w[0] * xx - b + 1) / w[1]
        yy_down = (-w[0] * xx - b - 1) / w[1]

        self.ax.plot(xx, yy, 'k-', linewidth=2, label="Hyperplane")
        self.ax.plot(xx, yy_up, 'k--', linewidth=1, alpha=0.6)
        self.ax.plot(xx, yy_down, 'k--', linewidth=1, alpha=0.6)
        self.ax.fill_between(xx, yy_down, yy_up, color='#cccccc', alpha=0.2)

        self.ax.set_title(f"SVM Linear Boundary (C={C_val:.4f})")
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(self.X[:, 1].min() - 1, self.X[:, 1].max() + 1)

        self.canvas_widget.draw()

        # Button Toggle
        self.next_button.config(state=tk.DISABLED if self.current_step_index >= total_steps - 1 else tk.NORMAL)

    def next_step(self):
        if self.current_step_index < len(self.C_STEPS) - 1:
            self.current_step_index += 1
            self.C.set(self.C_STEPS[self.current_step_index])
            self.update_visualization()

    def reset_demo(self):
        self.current_step_index = 0
        self.generate_new_data()
        self.C.set(self.C_STEPS[self.current_step_index])
        self.update_visualization()


if __name__ == "__main__":
    root = tk.Tk()
    app = SVMVisualizer(root)
    root.mainloop()