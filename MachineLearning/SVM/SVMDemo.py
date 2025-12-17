import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn import svm
from sklearn.datasets import make_blobs

# --- 1. Data Generation ---
# Generate a simple, slightly noisy linearly separable 2D dataset
X, y = make_blobs(n_samples=50, centers=2, random_state=6, cluster_std=0.8)


# --- 2. SVM Visualization Application Class ---

class SVMVisualizer:
    def __init__(self, master):
        self.master = master
        master.title("Interactive Support Vector Machine (SVM) Demo")
        master.geometry("800x700")

        # --- Variables ---
        self.C = tk.DoubleVar(value=1.0)

        # --- Matplotlib Figure Setup ---
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
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
        control_frame = ttk.LabelFrame(master, text="SVM Controls", padding="10")
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # --- C Parameter Slider ---
        ttk.Label(control_frame, text="Regularization Parameter C:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.c_slider = ttk.Scale(control_frame, from_=0.1, to=100, orient=tk.HORIZONTAL,
                                  variable=self.C, command=self.update_svm, length=400)
        self.c_slider.grid(row=0, column=1, padx=5, pady=5)
        self.c_value_label = ttk.Label(control_frame, textvariable=self.C)
        self.c_value_label.grid(row=0, column=2, padx=5, pady=5)

        # --- Info Label ---
        self.info_label = ttk.Label(control_frame, text="Adjust C to change the margin strictness.", foreground='blue')
        self.info_label.grid(row=1, column=0, columnspan=3, pady=5)

        # --- Initial Model Training and Drawing ---
        self.update_svm(self.C.get())

    def update_svm(self, c_value):
        """Trains the SVM model and updates the plot based on the C parameter."""

        # Convert c_value from string (Tkinter Scale command sends string) to float
        C = float(c_value)
        self.C.set(C)

        # Update info label based on C value
        if C > 50:
            info_text = "C is high (Hard Margin): Narrow margin, strict separation."
        elif C < 1:
            info_text = "C is low (Soft Margin): Wide margin, allows more misclassification."
        else:
            info_text = "C is medium (Soft Margin): Optimal trade-off between margin size and error."
        self.info_label.config(text=info_text, foreground='green' if C < 10 else 'red')

        # --- 3. Train SVM Model ---
        # We use a Linear Kernel to easily visualize the straight line
        clf = svm.SVC(kernel='linear', C=C)
        clf.fit(X, y)

        # --- 4. Draw Plot ---
        self.ax.clear()

        # Scatter plot of the data points
        self.ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.RdYlBu, zorder=10)

        # Highlight Support Vectors
        self.ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                        s=150, facecolors='none', edgecolors='gold', linewidth=2, label="Support Vectors", zorder=11)

        # --- 5. Draw Hyperplane and Margins ---

        # Get the hyperplane equation coefficients: w_0*x + w_1*y + b = 0
        w = clf.coef_[0]
        b = clf.intercept_[0]

        # Function to calculate the y-coordinate for a given x on the plane/margin
        def line(x):
            return (-w[0] * x - b) / w[1]

        # Calculate coordinates for the boundaries
        x_min, x_max = self.ax.get_xlim()
        if x_min == 0.0 and x_max == 1.0:  # Initial plot limits check
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

        xx = np.linspace(x_min, x_max)
        yy = line(xx)

        # Hyperplane (Decision Boundary): w*x + b = 0
        self.ax.plot(xx, yy, 'k-', linewidth=3, label="Hyperplane")

        # Margin boundaries: w*x + b = +1 and w*x + b = -1
        margin = 1 / np.sqrt(np.dot(w, w))
        yy_up = line(xx) + margin / np.cos(np.arctan(-w[0] / w[1]))
        yy_down = line(xx) - margin / np.cos(np.arctan(-w[0] / w[1]))

        self.ax.plot(xx, yy_up, 'k--', linewidth=1, alpha=0.7, label="Margin Boundaries")
        self.ax.plot(xx, yy_down, 'k--', linewidth=1, alpha=0.7)

        # Fill the margin area
        self.ax.fill_between(xx, yy_down, yy_up, color='#cccccc', alpha=0.3)

        # Final plot settings
        self.ax.set_title(f"SVM Hyperplane (C={C:.2f})")
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

        # Redraw the canvas
        self.canvas_widget.draw()


# --- 6. Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = SVMVisualizer(root)
    root.mainloop()
