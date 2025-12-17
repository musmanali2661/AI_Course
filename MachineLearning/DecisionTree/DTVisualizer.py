import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import os


# --- 1. Data Processing and Preparation ---

def load_and_preprocess_data(file_path='drug200.csv'):
    """
    Loads CSV data from the specified file path and preprocesses
    categorical features using LabelEncoder for Decision Tree compatibility.
    """
    try:
        # Read the data directly from the file path
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        messagebox.showerror("File Error",
                             f"The file '{file_path}' was not found. Please ensure it is in the same directory.")
        return None, None, None, None, None
    except Exception as e:
        messagebox.showerror("Data Error", f"An error occurred while loading data: {e}")
        return None, None, None, None, None

    # Assuming the column names are consistent: Age,Sex,BP,Cholesterol,Na_to_K,Drug
    if 'Drug' not in data.columns:
        messagebox.showerror("Data Format Error", "The CSV file must contain a 'Drug' column as the target variable.")
        return None, None, None, None, None

    # Separate features (X) and target (y)
    X_raw = data.drop('Drug', axis=1)
    y_raw = data['Drug']

    # Identify categorical columns (Sex, BP, Cholesterol)
    # Filter for columns that actually exist in the file
    all_cols = X_raw.columns.tolist()
    categorical_cols = [col for col in ['Sex', 'BP', 'Cholesterol'] if col in all_cols]

    X_processed = X_raw.copy()

    # Use LabelEncoder to convert categorical features to integers
    le_features = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col])
        le_features[col] = le

    # Convert target (Drug) to numerical
    le_y = LabelEncoder()
    y = le_y.fit_transform(y_raw)

    # Prepare feature and class names for the plot
    feature_names = X_processed.columns.tolist()
    class_names = le_y.classes_.tolist()

    return X_processed.values, y, feature_names, class_names, le_features


# --- 2. Decision Tree Visualization Application Class ---

class DecisionTreeVisualizer:
    # Max Depth steps for demonstration
    # We will step through depths 1 to 10
    MAX_DEPTH_STEPS = list(range(1, 11))
    current_step_index = 0

    def __init__(self, master):
        self.master = master
        master.title("Decision Tree Step-by-Step Growth Demo (drug200.csv)")

        # Configure the main window style
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TButton', padding=6, relief="flat", background='#007ACC', foreground='white')
        style.map('TButton', background=[('active', '#005f99')])

        # --- Data Initialization ---
        self.X, self.y, self.feature_names, self.class_names, self.le_features = load_and_preprocess_data()

        if self.X is None:
            # If data loading failed, stop initialization
            master.destroy()
            return

        # --- Variables ---
        self.current_depth = tk.IntVar(value=self.MAX_DEPTH_STEPS[self.current_step_index])
        self.step_label_text = tk.StringVar()

        # --- Matplotlib Figure Setup ---
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.fig.patch.set_facecolor('#f0f0f0')
        self.ax.set_title("Decision Tree Structure")
        self.ax.axis('off')

        # --- Tkinter Canvas Setup ---
        main_frame = ttk.Frame(master, padding="10 10 10 10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # --- Control Panel (Frame) ---
        control_frame = ttk.LabelFrame(main_frame, text="Max Depth Control", padding="15")
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        # --- Step Control ---
        step_control_frame = ttk.Frame(control_frame)
        step_control_frame.pack(fill=tk.X)

        ttk.Label(step_control_frame, textvariable=self.step_label_text, font=("Arial", 10, "bold")).pack(side=tk.LEFT,
                                                                                                          padx=10)

        self.next_button = ttk.Button(step_control_frame, text="Next Split \u279D", command=self.next_step)
        self.next_button.pack(side=tk.RIGHT, padx=5)

        self.reset_button = ttk.Button(step_control_frame, text="\u21BA Reset", command=self.reset_demo)
        self.reset_button.pack(side=tk.RIGHT, padx=5)

        # --- Info Label (Expanded) ---
        self.info_label = ttk.Label(control_frame, text="", wraplength=900, foreground='#333333', font=("Arial", 9))
        self.info_label.pack(fill=tk.X, pady=(10, 5))

        # --- Initial Model Training and Drawing ---
        self.update_visualization()

    def get_description(self, depth):
        """Returns a description based on the current tree depth."""
        if depth == 1:
            return "Step 1: The model finds the single best feature split (e.g., Na_to_K > 14.8) to separate the data into the purest possible groups (nodes). This is the simplest model."
        elif depth <= 3:
            return f"Step {depth}: The tree adds a few more splits (nodes) on secondary features (like BP or Age) to better isolate certain drug types. The model is still very interpretable."
        elif depth <= 5:
            return f"Step {depth}: The tree is growing, creating more detailed branches. Notice how the 'value' array in the leaf nodes is becoming more concentrated towards one drug, indicating high purity."
        else:
            return f"Step {depth}: The tree is highly complex, fitting the training data very closely. Every split is used to classify the individual data points. This high detail increases the risk of overfitting."

    def update_visualization(self):
        """Trains the Decision Tree model and updates the plot based on the current max_depth."""

        depth = self.current_depth.get()
        total_steps = len(self.MAX_DEPTH_STEPS)
        current_step = self.current_step_index + 1

        # --- Update Labels ---
        self.step_label_text.set(f"Step {current_step}/{total_steps} | Max Depth = {depth}")
        self.info_label.config(text=self.get_description(depth))

        # --- 3. Train Decision Tree Model ---
        # The best model for this data is often found around depth 5-6
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42, criterion='entropy')
        clf.fit(self.X, self.y)

        # --- 4. Draw Plot ---
        self.ax.clear()
        self.ax.axis('off')

        # Plot the tree structure
        plot_tree(
            clf,
            ax=self.ax,
            feature_names=self.feature_names,
            class_names=self.class_names,
            filled=True,
            rounded=True,
            precision=2,  # Show split values with 2 decimal places
            proportion=False,
            fontsize=8,  # Changed from 7.5 to 8 to fix the InvalidParameterError
            max_depth=depth  # Plot only up to the current depth
        )

        # Ensure plot fits the canvas
        self.fig.tight_layout()

        # Final plot settings
        self.ax.set_title(f"Decision Tree Growth (Max Depth = {depth})", fontsize=12)

        # Redraw the canvas
        self.canvas_widget.draw()

        # Disable/Enable Next Step button
        if self.current_step_index >= total_steps - 1:
            self.next_button.config(state=tk.DISABLED)
        else:
            self.next_button.config(state=tk.NORMAL)

    def next_step(self):
        """Advances to the next predefined Max Depth value."""
        if self.current_step_index < len(self.MAX_DEPTH_STEPS) - 1:
            self.current_step_index += 1
            self.current_depth.set(self.MAX_DEPTH_STEPS[self.current_step_index])
            self.update_visualization()

    def reset_demo(self):
        """Re sets the visualization to the first step (Depth 1)."""
        self.current_step_index = 0
        self.current_depth.set(self.MAX_DEPTH_STEPS[self.current_step_index])
        self.update_visualization()


# --- 5. Main Execution ---
if __name__ == "__main__":
    # Tkinter requires the root window to be created first
    root = tk.Tk()
    app = DecisionTreeVisualizer(root)
    # Start the Tkinter event loop
    root.mainloop()