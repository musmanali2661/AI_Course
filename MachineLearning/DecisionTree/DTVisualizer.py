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

    if 'Drug' not in data.columns:
        messagebox.showerror("Data Format Error", "The CSV file must contain a 'Drug' column.")
        return None, None, None, None, None

    X_raw = data.drop('Drug', axis=1)
    y_raw = data['Drug']

    all_cols = X_raw.columns.tolist()
    categorical_cols = [col for col in ['Sex', 'BP', 'Cholesterol'] if col in all_cols]

    X_processed = X_raw.copy()

    le_features = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col])
        le_features[col] = le

    le_y = LabelEncoder()
    y = le_y.fit_transform(y_raw)

    feature_names = X_processed.columns.tolist()
    class_names = le_y.classes_.tolist()

    return X_processed.values, y, feature_names, class_names, le_features


# --- 2. Decision Tree Visualization Application Class ---

class DecisionTreeVisualizer:
    MAX_DEPTH_STEPS = list(range(1, 11))
    current_step_index = 0

    def __init__(self, master):
        self.master = master
        master.title("Decision Tree Step-by-Step Growth Demo")
        master.geometry("1200x800")

        # Configure the main window style
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TButton', padding=6)
        style.configure('Header.TLabel', font=("Arial", 12, "bold"), background='#f0f0f0')

        # --- Data Initialization ---
        self.X, self.y, self.feature_names, self.class_names, self.le_features = load_and_preprocess_data()

        if self.X is None:
            master.destroy()
            return

        # --- Variables ---
        self.current_depth = tk.IntVar(value=self.MAX_DEPTH_STEPS[self.current_step_index])
        self.step_label_text = tk.StringVar()

        # --- Main Layout Container ---
        # Using a horizontal layout: [ Tree View (Left) | Controls (Right) ]
        container = ttk.Frame(master)
        container.pack(fill=tk.BOTH, expand=True)

        # Left Side: Tree Plot
        self.plot_frame = ttk.Frame(container, padding="10")
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(8, 7))
        self.fig.patch.set_facecolor('#f0f0f0')
        self.ax.axis('off')

        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add Matplotlib Toolbar below plot
        self.toolbar_frame = ttk.Frame(self.plot_frame)
        self.toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas_widget, self.toolbar_frame)
        toolbar.update()

        # Right Side: Control Pane
        self.side_pane = ttk.Frame(container, width=300, padding="20", relief="sunken")
        self.side_pane.pack(side=tk.RIGHT, fill=tk.Y)
        self.side_pane.pack_propagate(False)  # Maintain fixed width

        ttk.Label(self.side_pane, text="Navigation", style='Header.TLabel').pack(anchor='w', pady=(0, 10))

        # Step Indicator
        ttk.Label(self.side_pane, textvariable=self.step_label_text, font=("Arial", 10, "bold")).pack(anchor='w',
                                                                                                      pady=5)

        # Progress bar for visual depth feedback
        self.progress = ttk.Progressbar(self.side_pane, orient=tk.HORIZONTAL, length=260, mode='determinate',
                                        maximum=10)
        self.progress.pack(pady=10)

        # Buttons
        btn_frame = ttk.Frame(self.side_pane)
        btn_frame.pack(fill=tk.X, pady=10)

        self.next_button = ttk.Button(btn_frame, text="Next Step \u279D", command=self.next_step)
        self.next_button.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.reset_button = ttk.Button(btn_frame, text="\u21BA Reset Tree", command=self.reset_demo)
        self.reset_button.pack(side=tk.TOP, fill=tk.X, pady=5)

        ttk.Separator(self.side_pane, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)

        ttk.Label(self.side_pane, text="Insight", style='Header.TLabel').pack(anchor='w', pady=(0, 10))

        # Description
        self.info_label = ttk.Label(self.side_pane, text="", wraplength=260, foreground='#333333', font=("Arial", 10))
        self.info_label.pack(anchor='w', fill=tk.BOTH, expand=True)

        # --- Initial Drawing ---
        self.update_visualization()

    def get_description(self, depth):
        if depth == 1:
            return "The algorithm selects the single best feature (Na_to_K) to create a 'stump'. This split provides the highest Information Gain."
        elif depth <= 3:
            return f"The tree begins to branch into secondary levels. It is now looking at factors like 'BP' or 'Age' to further refine its drug classifications."
        elif depth <= 5:
            return f"Depth {depth}: Many nodes are now 'pure' (containing only one class of drug). The logic is becoming more sophisticated."
        else:
            return f"At Depth {depth}, the tree is extremely specific. While accurate for this data, it may struggle with new, unseen patients (overfitting)."

    def update_visualization(self):
        depth = self.current_depth.get()
        total_steps = len(self.MAX_DEPTH_STEPS)
        current_step = self.current_step_index + 1

        self.step_label_text.set(f"Step {current_step} of {total_steps} (Depth: {depth})")
        self.info_label.config(text=self.get_description(depth))
        self.progress['value'] = current_step

        # Train Model
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42, criterion='entropy')
        clf.fit(self.X, self.y)

        # Clear and Draw
        self.ax.clear()
        self.ax.axis('off')

        plot_tree(
            clf,
            ax=self.ax,
            feature_names=self.feature_names,
            class_names=self.class_names,
            filled=True,
            rounded=True,
            precision=2,
            proportion=False,
            fontsize=8,
            max_depth=depth
        )

        self.ax.set_title(f"Decision Tree Path (Max Depth = {depth})", fontsize=12, pad=20)
        self.fig.tight_layout()
        self.canvas_widget.draw()

        # Button State
        if self.current_step_index >= total_steps - 1:
            self.next_button.config(state=tk.DISABLED)
        else:
            self.next_button.config(state=tk.NORMAL)

    def next_step(self):
        if self.current_step_index < len(self.MAX_DEPTH_STEPS) - 1:
            self.current_step_index += 1
            self.current_depth.set(self.MAX_DEPTH_STEPS[self.current_step_index])
            self.update_visualization()

    def reset_demo(self):
        self.current_step_index = 0
        self.current_depth.set(self.MAX_DEPTH_STEPS[self.current_step_index])
        self.update_visualization()


if __name__ == "__main__":
    root = tk.Tk()
    app = DecisionTreeVisualizer(root)
    root.mainloop()