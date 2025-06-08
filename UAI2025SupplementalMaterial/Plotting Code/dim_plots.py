import os
import re  # For extracting dimensionality
import numpy as np
import matplotlib.pyplot as plt

# Optionally, use seaborn for improved aesthetics.
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)
except ImportError:
    pass

def load_experiment_scores(root_dir, linear_dirs, method_names, metric):
    """
    Loads scores for a given metric from the specified linear proportion folders.
    Returns a dictionary mapping each method to a list of scores.
    """
    score_data = {m: [] for m in method_names}
    for lin_dir in linear_dirs:
        lin_path = os.path.join(root_dir, lin_dir)
        for m in method_names:
            npy_path = os.path.join(lin_path, f"{m}_{metric}.npy")
            if os.path.isfile(npy_path):
                data = np.load(npy_path)
                # Use median (or change to np.mean if desired)
                score_data[m].append(float(np.median(data)))
            else:
                print(f"Warning: No file found for {m} in {lin_dir}. Skipping...")
                score_data[m].append(np.nan)
    return score_data

def plot_aggregated_metrics(root_dir, datasets, metrics, linear_folder, method_list):
    """
    Creates one figure with three subplots (one per metric).
    
    For each dataset, the function extracts the dimensionality (e.g., 10, 20, 30)
    from the dataset path (searching for a pattern like 'd10'). It then loads the score
    for each method (from the specified linear_folder, e.g., "linear_proportion_0.5")
    and aggregates these scores across datasets.
    
    In each subplot:
      - x-axis: Dimensionality
      - y-axis: The metric value
      - Each line corresponds to one method.
    
    Parameters:
      - root_dir: Base directory containing dataset directories.
      - datasets: Dictionary mapping dataset labels (should be unique) to their paths.
      - metrics: List of metric names (e.g., ["Atop", "F1", "SHD"]).
      - linear_folder: A single linear proportion folder (e.g., "linear_proportion_0.5").
      - method_list: List of method names to plot.
    """
    def extract_dimension(path):
        # Extracts a dimension value from the dataset path (e.g., finds "d10" and returns 10)
        match = re.search(r'd(\d+)', path)
        return int(match.group(1)) if match else None

    # Initialize a dictionary to hold, for each metric, for each method, a list of (dimension, score) pairs.
    metric_method_points = {metric: {m: [] for m in method_list} for metric in metrics}

    # Loop through all datasets to accumulate (dimension, score) data.
    for dataset_label, dataset_path in datasets.items():
        dim = extract_dimension(dataset_path)
        if dim is None:
            print(f"Warning: Could not extract dimension from {dataset_path}. Skipping dataset.")
            continue
        for metric in metrics:
            # Load scores from the specified linear folder.
            score_data = load_experiment_scores(dataset_path, [linear_folder], method_list, metric)
            for m in method_list:
                # Since there is only one folder in linear_dirs, take the first score.
                score = score_data[m][0] if score_data[m] else np.nan
                metric_method_points[metric][m].append((dim, score))
    
    # Create one figure with one subplot per metric.
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
    
    # Loop over metrics to populate each subplot.
        # Loop over metrics to populate each subplot.
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for m in method_list:
            points = metric_method_points[metric][m]
            if points:
                # Sort points by dimension.
                points.sort(key=lambda x: x[0])
                dims, scores = zip(*points)
            else:
                dims, scores = [], []
            # Use a special style for TDLHD.
            if m == "TDLHD":
                ax.plot(dims, scores, marker='*', linewidth=3, markersize=10, color='red', label="LoSAM (ours)")
            else:
                ax.plot(dims, scores, marker='o', linewidth=2, markersize=6, label=m)
        ax.set_xlabel("Dimensionality", fontsize=14)
        # ax.set_title(metric if metric != "Atop" else "$A_{top}$", fontsize=16)
        ax.set_ylabel(metric, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis limits and ticks based on the metric.
        if metric == "SHD":
            # Calculate the maximum score among all methods for this metric.
            max_val = 0
            for m in method_list:
                # Extract valid scores (filtering out nan)
                valid_scores = [score for (_, score) in metric_method_points[metric][m] if not np.isnan(score)]
                if valid_scores:
                    max_val = max(max_val, max(valid_scores))
            # Compute an upper bound as the next multiple of 5.
            y_max = 5 * (int(max_val / 5) + 1) if max_val > 0 else 5
            ax.set_ylim(0, y_max)
            ax.set_yticks(np.arange(0, y_max + 5, 5))
        else:
            ax.set_ylim(0, 1.01)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
        
        # Set x-ticks to the unique dimensions collected.
        all_dims = []
        for m in method_list:
            all_dims.extend([pt[0] for pt in metric_method_points[metric][m]])
        ax.set_xticks(sorted(set(all_dims)))
        
        if i == 0:
            ax.legend(fontsize=10)

    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ensure each dataset key is unique (e.g., include the dimension in the label)
    datasets = {
        "Uniform ER1 (d10)": os.path.join(root, "uniformd10ER1n1000"),
        "Uniform ER1 (d15)": os.path.join(root, "uniformd15ER1n1000"),
        "Uniform ER1 (d20)": os.path.join(root, "uniformd20ER1n1000"),
        "Uniform ER1 (d25)": os.path.join(root, "uniformd25ER1n1000")
    }
    metrics = ["Atop", "F1", "SHD"]
    # metrics = ["times", "matrix_times"]
    linear_folder = "linear_proportion_0.5"  # Use this folder for all datasets.
    method_list = ["TDLHD", "DLiNGAM", "RandSort", "VarSort"]

    plot_aggregated_metrics(root, datasets, metrics, linear_folder, method_list)
