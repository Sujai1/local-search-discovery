# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Optionally set seaborn style
# sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)

# def load_metrics_from_files(local_dir, method_names, metrics):
#     data = {method: {} for method in method_names}
#     for method in method_names:
#         for metric in metrics:
#             file_path = os.path.join(local_dir, f"{method}_{metric}.npy")
#             if os.path.isfile(file_path):
#                 data[method][metric] = np.load(file_path)
#             else:
#                 data[method][metric] = np.nan  # Handle missing data gracefully
#     return data

# def plot_metrics(data, metrics):
#     methods = list(data.keys())
#     n_methods = len(methods)

#     # Prepare the data matrix
#     metric_matrix = np.zeros((n_methods, len(metrics)))
#     for i, method in enumerate(methods):
#         for j, metric in enumerate(metrics):
#             values = data[method].get(metric)
#             if isinstance(values, np.ndarray) and len(values) > 0:
#                 metric_matrix[i, j] = np.median(values)
#             else:
#                 metric_matrix[i, j] = np.nan

#     # Plotting
#     fig, ax = plt.subplots(figsize=(12, 8))
#     im = ax.imshow(metric_matrix, cmap="viridis", aspect="auto")

#     # Add labels
#     ax.set_xticks(np.arange(len(metrics)))
#     ax.set_yticks(np.arange(n_methods))
#     ax.set_xticklabels(metrics)
#     ax.set_yticklabels(methods)

#     # Rotate the tick labels and set alignment
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

#     # Loop over data dimensions and create text annotations
#     for i in range(n_methods):
#         for j in range(len(metrics)):
#             value = metric_matrix[i, j]
#             if not np.isnan(value):
#                 ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white")

#     ax.set_title("Performance Metrics by Method")
#     fig.colorbar(im, ax=ax)
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     # Directory where the local files are stored

#     method_names = ["TDLHD", "NHTS", "DLiNGAM", "RESIT", "SCORE", "NoGAM", "CAM", "RandSort", "VarSort", "R2Sort"]
#     metrics = ["atop", "SHD", "F1", "Precision", "Recall"]

#     # Load data
#     data = load_metrics_from_files(local_dir, method_names, metrics)

#     # Plot the metrics
#     plot_metrics(data, metrics)

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optionally set seaborn style
sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)

def load_metrics_from_files(local_dir, method_names, metrics):
    data = {method: {} for method in method_names}
    for method in method_names:
        for metric in metrics:
            file_path = os.path.join(local_dir, f"{method}_{metric}.npy")
            if os.path.isfile(file_path):
                data[method][metric] = np.load(file_path)
            else:
                data[method][metric] = np.nan  # Handle missing data gracefully
    return data

def plot_metrics(data, metrics):
    methods = list(data.keys())
    n_methods = len(methods)

    # Prepare the data matrix
    metric_matrix = np.zeros((n_methods, len(metrics)))
    std_matrix = np.zeros((n_methods, len(metrics)))
    for i, method in enumerate(methods):
        for j, metric in enumerate(metrics):
            values = data[method].get(metric)
            if isinstance(values, np.ndarray) and len(values) > 0:
                # metric_matrix[i, j] = np.median(values)
                metric_matrix[i, j] = np.mean(values)
                std_matrix[i, j] = np.std(values)
            else:
                metric_matrix[i, j] = np.nan
                std_matrix[i, j] = np.nan

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 0.5 * (n_methods + 2)))
    ax.axis('off')

    # Create table data with mean ± std format
    table_data = [[method] + [f"{metric_matrix[i, j]:.2f}±{std_matrix[i, j]:.2f}" if not np.isnan(metric_matrix[i, j]) else "-" for j in range(len(metrics))] for i, method in enumerate(methods)]

    # Add headers
    column_labels = ["Method"] + metrics

    # Create table
    table = ax.table(cellText=table_data, colLabels=column_labels, loc='center', cellLoc='center', cellColours=[['#f5f5f5']*len(column_labels) for _ in methods])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    # Bold header
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(weight='bold', backgroundcolor='#d9d9d9')

    # Highlight the best and second-best metrics
    for j in range(1, len(metrics) + 1):  # Skip 'Method' column
        column_data = [metric_matrix[i, j - 1] for i in range(n_methods)]

        if metrics[j - 1] in ["SHD"]:  # Lower is better
            sorted_indices = np.argsort(column_data)
        else:  # Higher is better
            sorted_indices = np.argsort(column_data)[::-1]

        # Bold best
        if not np.isnan(column_data[sorted_indices[0]]):
            table[(sorted_indices[0] + 1, j)].set_text_props(weight='bold')

        # Underline second best
        if len(sorted_indices) > 1 and not np.isnan(column_data[sorted_indices[1]]):
            cell = table[(sorted_indices[1] + 1, j)]
            cell_text = cell.get_text().get_text()
            cell.get_text().set_text(f'\u0332'.join(cell_text) + '\u0332')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Directory where the local files are stored
   
    # for random sample
  

    method_names = ["TDLHD", "NHTS", "DLiNGAM", "RESIT", "SCORE", "NoGAM", "CAM", "RandSort", "VarSort", "R2Sort"]
    # method_names = ["TDLHD", "NHTS", "DLiNGAM", "RESIT", "SCORE", "NoGAM", "CAM", "CAPS"]
    metrics = ["atop", "SHD", "F1"]

    # Load dataº
    data = load_metrics_from_files(local_dir, method_names, metrics)

    # Plot the metrics
    plot_metrics(data, metrics)
