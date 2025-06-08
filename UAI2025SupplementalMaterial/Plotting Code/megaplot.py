import os
import numpy as np
import matplotlib.pyplot as plt

# Optionally, use seaborn for improved aesthetics
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)
    sns.set_palette("deep")
except ImportError:
    pass

def load_experiment_scores(root_dir, linear_dirs, method_names, metric):
    score_data = {m: [] for m in method_names}
    for lin_dir in linear_dirs:
        lin_path = os.path.join(root_dir, lin_dir)
        temp_scores = {m: [] for m in method_names}
        for m in method_names:
            npy_path = os.path.join(lin_path, f"{m}_{metric}.npy")
            if os.path.isfile(npy_path):
                data = np.load(npy_path)
                temp_scores[m].append(data)
            else:
                print(f"Warning: No file found for {m} in {lin_dir}. Skipping...")

        for m in method_names:
            if temp_scores[m]:
                mean_score = float(np.median(temp_scores[m]))
                # mean_score = float(np.mean(temp_scores[m]))
            else:
                mean_score = np.nan
            score_data[m].append(mean_score)

    return score_data

def plot_mega_chart(root_dir, datasets, metrics, linear_dirs, method_list, selected_datasets=None):
    """
    Plot the mega chart for the given datasets and metrics.

    Parameters:
      - root_dir: Base directory containing the dataset directories.
      - datasets: A dictionary mapping dataset names to their paths.
      - metrics: A list of metric names.
      - linear_dirs: A list of linear proportion folder names.
      - method_list: A list of method names.
      - selected_datasets: Optional list of dataset names to plot (if None, plot all).
    """

    # If selected_datasets is provided, filter the datasets dictionary
    if selected_datasets is not None:
        datasets = {k: v for k, v in datasets.items() if k in selected_datasets}
        if not datasets:
            raise ValueError("No datasets match the selected_datasets list.")

    n_datasets = len(datasets)
    n_metrics = len(metrics)

    # Create subplots with proper dimensions
    fig, axes = plt.subplots(n_datasets, n_metrics, figsize=(18, 24), sharey='col')
    # Ensure axes is always a 2D array even if n_datasets or n_metrics is 1
    axes = np.atleast_2d(axes)

    x_labels = [ld.replace("linear_proportion_", "") for ld in linear_dirs]
    x_values = [float(ld.split("_")[-1]) for ld in linear_dirs]

    for i, (dataset_name, dataset_path) in enumerate(datasets.items()):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            score_data = load_experiment_scores(dataset_path, linear_dirs, method_list, metric)
            
            for method, scores in score_data.items():
                if method == "TDLHD":
                    ax.plot(x_values, scores, marker='*', linewidth=3, markersize=10, color='red', label="LoSAM (ours)")
                else:
                    ax.plot(x_values, scores, marker='o', linewidth=2, markersize=6, label=method)

            # Set the title based on the metric
            if metric == "Atop":
                ax.set_title("$A_{top}$", fontsize=16)
            else:
                ax.set_title(metric, fontsize=16)
            
            ax.set_xticks(x_values)
            ax.set_xticklabels(x_labels, fontsize=12)
            if metric == "SHD":
                # Ensure there is at least one valid score to compute the max
                max_score = max([s for s in scores if not np.isnan(s)] or [5])
                ax.set_yticks(np.arange(0, np.ceil(max_score / 5) * 5 + 5, 5))
            elif metric not in ["times", "matrix_times"]:
                ax.set_ylim(0, 1.05)
                ax.set_yticks(np.arange(0, 1.01, 0.1))
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(axis='y', labelsize=16, pad=0.001)
            ax.tick_params(axis='x', labelsize=16, pad=0.001)
            ax.set_xlabel("Linear Proportion", fontsize=18)
            
            # Set the dataset name as the y-axis label for the left-most subplot in each row.
            # if j == 0:
            #     ax.set_ylabel(dataset_name, fontsize=14)

    # handles, labels = ax.get_legend_handles_labels()
    # # remove for some plots, on right
    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=10, fontsize=12)
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, .985),
    #         ncol=10, fontsize=16, markerscale=1.5, labelspacing=1.0, handlelength=2, borderpad=1)


    plt.tight_layout(rect=[0, 0.12, 1, 0.90])
    plt.subplots_adjust(hspace=0.6)
    plt.show()

if __name__ == "__main__":
    datasets = {
        # "Uniform ER1": os.path.join(root, "uniformd10ER1n1000")
        #  "Uniform ER1": os.path.join(root, "uniformd20ER1n1000")
        #   "Uniform ER1": os.path.join(root, "uniformd30ER1n1000")
        "Laplace ER1": os.path.join(root, "laplaced10ER1n1000")
    }
    metrics = ["Atop", "F1", "SHD"]
    # metrics = ["times", "matrix_times"]
    linear_folders = [
        "linear_proportion_0", 
        "linear_proportion_0.25", 
        "linear_proportion_0.5",
        "linear_proportion_0.75", 
        "linear_proportion_1"
    ]
    # linear_folders = [ 
    #     "linear_proportion_0.5",
    # ]
    method_list = ["TDLHD", "CAPS", "NHTS", "DLiNGAM", "RESIT", "SCORE", "NoGAM", "CAM", "RandSort", "VarSort"]

    # method_list = ["TDLHD", "DLiNGAM",  "SCORE", "RandSort", "VarSort"]
    # To plot all datasets:
    plot_mega_chart(root, datasets, metrics, linear_folders, method_list)
    
    # Or, to plot a subset (for example, only "Laplace ER1"):
    # plot_mega_chart(root, datasets, metrics, linear_folders, method_list, selected_datasets=["Laplace ER1"])

    
    # Or, to plot a subset (for example, only "Laplace ER1"):
    # plot_mega_chart(root, datasets, metrics, linear_folders, method_list, selected_datasets=["Laplace ER1"])

# import os
# import numpy as np
# import matplotlib.pyplot as plt

# # Optionally, use seaborn for improved aesthetics
# try:
#     import seaborn as sns
#     sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)
# except ImportError:
#     pass

# def load_experiment_scores(root_dir, linear_dirs, method_names, metric):
#     score_data = {m: [] for m in method_names}
#     for lin_dir in linear_dirs:
#         lin_path = os.path.join(root_dir, lin_dir)
#         temp_scores = {m: [] for m in method_names}
#         for m in method_names:
#             npy_path = os.path.join(lin_path, f"{m}_{metric}.npy")
#             if os.path.isfile(npy_path):
#                 data = np.load(npy_path)
#                 temp_scores[m].append(data)
#             else:
#                 print(f"Warning: No file found for {m} in {lin_dir}. Skipping...")

#         for m in method_names:
#             if temp_scores[m]:
#                 mean_score = float(np.mean(temp_scores[m]))
#             else:
#                 mean_score = np.nan
#             score_data[m].append(mean_score)

#     return score_data

# def plot_mega_chart(root_dir, datasets, metrics, linear_dirs, method_list):
#     fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(18, 24), sharey='col')
#     x_labels = [ld.replace("linear_proportion_", "") for ld in linear_dirs]
#     x_values = [float(ld.split("_")[-1]) for ld in linear_dirs]

#     for i, (dataset_name, dataset_path) in enumerate(datasets.items()):
#         for j, metric in enumerate(metrics):
#             ax = axes[i, j]
#             score_data = load_experiment_scores(dataset_path, linear_dirs, method_list, metric)
            
#             for method, scores in score_data.items():
#                 if method == "TDLHD":
#                     ax.plot(x_values, scores, marker='*', linewidth=3, markersize=10, color='red', label=f"LoSAM (ours)")
#                 else:
#                     ax.plot(x_values, scores, marker='o', linewidth=2, markersize=6, label=method)

            
#             if metric == "Atop":
#                 ax.set_title("$A_{top}$", fontsize=16)
#             else:
#                 ax.set_title(metric, fontsize=16)
#             ax.set_xticks(x_values)
#             ax.set_xticklabels(x_labels, fontsize=12)
#             if metric == "SHD":
#                 ax.set_yticks(np.arange(0, np.ceil(max(scores) / 5) * 5 + 5, 5))
#             elif metric not in ["times", "matrix_times"]:
#                 ax.set_ylim(0, 1.05)
#                 ax.set_yticks(np.arange(0, 1.01, 0.1))
#             ax.grid(True, linestyle='--', alpha=0.7)
#             ax.tick_params(axis='y', labelsize=10, pad = .001)  # Adjust font size as needed

#             # Label only the first column with the dataset name
#             # if j == 0:
#             #     ax.set_ylabel(dataset_name, fontsize=10)
            
#             # Ensure every x-axis is labeled
#             ax.set_xlabel("Linear Proportion", fontsize=14)
#           # Place text below the three graphs for each dataset
#            # Add "Bob" between dataset rows
#     for i in range(len(datasets)):  # Ensure "Bob" isn't placed after the last row
#         row_axes = axes[i, :]  # Get all subplots in the row
#         y_min, y_max = row_axes[0].get_ylim()  # Get y-axis range for reference
#         row_y = row_axes[0].get_position().y0  # Get bottom y position of the row
#         if i == 0:
#             fig.text(0.5, row_y - 0.03, f"a) {list(datasets.keys())[i]}", ha='center', fontsize=14, fontweight='bold')
#         else:
#             fig.text(0.5, row_y - 0.05, f"b) {list(datasets.keys())[i]}", ha='center', fontsize=14, fontweight='bold')

#     handles, labels = ax.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=10, fontsize=12)
    
#     # Adjust layout to fit legend at the top and ensure x-axis labels fit
#     plt.tight_layout(rect=[0, 0.12, 1, 0.90])  # Adjust to leave room for "Bob"
#     plt.subplots_adjust(hspace=0.6)  # Adjust spacing between rows

#     plt.show()

# if __name__ == "__main__":
#     datasets = {
#         # "Uniform ER1": os.path.join(root, "uniformd10ER1n1000"),
#         "Laplace ER1": os.path.join(root, "laplaced10ER1n1000")
#     }
#     metrics = ["Atop", "F1", "SHD"]
#     linear_folders = [
#         "linear_proportion_0", 
#         "linear_proportion_0.25", 
#         "linear_proportion_0.5",
#         "linear_proportion_0.75", 
#         "linear_proportion_1"
#     ]
#     method_list = ["TDLHD", "CAPS", "NHTS", "DLiNGAM", "RESIT", "SCORE", "NoGAM", "CAM", "RandSort", "VarSort"]

#     plot_mega_chart(root, datasets, metrics, linear_folders, method_list)





# import os
# import numpy as np
# import matplotlib.pyplot as plt

# # Optionally, use seaborn for improved aesthetics
# try:
#     import seaborn as sns
#     sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)
# except ImportError:
#     pass

# def load_experiment_scores(root_dir, linear_dirs, method_names, metric):
#     score_data = {m: [] for m in method_names}
#     for lin_dir in linear_dirs:
#         lin_path = os.path.join(root_dir, lin_dir)
#         temp_scores = {m: [] for m in method_names}
#         for m in method_names:
#             npy_path = os.path.join(lin_path, f"{m}_{metric}.npy")
#             if os.path.isfile(npy_path):
#                 data = np.load(npy_path)
#                 temp_scores[m].append(data)
#             else:
#                 print(f"Warning: No file found for {m} in {lin_dir}. Skipping...")

#         for m in method_names:
#             if temp_scores[m]:
#                 mean_score = float(np.mean(temp_scores[m]))
#                 # mean_score = float(np.median(temp_scores[m]))
#             else:
#                 mean_score = np.nan
#             score_data[m].append(mean_score)

#     return score_data

# def plot_mega_chart(root_dir, datasets, metrics, linear_dirs, method_list):
#     fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(18, 24), sharex=True, sharey='col')
    
#     for i, (dataset_name, dataset_path) in enumerate(datasets.items()):
#         for j, metric in enumerate(metrics):
#             ax = axes[i, j]
#             score_data = load_experiment_scores(dataset_path, linear_dirs, method_list, metric)
#             x_values = [float(ld.split("_")[-1]) for ld in linear_dirs]

#             for method, scores in score_data.items():
#                 if method == "TDLHD":
#                     ax.plot(x_values, scores, marker='*', linewidth=3, markersize=10, color='red', label=f"{method} (Ours)")
#                 else:
#                     ax.plot(x_values, scores, marker='o', linewidth=2, markersize=6, label=method)

#             if i == 0:
#                 ax.set_title(metric, fontsize=16)
#             if j == 0:
#                 ax.set_ylabel(dataset_name, fontsize=14)
#             ax.set_xticks(np.arange(0.0, 1.01, 0.25))
#             if metric not in ["SHD", "times", "matrix_times"]:
#                 ax.set_ylim(0, 1.05)
#                 # ax.set_ylim(0.3, 1.05)
#                 ax.set_yticks(np.arange(0.1, 1.01, 0.1))
#             ax.grid(True, linestyle='--', alpha=0.7)

#     handles, labels = ax.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=12)
#     plt.tight_layout(rect=[0, 0.05, 1, 0.97])
#     plt.show()

# if __name__ == "__main__":
#     datasets = {
#         "Uniform ER1": os.path.join(root, "uniformd10ER1n1000"),
#         # "Uniform ER1": os.path.join(root, "uniformd10ER1n1000"),
#         # "Uniform ER1": os.path.join(root, "new_rootuniformd10ER1n1000"),
#         # "Uniform ER2": os.path.join(root, "uniformd10ER2n1000")
#         # "Uniform ER2": os.path.join(root, "uniformd10ER2n4000")
#         # "Uniform ER3": os.path.join(root, "uniformd10ER3n1000")
#         # "Uniform ER3": os.path.join(root, "new_rootuniformd10ER3n1000")
#         # "Gaussian ER1": os.path.join(root, "gaussiand10ER1n1000"),
#         # "Gaussian ER3": os.path.join(root, "gaussiand10ER3n1000"),
#         "Laplace ER1": os.path.join(root, "laplaced10ER1n1000")
#         # "Laplace ER3": os.path.join(root, "laplaced10ER2n1000")
#         # "Laplace ER3": os.path.join(root, "laplaced10ER3n1000")
#     }
#     # metrics = ["root_f1", "root_rec", "root_pre"]
#     metrics = ["Atop", "F1", "SHD"]
#     # metrics = ["times", "matrix_times"]
#     linear_folders = [
#         "linear_proportion_0", 
#         "linear_proportion_0.25", 
#         "linear_proportion_0.5",
#         "linear_proportion_0.75", 
#         "linear_proportion_1"
#     ]
#     method_list = ["TDLHD", "CAPS", "NHTS", "DLiNGAM", "RESIT", "SCORE", "NoGAM", "CAM", "RandSort", "VarSort", "R2Sort"]
#     # method_list = ["TDLHD", "CAPS", "CAM", "DLiNGAM"]
#     # method_list = ["TDLHD", "TDLHD_NEW"]

#     plot_mega_chart(root, datasets, metrics, linear_folders, method_list)


# import os
# import numpy as np
# import matplotlib.pyplot as plt

# # Optionally, use seaborn for improved aesthetics
# try:
#     import seaborn as sns
#     sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)
# except ImportError:
#     pass

# def bootstrap_ci(data, num_bootstrap=1000, ci=95):
#     """Compute bootstrapped confidence intervals."""
#     if len(data) == 0:
#         return np.nan, np.nan
    
#     boot_samples = np.random.choice(data, size=(num_bootstrap, len(data)), replace=True)
#     boot_means = np.median(boot_samples, axis=1)
#     lower_bound = np.percentile(boot_means, (100 - ci) / 2)
#     upper_bound = np.percentile(boot_means, 100 - (100 - ci) / 2)
#     return lower_bound, upper_bound

# def compute_iqr(data):
#     """Compute the interquartile range (IQR) for given data."""
#     if len(data) == 0:
#         return np.nan, np.nan
#     lower_bound = np.percentile(data, 25)
#     upper_bound = np.percentile(data, 75)
#     return lower_bound, upper_bound

# def load_experiment_scores(root_dir, linear_dirs, method_names, metric):
#     score_data = {m: [] for m in method_names}
#     score_cis = {m: [] for m in method_names}  # Store confidence intervals
    
#     for lin_dir in linear_dirs:
#         lin_path = os.path.join(root_dir, lin_dir)
#         temp_scores = {m: [] for m in method_names}
        
#         for m in method_names:
#             npy_path = os.path.join(lin_path, f"{m}_{metric}.npy")
#             if os.path.isfile(npy_path):
#                 data = np.load(npy_path)
#                 temp_scores[m].extend(data)
#             else:
#                 print(f"Warning: No file found for {m} in {lin_dir}. Skipping...")
        
#         for m in method_names:
#             if temp_scores[m]:
#                 mean_score = float(np.median(temp_scores[m]))
#                 lower_ci, upper_ci = bootstrap_ci(temp_scores[m])
#                 # lower_ci, upper_ci = compute_iqr(temp_scores[m])
#             else:
#                 mean_score, lower_ci, upper_ci = np.nan, np.nan, np.nan
            
#             score_data[m].append(mean_score)
#             score_cis[m].append((lower_ci, upper_ci))
    
#     return score_data, score_cis

# def plot_mega_chart(root_dir, datasets, metrics, linear_dirs, method_list):
#     fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(18, 24), sharex=True, sharey='col')
    
#     for i, (dataset_name, dataset_path) in enumerate(datasets.items()):
#         for j, metric in enumerate(metrics):
#             ax = axes[i, j]
#             score_data, score_cis = load_experiment_scores(dataset_path, linear_dirs, method_list, metric)
#             x_values = [float(ld.split("_")[-1]) for ld in linear_dirs]

#             for method, scores in score_data.items():
#                 lower_cis, upper_cis = zip(*score_cis[method])
                
#                 if method == "TDLHD":
#                     ax.plot(x_values, scores, marker='*', linewidth=3, markersize=10, color='red', label=f"{method} (Ours)")
#                     ax.fill_between(x_values, lower_cis, upper_cis, color='red', alpha=0.2)
#                 else:
#                     ax.plot(x_values, scores, marker='o', linewidth=2, markersize=6, label=method)
#                     ax.fill_between(x_values, lower_cis, upper_cis, alpha=0.2)
            
#             if i == 0:
#                 ax.set_title(metric, fontsize=16)
#             if j == 0:
#                 ax.set_ylabel(dataset_name, fontsize=14)
#             ax.set_xticks(np.arange(0.0, 1.01, 0.25))
#             if metric not in ["SHD", "times", "matrix_times"]:
#                 ax.set_ylim(0, 1.05)
#                 ax.set_yticks(np.arange(0.1, 1.01, 0.1))
#             ax.grid(True, linestyle='--', alpha=0.7)
    
#     handles, labels = ax.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=12)
#     plt.tight_layout(rect=[0, 0.05, 1, 0.97])
#     plt.show()

# if __name__ == "__main__":
#     datasets = {
#         "Laplace ER1": os.path.join(root, "laplaced10ER1n1000"),
#         "Laplace ER3": os.path.join(root, "laplaced10ER2n1000")
#     }
#     metrics = ["Atop", "F1", "SHD"]
#     linear_folders = [
#         "linear_proportion_0", 
#         "linear_proportion_0.25", 
#         "linear_proportion_0.5",
#         "linear_proportion_0.75", 
#         "linear_proportion_1"
#     ]
#     method_list = ["TDLHD", "CAPS", "NHTS", "DLiNGAM", "RESIT", "SCORE", "NoGAM", "CAM", "RandSort", "VarSort", "R2Sort"]
    
#     plot_mega_chart(root, datasets, metrics, linear_folders, method_list)

