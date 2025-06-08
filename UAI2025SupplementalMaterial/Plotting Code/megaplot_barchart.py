
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)
    sns.set_palette("vlag")
except ImportError:
    pass


def load_experiment_scores(root_dir, linear_dirs, file_methods, metric):
    """
    Loads scores for each method from the specified directories.
    Returns a dictionary where keys are file method names, and values are lists of scores.
    """
    score_data = {m: [] for m in file_methods}
    for lin_dir in linear_dirs:
        lin_path = os.path.join(root_dir, lin_dir)
        for m in file_methods:
            npy_path = os.path.join(lin_path, f"{m}_{metric}.npy")
            if os.path.isfile(npy_path):
                data = np.load(npy_path)
                score_data[m].extend(data.tolist())  # Append all scores
            else:
                print(f"Warning: No file found for {m} in {lin_dir}. Skipping...")
    return score_data


def plot_boxplots_all_linear(root_dir, datasets, metrics, linear_folders, file_methods, display_methods):
    """
    Creates a grid of boxplots using data from all specified linear folders.
    Uses file_methods to load the data and display_methods for the y-axis tick labels.
    """
    fig, axes = plt.subplots(len(datasets), len(metrics),
                             figsize=(18, 12), sharex=False, sharey=False)
    axes = np.atleast_2d(axes)

    # Ensure that the file corresponding to "TDLHD" (to be displayed as "LoSAM")
    # is placed first. (The order of display_methods should correspond to file_methods.)
    if "TDLHD" in file_methods:
        file_methods = ["TDLHD"] + [m for m in file_methods if m != "TDLHD"]
        # Rebuild display_methods so that the file "TDLHD" is labeled as "LoSAM"
        display_methods = ["LoSAM" if m == "TDLHD" else m for m in file_methods]

    # Reverse the order for plotting so that the first element appears at the top.
    rev_file_methods = file_methods[::-1]
    rev_display_methods = display_methods[::-1]

    num_methods = len(file_methods)
    # Choose a professional color palette (here we use Seaborn's "deep")
    colors = sns.color_palette("deep", num_methods)

    for i, (dataset_name, dataset_path) in enumerate(datasets.items()):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            # Load scores using the file names (file_methods)
            score_data = load_experiment_scores(dataset_path, linear_folders, file_methods, metric)
            # Prepare data in reversed order for the boxplot
            boxplot_data = [score_data[m] for m in rev_file_methods]

            # Create the horizontal boxplot (hide outliers with showfliers=False)
            box = ax.boxplot(
                boxplot_data,
                vert=False,
                patch_artist=True,
                widths=0.6,
                medianprops=dict(color="black", linewidth=3),
                showfliers=False
            )
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)

            y_positions = np.arange(1, len(rev_display_methods) + 1)
            if j == 0:
                ax.set_yticks(y_positions)
                ax.set_yticklabels(rev_display_methods, fontsize=20, fontweight="bold")
                # Removed the y-axis label per your request:
                # ax.set_ylabel(dataset_name, fontsize=14, fontweight="bold", labelpad=20)
            else:
                ax.set_yticks(y_positions)
                ax.set_yticklabels([])

            # For non-bottom rows, hide x-axis tick labels.
            if i != len(datasets) - 1:
                ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
            else:
                # For the bottom row, show x-axis tick labels with increased font size.
                ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labelsize=16)
                # Increase the x-axis label font size.
                ax.set_xlabel(f"{metric}", fontsize=18)

            ax.grid(True, linestyle="--", alpha=0.7, axis="x")
            ax.xaxis.set_major_locator(MaxNLocator(5))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    datasets = {
        # "Uniform ER1": os.path.join(root, "uniformd10ER1n1000")
        "Laplace ER1": os.path.join(root, "laplaced10ER1n1000")
        # "Gaussian ER1": os.path.join(root, "gaussiand10ER1n1000")
    }
    metrics = ["Atop", "F1", "SHD"]
    # File names used to load the data.
    file_methods = ["TDLHD", "CAPS", "NHTS", "DLiNGAM", "RESIT", "SCORE", "NoGAM", "CAM", "RandSort", "VarSort"]
    # Display names to be shown on the plot. Note that "TDLHD" is replaced by "LoSAM".
    display_methods = ["LoSAM", "CAPS", "NHTS", "DLiNGAM", "RESIT", "SCORE", "NoGAM", "CAM", "RandSort", "VarSort"]

    # Generate a list of linear folders (one for each linear proportion)
    # linear_folders = [f"linear_proportion_{suffix}" for suffix in ["0", "0.25", "0.5", "0.75", "1"]]
    linear_folders = [f"linear_proportion_{suffix}" for suffix in ["0.5"]]
    # for gaussian case
    # linear_folders = ["linear_proportion_0 best run copy"]

    plot_boxplots_all_linear(root, datasets, metrics, linear_folders, file_methods, display_methods)

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator


# try:
#     import seaborn as sns
#     sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)
# except ImportError:
#     pass


# def load_experiment_scores(root_dir, linear_dirs, file_methods, metric):
#     """
#     Loads scores for each method from the specified directories.
#     Returns a dictionary where keys are file method names, and values are lists of scores.
#     """
#     score_data = {m: [] for m in file_methods}
#     for lin_dir in linear_dirs:
#         lin_path = os.path.join(root_dir, lin_dir)
#         for m in file_methods:
#             npy_path = os.path.join(lin_path, f"{m}_{metric}.npy")
#             if os.path.isfile(npy_path):
#                 data = np.load(npy_path)
#                 score_data[m].extend(data.tolist())  # Append all scores
#             else:
#                 print(f"Warning: No file found for {m} in {lin_dir}. Skipping...")
#     return score_data


# def plot_boxplots_all_linear(root_dir, datasets, metrics, linear_folders, file_methods, display_methods):
#     """
#     Creates a grid of boxplots using data from all specified linear folders.
#     Uses file_methods to load the data and display_methods for the y-axis labels.
#     """
#     fig, axes = plt.subplots(len(datasets), len(metrics),
#                              figsize=(18, 12), sharex=False, sharey=False)
#     axes = np.atleast_2d(axes)

#     # Ensure that the file corresponding to "TDLHD" (to be displayed as "LoSAM")
#     # is placed first. (The order of display_methods should correspond to file_methods.)
#     if "TDLHD" in file_methods:
#         file_methods = ["TDLHD"] + [m for m in file_methods if m != "TDLHD"]
#         # Rebuild display_methods so that the file "TDLHD" is labeled as "LoSAM"
#         display_methods = ["LoSAM" if m == "TDLHD" else m for m in file_methods]

#     # Reverse the order for plotting so that the first element appears at the top.
#     rev_file_methods = file_methods[::-1]
#     rev_display_methods = display_methods[::-1]

#     num_methods = len(file_methods)
#     # Choose a professional color palette (here we use Seaborn's "deep")
#     colors = sns.color_palette("deep", num_methods)

#     for i, (dataset_name, dataset_path) in enumerate(datasets.items()):
#         for j, metric in enumerate(metrics):
#             ax = axes[i, j]
#             # Load scores using the file names (file_methods)
#             score_data = load_experiment_scores(dataset_path, linear_folders, file_methods, metric)
#             # Prepare data in reversed order for the boxplot
#             boxplot_data = [score_data[m] for m in rev_file_methods]

#             # Create the horizontal boxplot (hide outliers with showfliers=False)
#             box = ax.boxplot(
#                 boxplot_data,
#                 vert=False,
#                 patch_artist=True,
#                 widths=0.6,
#                 medianprops=dict(color="black", linewidth=3),
#                 showfliers=False
#             )
#             for patch, color in zip(box['boxes'], colors):
#                 patch.set_facecolor(color)

#             y_positions = np.arange(1, len(rev_display_methods) + 1)
#             if j == 0:
#                 ax.set_yticks(y_positions)
#                 ax.set_yticklabels(rev_display_methods, fontsize=12, fontweight="bold")
#                 ax.set_ylabel(dataset_name, fontsize=14, fontweight="bold", labelpad=20)
#             else:
#                 ax.set_yticks(y_positions)
#                 ax.set_yticklabels([])

#                # For non-bottom rows, show tick marks but remove tick labels.
#             if i != len(datasets) - 1:
#                 ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
#             else:
#                 # For the bottom row, show both tick marks and tick labels.
#                 ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labelsize = 10)
#                 ax.set_xlabel(f"{metric}", fontsize=15)
            
#             ax.grid(True, linestyle="--", alpha=0.7, axis="x")

#             ax.xaxis.set_major_locator(MaxNLocator(5))

#     plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     datasets = {
#         # "Uniform ER1": os.path.join(root, "uniformd10ER1n1000")
#         # "Laplace ER1": os.path.join(root, "laplaced10ER1n1000")
#         "Gaussian ER1": os.path.join(root, "gaussiand10ER1n1000")
#     }
#     metrics = ["Atop", "F1", "SHD"]
#     # File names used to load the data.
#     file_methods = ["TDLHD", "CAPS", "NHTS", "DLiNGAM", "RESIT", "SCORE", "NoGAM", "CAM", "RandSort", "VarSort"]
#     # Display names to be shown on the plot. Note that "TDLHD" is replaced by "LoSAM".
#     display_methods = ["LoSAM", "CAPS", "NHTS", "DLiNGAM", "RESIT", "SCORE", "NoGAM", "CAM", "RandSort", "VarSort"]

#     # Generate a list of linear folders (one for each linear proportion)
#     # linear_folders = [f"linear_proportion_{suffix}" for suffix in ["0", "0.25", "0.5", "0.75", "1"]]
#     linear_folders = [f"linear_proportion_{suffix}" for suffix in ["0.5"]]
#     # for gaussian case
#     linear_folders = ["linear_proportion_0 best run copy"]

#     plot_boxplots_all_linear(root, datasets, metrics, linear_folders, file_methods, display_methods)


# import os
# import numpy as np
# import matplotlib.pyplot as plt

# try:
#     import seaborn as sns
#     sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)
# except ImportError:
#     pass



# def load_experiment_scores(root_dir, linear_dirs, method_names, metric):
#     """
#     Loads scores for each method from the specified directory.
#     Returns a dictionary where keys are methods, and values are lists of scores.
#     """
#     score_data = {m: [] for m in method_names}
#     for lin_dir in linear_dirs:
#         lin_path = os.path.join(root_dir, lin_dir)
#         for m in method_names:
#             npy_path = os.path.join(lin_path, f"{m}_{metric}.npy")
#             if os.path.isfile(npy_path):
#                 data = np.load(npy_path)
#                 score_data[m].extend(data.tolist())  # Store all scores, not just mean
#             else:
#                 print(f"Warning: No file found for {m} in {lin_dir}. Skipping...")
#     return score_data

# def plot_boxplots_single_linear(root_dir, datasets, metrics, single_linear_folder, method_list):
#     # Disable shared y-axis to allow independent y-axis tick labels
#     fig, axes = plt.subplots(len(datasets), len(metrics), 
#                              figsize=(18, 12), sharex=False, sharey=False)
#     axes = np.atleast_2d(axes)

#     # Move "TDLHD" to the top by sorting method_list,
#     # then reverse the entire list so that "TDLHD" appears at the top of the plot.
#     method_list = ["TDLHD"] + [m for m in method_list if m != "TDLHD"]
#     reversed_methods = method_list[::-1]

#     # Generate distinct colors for each method
#     num_methods = len(method_list)
#     colors = sns.color_palette("deep", num_methods)

#     for i, (dataset_name, dataset_path) in enumerate(datasets.items()):
#         for j, metric in enumerate(metrics):
#             ax = axes[i, j]
#             # Load scores for the current metric and folder
#             score_data = load_experiment_scores(dataset_path, [single_linear_folder], method_list, metric)
#             # Use reversed order for the boxplot data
#             boxplot_data = [score_data[m] for m in reversed_methods]

#             # Create the horizontal boxplot
#             box = ax.boxplot(boxplot_data, vert=False, patch_artist=True, widths=0.6, 
#                              medianprops=dict(color="black", linewidth=3), showfliers = False)
#             for patch, color in zip(box['boxes'], colors):
#                 patch.set_facecolor(color)

#             y_positions = np.arange(1, len(reversed_methods) + 1)
#             if j == 0:
#                 ax.set_yticks(y_positions)
#                 ax.set_yticklabels(reversed_methods, fontsize=12, fontweight="bold")
#             else:
#                 ax.set_yticks(y_positions)
#                 ax.set_yticklabels([])

#             ax.set_title(f"{dataset_name} – {metric}", fontsize=14, pad=10)
#             ax.grid(True, linestyle="--", alpha=0.7, axis="x")

#     plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     datasets = {
#         "Uniform ER1": os.path.join(root, "uniformd10ER1n1000"),
#         "Laplace ER1": os.path.join(root, "laplaced10ER1n1000")
#     }
#     metrics = ["Atop", "F1", "SHD"]
#     method_list = ["TDLHD", "CAPS", "NHTS", "DLiNGAM", "RESIT", "SCORE", "NoGAM", "CAM", "RandSort", "VarSort"]

#     # Use only the folder "linear_proportion_0.5"
#     single_linear_folder = "linear_proportion_0.5"

#     plot_boxplots_single_linear(root, datasets, metrics, single_linear_folder, method_list)
