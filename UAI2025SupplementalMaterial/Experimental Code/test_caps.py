import torch
import numpy as np
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from npeet import entropy_estimators as ee
from causallearn.utils.cit import CIT
from conditional_independence import hsic_test
import time
import time
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from PyRKHSstats import hsic
from conditional_independence import hsic_test
from sklearn.linear_model import LinearRegression
from npeet import entropy_estimators as ee
import numpy as np
from scipy.stats import bernoulli, uniform
from scipy.spatial import KDTree
import lingam
import matplotlib.pyplot as plt
import os
import pandas as pd
from CausalDisco.analytics import r2_sortability
from CausalDisco.analytics import r2coeff
from CausalDisco.baselines import var_sort_regress
from sklearn.ensemble import RandomForestRegressor
from fcit import fcit
# turn this off to run on aws
import dodiscover
from collections import deque
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.PermutationBased.GRaSP import grasp
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
from causal_discovery.scamuv import SCAMUV 
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, recall_score, f1_score
import multiprocessing
import concurrent.futures
import numpy as np
import time
from tqdm import tqdm
from causallearn.utils.PDAG2DAG import pdag2dag
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import torch
from cdt.metrics import SHD

# packages for proxy var
import numpy as np
import pandas as pd
from causallearn.utils.cit import CIT

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="sklearn")


def topological_sort_from_CAPS(X):
    """
    Run CAPS and return the topological sort of variables.
    
    Args:
        X (np.ndarray): Input dataset.

    Returns:
        list: Topological sort of variables.
    """
    # Use GPU if available; otherwise, use CPU
    device = torch.device("cpu")  # Force CPU execution to avoid multiprocessing GPU conflicts

    def Stein_hess(X, eta_G, eta_H, s=None):
        """
        Estimates the diagonal of the Hessian of log p_X at the provided sample points
        X, using first and second-order Stein identities.
        """
        n, d = X.shape
        X = X.to(device)
        X_diff = X.unsqueeze(1) - X
        
        if s is None:
            D = torch.norm(X_diff, dim=2, p=2)
            s = D.flatten().median()
        
        K = torch.exp(-torch.norm(X_diff, dim=2, p=2) ** 2 / (2 * s**2)) / s
        
        nablaK = -torch.einsum('kij,ik->kj', X_diff, K) / s**2
        G = torch.linalg.solve(K + eta_G * torch.eye(n).to(device), nablaK)  # Safer than inverse()
        
        nabla2K = torch.einsum('kij,ik->kj', -1/s**2 + X_diff**2/s**4, K)
        return (-G**2 + torch.linalg.solve(K + eta_H * torch.eye(n).to(device), nabla2K)).to('cpu')

    def compute_top_order(X, eta_G, eta_H, dispersion="mean"):
        """
        Compute a topological order based on the Hessian estimated from Stein's method.
        """
        n, d = X.shape
        full_X = X.to(device)
        order = []
        active_nodes = list(range(d))

        for i in range(d-1):
            H = Stein_hess(X, eta_G, eta_H)
            
            # print(f"\nIteration {i}: Hessian Matrix H\n", H.numpy())  # Debugging

            if dispersion == "mean":  # Lemma 1 of CaPS
                l = int(H.mean(axis=0).argmax())
            else:
                raise ValueError("Unknown dispersion criterion")

            # print(f"Selected node: {active_nodes[l]}")  # Debugging

            order.append(active_nodes[l])
            active_nodes.pop(l)

            # Remove the selected variable from X
            X = torch.hstack([X[:, :l], X[:, l+1:]]).to(device)

        order.append(active_nodes[0])
        order.reverse()
        return order

    # Convert input to PyTorch tensor
    train_set = torch.tensor(X, dtype=torch.float32, device=device)
    
    return compute_top_order(train_set, eta_G=0.001, eta_H=0.001, dispersion="mean")



# num_hidden was 10 for initial AISTATS sub
def neural_network_transform(parent_data: np.ndarray, num_hidden: int = 10) -> np.ndarray:
    """
    Apply a neural network transformation to the input parent data.
    
    Args:
        parent_data (np.ndarray): The data from parent nodes, shape (n_samples, num_parents).
        num_hidden (int): Number of hidden units in the neural network.
        
    Returns:
        np.ndarray: Transformed data with shape (n_samples,).
    """
    # Initialize random weights for input to hidden layer and hidden to output layer
    # used for aistats
    weights_in = np.random.uniform(-5, 5, (parent_data.shape[1], num_hidden))  # (num_parents, num_hidden)
    bias_hidden = np.random.uniform(-5, 5, num_hidden)  # (num_hidden,)
    weights_out = np.random.uniform(-5, 5, num_hidden)  # (num_hidden,)

    # Compute hidden layer activations using tanh
    hidden_layer = np.tanh(np.dot(parent_data, weights_in) + bias_hidden)  # (n_samples, num_hidden)

    # Compute the final output as a weighted sum of hidden activations
    output = np.dot(hidden_layer, weights_out)  # (n_samples,)

    return output

def permute_data(X, adjacency_matrix, topological_order):
    # Generate a random permutation of indices
    d = adjacency_matrix.shape[0]
    permutation = np.random.permutation(d)

    # Permute the columns of X
    X = X[:, permutation]

    # Permute the rows and columns of adjacency_matrix
    adjacency_matrix = adjacency_matrix[permutation, :][:, permutation]

    # Update the topological_order according to the permutation
    topological_order = [np.where(permutation == i)[0][0] for i in topological_order]

    return X, adjacency_matrix, topological_order

def generate_adjacency_matrix(d, p):
    adjacency_matrix = np.zeros((d, d), dtype=int)
    for i in range(d):
        for j in range(i + 1, d):
            if np.random.rand() < p:
                adjacency_matrix[i, j] = 1
    return adjacency_matrix

def topological_sort(adjacency_matrix):
    d = adjacency_matrix.shape[0]
    in_degree = np.sum(adjacency_matrix, axis=0)
    zero_in_degree = [node for node in range(d) if in_degree[node] == 0]
    topological_order = []

    while zero_in_degree:
        node = zero_in_degree.pop()
        topological_order.append(node)
        for i in range(d):
            if adjacency_matrix[node, i] == 1:
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    zero_in_degree.append(i)

    if len(topological_order) != d:
        raise ValueError("The graph has cycles or is disconnected.")

    return topological_order
def generate_quadratic_data(n, d, avg_edges, dgm, noise, max_r2_sortability=0.7, max_attempts=1000):
    from CausalDisco.analytics import r2_sortability
     # Linear
    # c = 1
    # Mixed
    # c = 0.5
    # Nonlinear
    # c = 0
    # Parameter to control dgm process
    c = dgm

    def generate_data():
        # print(1)
        p = avg_edges
        adjacency_matrix = generate_adjacency_matrix(d, p)
        topological_order = topological_sort(adjacency_matrix)
        X = np.zeros((n, d))
        # print(adjacency_matrix)

        for node in topological_order:
            parents = np.where(adjacency_matrix[:, node] == 1)[0]
            if len(parents) == 0:
                if noise == "uniform":
                    # Uniform Variance
                    variance = np.where(np.random.rand(n) < 0.5, 1, np.sqrt(3))
                    X[:, node] = np.random.uniform(0, variance, n)
                if noise == "laplace":
                    variance = np.where(np.random.rand(n) < 0.5, np.sqrt(1/24), np.sqrt(9/24))
                    X[:, node] = np.random.laplace(0, variance, n)
                if noise == "gaussian":
                    variance = np.where(np.random.rand(n) < 0.5, (1/12)**0.5, (1/4)**0.5)
                    X[:, node] = np.random.normal(0, variance, n)
                
            else:
                parent_data = X[:, parents]
                quadratic_sum = 0

                if np.random.uniform(0,1) < c:
                    # Random Weights
                    lower_range = np.random.uniform(-1.5, -0.5, parent_data.shape[1])
                    upper_range = np.random.uniform(0.5, 1.5, parent_data.shape[1])
                    random_multipliers = np.where(np.random.rand(parent_data.shape[1]) < 0.5, lower_range, upper_range)
                    parent_data = parent_data * random_multipliers  
                    quadratic_sum = np.sum(parent_data, axis=1)
                else:
       
                    quadratic_sum = neural_network_transform(parent_data)

                # Unif Variance
                if noise == "uniform":
                    variance = np.where(np.random.rand(n) < 0.5, 1, np.sqrt(3))
                    X[:, node] = quadratic_sum + np.random.uniform(0, variance, n)
                if noise == "laplace":
                    variance = np.where(np.random.rand(n) < 0.5, np.sqrt(1/24), np.sqrt(9/24))
                    X[:, node] = quadratic_sum + np.random.laplace(0, variance, n)
                if noise == "gaussian":
                    variance = np.where(np.random.rand(n) < 0.5, (1/12)**0.5, (1/4)**0.5)
                    X[:, node] = quadratic_sum + np.random.normal(0, variance, n)

                # Normalize generated variable to prevent values from collapsing to 0 due to quadratic - should I keep this?
                # X[:, node] = normalize_vector(X[:, node])

        # Normalize all variables at the end just to make sure
        for node in range(d):
            X[:, node] = normalize_vector(X[:, node])
        
        return X, adjacency_matrix, topological_order
    
    attempt = 0
    while attempt < max_attempts:
        X, adjacency_matrix, topological_order = generate_data()
        try:
            r2_value = r2_sortability(X, adjacency_matrix)
        except Exception as e:
            continue

        if r2_value <= max_r2_sortability:
            break
        attempt += 1

    
    
    if attempt == max_attempts:
        print(f"Reached maximum attempts ({max_attempts}) without achieving desired sortability.")
    
    # Permute the Data
    X, adjacency_matrix, topological_order  = permute_data(X, adjacency_matrix, topological_order)

    parents_list = [set(np.where(adjacency_matrix[:, node] == 1)[0]) for node in range(d)]

    return X, adjacency_matrix, topological_order, parents_list

def normalize_vector(v):
    return (v - np.mean(v)) / np.std(v)

def count_topological_errors(M, k):
    """
    Counts the number of topological sorting errors in a DAG given its adjacency matrix and a topological order.

    :param M: A 2D list (list of lists) representing the adjacency matrix of the DAG.
              M[i][j] != 0 means there is a directed edge from j to i.
    :param k: A list representing the nodes in topological order.
    :return: The number of topological errors.
    """
    # Index each node based on its position in the topological order for quick lookup.
    index_map = {node: idx for idx, node in enumerate(k)}

    #Sum of potential errors
    sum = 0
    
    errors = 0
    # Check each pair (i, j) based on their indices in the topological order.
    for idx_i, i in enumerate(k):
        for idx_j, j in enumerate(k):
            if M[i][j] != 0:
                sum+=1
            # If i appears after j in the topological order but i causes j,
            # it's an error because i -> j should mean i should come before j.
                if idx_i > idx_j :
                    errors += 1

    if sum == 0:
        return 1

    # This function returns the % of correct ancestral relations determined (number of necessary ancestral relations)
    return (sum-errors)/sum


def cam_prune_from_order(topological_order, X):
    n_nodes = len(topological_order)

    # Generate dense adjacency matrix based on the topological order
    A_dense = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes-1):
        for j in range(i+1, n_nodes):
            A_dense[topological_order[i], topological_order[j]] = 1


    # Apply CAM pruning
    cam_model = dodiscover.toporder.CAM(prune=True)
    # A_pruned = cam_model.prune(X, A_dense, nx.DiGraph(), nx.DiGraph())
    A_pruned = cam_model.prune(X, A_dense, nx.DiGraph(), nx.DiGraph())
    # print(A_pruned)
    return A_pruned



# Parameters
d = 10  # Number of nodes
n = 1000  # Number of samples
num_er = 2
avg_edges = num_er * (2 / (d - 1))  # Average edge calculation
noise = "laplace"  # Noise type


# Number of trials
print(wd)

for dgm in [0, 0.25, 0.5, 0.75, 1]:
    for i in range(1):
        num_trials = 30
        # Store topological scores
        caps_Atop = []
        caps_SHD = []
        caps_F1 = []
        caps_Precision = []
        caps_Recall = []
        caps_times = []
        caps_matrix_times = []

        for trial in tqdm(range(num_trials)):
            # print(f"\n[Trial {trial + 1}] Generating data...")
            # Generate data using the new quadratic method
            X_sample, true_dag, true_top_order, parents_list = generate_quadratic_data(n, d, avg_edges, dgm, noise)
            start_time = time.time()
            # Run CAPS algorithm
            caps_top_order = topological_sort_from_CAPS(X_sample)
            end_time = time.time()
            matrix = cam_prune_from_order(caps_top_order, X_sample)
            matrix_time = time.time()

            # Compute errors
            caps_Atop_score = count_topological_errors(true_dag, caps_top_order)
            caps_SHD_score = SHD(true_dag, matrix)
            caps_f1_score = f1_score(true_dag.flatten(), matrix.flatten())
            caps_precision_score = precision_score(true_dag.flatten(), matrix.flatten())
            caps_recall_score = recall_score(true_dag.flatten(), matrix.flatten())

            # Store results
            caps_matrix_times.append(matrix_time - start_time)
            caps_times.append(end_time - start_time)
            caps_Atop.append(caps_Atop_score)
            caps_SHD.append(caps_SHD_score)
            caps_F1.append(caps_f1_score)
            caps_Precision.append(caps_precision_score)
            caps_Recall.append(caps_recall_score)

  
        np.save(wd + "/" + "linear_proportion_" +  str(dgm) + "/" + "CAPS_atop.npy", caps_Atop)
        np.save(wd + "/" + "linear_proportion_" +  str(dgm) + "/" + "CAPS_SHD.npy", caps_SHD)
        np.save(wd + "/" + "linear_proportion_" +  str(dgm) + "/" + "CAPS_F1.npy", caps_F1)
        np.save(wd + "/" + "linear_proportion_" +  str(dgm) + "/" + "CAPS_Precision.npy", caps_Precision)
        np.save(wd + "/" + "linear_proportion_" +  str(dgm) + "/" + "CAPS_Recall.npy", caps_Recall)
        np.save(wd + "/" + "linear_proportion_" +  str(dgm) + "/" + "CAPS_times.npy", caps_times)
        np.save(wd + "/" + "linear_proportion_" +  str(dgm) + "/" + "CAPS_matrix_times.npy", caps_matrix_times)

