import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from npeet import entropy_estimators as ee
from causallearn.utils.cit import CIT
from conditional_independence import hsic_test
import time
import time
import numpy as np
from collections import defaultdict
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
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
from CausalDisco.analytics import r2_sortability
from CausalDisco.analytics import r2coeff
from CausalDisco.baselines import var_sort_regress
from CausalDisco.baselines import r2_sort_regress
from sklearn.ensemble import RandomForestRegressor
from cdt.metrics import SHD
# from cdt.metrics import SID
# from cdt.metrics import precision_recall
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
from sklearn.preprocessing import StandardScaler
# import multiprocessing
# multiprocessing.set_start_method("spawn", force=True)


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


def topological_sort_CPDAG(graph):
    """
    Perform a topological sort on a DAG.
    
    Parameters:
    graph (np.ndarray): A 2D numpy array representing the adjacency matrix of the DAG.
                        graph[j, i] = 1 and graph[i, j] = -1 indicate i --> j.
                        graph[i, j] = graph[j, i] = -1 indicates i -- j.
                        
    Returns:
    list: A list of nodes in topologically sorted order.
    """
    n = graph.shape[0]
    in_degree = np.zeros(n, dtype=int)
    
    # Calculate in-degrees of all vertices
    for i in range(n):
        for j in range(n):
            if graph[j, i] == 1:  # j --> i
                in_degree[i] += 1
    
    # Enqueue all vertices with in-degree 0
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        
        # For all vertices v adjacent to u, decrease in-degree by 1
        for v in range(n):
            if graph[u, v] == 1:  # u --> v
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
    
    # Check if there was a cycle
    if len(topo_order) != n:
        raise ValueError("The graph has at least one cycle")
    
    return topo_order



# %%
def check_independence(xi, xj, thresh):
    """
    Check if xi and xj are independent using Kernel Conditional Independence (KCI) test.
    """
    data = np.column_stack((xi, xj))
    kci_obj = CIT(data, "kci")
    pValue = kci_obj(0, 1, [])
    return pValue > thresh

def check_conditional_independence(xi, xj, given, thresh):
    """
    Check if xi and xj are conditionally independent given 'given' using Kernel Conditional Independence (KCI) test.
    """
    data = np.column_stack((xi, xj, given))
    kci_obj = CIT(data, "kci")
    pValue = kci_obj(0, 1, list(range(2, data.shape[1])))
    return pValue > thresh

def calculate_residual(y, X):
    """
    Calculate the residual of y regressed on X using Kernel Ridge Regression.
    """
    # og
    # krr = KernelRidge(kernel='polynomial', alpha=1, degree=3, coef0=1)
    krr = KernelRidge(kernel='polynomial', alpha= 0.1, degree=8, coef0=1)
    #See if this worsens performance
    # krr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.01)
    krr.fit(X, y)
    y_pred = krr.predict(X)
    residuals = y - y_pred
    return residuals

def get_Pij(i, j, ind, features, d):
    """
    Get the set of features that are independent of xi but not independent of xj.
    """
    Pij = []
    for k in range(d):
        if k != i and k != j:
            if k not in ind[i] and k in ind[j]:
                Pij.append(features[k])
    return np.array(Pij).T

def check_PP2(i, PRS, d):
    '''Checks whether PP2 criterion holds for i: i must be identified in PP2 relation with at least one j to be a root, and if a j is in PP2 relation with i,
    i cannot be a root.'''
    pot_root = True
    #count = 0
    for j in range(d):
        if j!=i:
            if (j,i) in PRS and PRS[(j,i)] == 'PP2':
                pot_root = False
            #if (i,j) in PRS and PRS[(i,j)] == 'PP2':
                #count = 1
    #if count == 0:
        #pot_root = False
    return pot_root




def hierarchical_topological_sort(features, ind):
    d = len(features)
    PRS = {}
    pi_H = {}

    # Stage 1: Not-PP1 Relations
    for i in range(d):
        for j in range(d):
            if i != j:
                if i in ind[j] or j in ind[i]:
                    PRS[(i, j)] = 'Not in PP1'

    for i in range(d):
        if ind[i] == []:
            PRS[i] = 'Isolated'
            pi_H[i] = 1

    # Stage 2: PP2 Relations
    for i in range(d):
        for j in range(d):
            if (i, j) not in PRS or PRS[(i, j)] != 'Not in PP1':
                continue
            Pij = get_Pij(i, j, ind, features, d)
            xj_residual = calculate_residual(features[j], features[i].reshape(-1, 1))
            if Pij.size > 0:
                xj_residual_P = calculate_residual(features[j], np.hstack((features[i].reshape(-1, 1), Pij)))
            else:
                xj_residual_P = xj_residual
                # thresh should = 0.05
            if check_independence(features[i], xj_residual, thresh=0.05) or check_independence(features[i], xj_residual_P, thresh=0.05):
                PRS[(i, j)] = 'PP2'

    # Stage 3: Root Identification
    for i in range(d):
        if i in PRS and PRS[i] == 'Isolated':
            continue
        
        # Need to only check vertices that pass PP2 criterion
        #if not check_PP2(i,PRS,d):
        #   continue

        dependents = [features[k] for k in range(d) if k != i and (i, k) in PRS and PRS[(i, k)] != 'PP2']
        flag = True
        for xk in dependents:
            # was alpha = 0.05
            # trying .10 on 11/25
            # trying .20
            # was 0.5?
            if all(check_conditional_independence(features[j], xk, features[i], thresh=0.5) for j in range(d) if (i, j) in PRS and PRS[(i, j)] == 'PP2'):
                flag = False
                # If the above condition holds, i cannot be a root, so we stop immediately
                break
        if flag == True:
            pi_H[i] = 1
   
    roots = [i for i in range(d) if i in pi_H and pi_H[i] == 1]

    return roots

def marg_dep(data, alpha=0.05):
    d = data.shape[1]
    ind_collection = [[] for _ in range(d)]
    for i in range(d):
        for j in range(i + 1, d):
            if hsic_test(data, i, j, [])['p_value'] < alpha:
                ind_collection[i].append(j)
                ind_collection[j].append(i)
    return ind_collection

def nonlinear_sort_new(sorted_list, unsorted_list, ind, data):
    while unsorted_list:
        measures = np.full(data.shape[1], np.inf)
        for x in unsorted_list:
            anc_x = ind[x]
            features = list(set(anc_x) & set(sorted_list))
            if not features:
                # If no features are found, set measure to a high value (indicating low priority)
                measures[x] = np.inf
                continue
            X = np.array([data[:, y] for y in features]).T
            y = np.array(data[:, x])
            #alpha = 0.1
            #og
            # krr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.01)
            krr = KernelRidge(kernel='polynomial', alpha= 0.1, degree=8, coef0=1)
            krr.fit(X, y)
            residuals = y - krr.predict(X)
            mi_values = []
            for y in features:
                mi = ee.mi(data[:, y], residuals)
                mi_values.append(max(0, mi))
            # used to be:
            # measures[x] = np.mean(mi_values)
            # bottom is new
            # Use a cutoff to decide if x in next layer
            if all(mi_values[j] < 0.05 for j in range(0,len(features))):
                measures[x] = 0
            #Else, use avg to ensure at least one vertex gets selected
            else:
                measures[x] = np.mean(mi_values)

        
        # Check if all measures are np.inf
        if np.all(measures == np.inf):
            # If all measures are np.inf, randomly select an element from unsorted_list
            min_index = np.random.choice(unsorted_list)
        else:
            # Select just one vertex for comparison with linear topological sorts
            min_index = np.argmin(measures)
        
        sorted_list.append(min_index)
        unsorted_list.remove(min_index)
    return sorted_list

def NHTS_old(data):
    """
    Nonlinear Hierarchical Topological Sort (NHTS) function.
    
    Parameters:
    data (np.array): Dataset with d variables as columns and n samples as rows.
    
    Returns:
    list: Topological ordering of the variables.
    """
    ind = marg_dep(data)
    roots = hierarchical_topological_sort(data.T, ind)
    real_roots = deepcopy(roots)
    unsorted = [i for i in range(data.shape[1]) if i not in roots]
    output = nonlinear_sort_new(roots, unsorted, ind, data)
    return output, real_roots

def NHTS_old_sort(data, true_roots):
    """
    Nonlinear Hierarchical Topological Sort (NHTS) function.
    
    Parameters:
    data (np.array): Dataset with d variables as columns and n samples as rows.
    
    Returns:
    list: Topological ordering of the variables.
    """
    ind = marg_dep(data)
    # roots = hierarchical_topological_sort(data.T, ind)
    roots = true_roots
    # real_roots = deepcopy(roots)
    unsorted = [i for i in range(data.shape[1]) if i not in roots]
    output = nonlinear_sort_new(roots, unsorted, ind, data)
    return output

def check_independence_L(xi, xj, thresh):
    """
    Check if xi and xj are independent using Kernel Conditional Independence (KCI) test.
    """
    data = np.column_stack((xi, xj))
    kci_obj = CIT(data, "kci")
    pValue = kci_obj(0, 1, [])
    return pValue > thresh

def check_independence_pvalue_L(xi, xj):
    """
    Check if xi and xj are independent using Kernel Conditional Independence (KCI) test.
    """
    data = np.column_stack((xi, xj))
    kci_obj = CIT(data, "kci")
    pValue = kci_obj(0, 1, [])
    return pValue

def check_conditional_independence_L(xi, xj, given, thresh):
    """
    Check if xi and xj are conditionally independent given 'given' using Kernel Conditional Independence (KCI) test.
    """
    data = np.column_stack((xi, xj, given))
    kci_obj = CIT(data, "kci")
    pValue = kci_obj(0, 1, list(range(2, data.shape[1])))
    return pValue > thresh

def calculate_residual_L(y, X):
    """
    Calculate the residual of y regressed on X using Kernel Ridge Regression.
    """
    # krr = KernelRidge(kernel='polynomial', alpha=0.1, degree=3, coef0=1)
    # krr = KernelRidge(kernel='polynomial', alpha=0.1, degree=8, coef0=1)
    # krr = KernelRidge(kernel='polynomial', alpha=1, degree=8, coef0=1)
#     krr = xgb.XGBRegressor(
#     n_estimators=50,           # Fewer trees to prevent overfitting
#     learning_rate=0.1,         # Lower learning rate for stability
#     max_depth=3,               # Shallower trees to avoid overfitting
#     min_child_weight=5,        # Minimum sum of weights in a child node (regularization)
#     subsample=0.8,             # Randomly sample 80% of rows per tree
#     colsample_bytree=0.8,      # Randomly sample 80% of features per tree
#     reg_alpha=1.0,             # L1 regularization (lasso)
#     reg_lambda=1.0,            # L2 regularization (ridge)
#     random_state=42            # For reproducibility
# )
# this was the best
    krr = RandomForestRegressor(
    n_estimators=100,        # Fewer trees suffice for small datasets
    max_depth=10,            # Limit depth to avoid overfitting
    min_samples_split=10,    # Require more samples to split nodes (regularization)
    min_samples_leaf=5,      # Ensure leaf nodes have at least 5 samples
    max_features="sqrt",     # Use a subset of features to make trees diverse
    random_state=42          # For reproducibility
)
#     krr = RandomForestRegressor(
#     n_estimators=200,        # Increase the number of trees for better stability
#     max_depth=20,            # Allow deeper trees but prevent overfitting
#     min_samples_split=5,     # Looser regularization compared to smaller datasets
#     min_samples_leaf=2,      # Smaller leaf size allows more granularity
#     max_features="sqrt"     # Subset of features for diversity
# )
#     krr = RandomForestRegressor(
#     n_estimators=300,        # Increase trees to stabilize predictions
#     max_depth=None,          # Let trees grow fully (Random Forests reduce overfitting via averaging)
#     min_samples_split=2,     # Default splitting threshold
#     min_samples_leaf=1,      # Default for capturing as much information as possible
#     max_features="sqrt",     # Default for random forests
#     random_state=42          # For reproducibility
# )

    # krr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.01)
    krr.fit(X, y)
    y_pred = krr.predict(X)
    residuals = y - y_pred

    # added stuff to consider nonlin reg - can block out
    # loss_nonlinear = mean_squared_error(y, y_pred)
    lin_reg = LinearRegression()
    # lin_reg = Ridge(alpha = .0001)
    lin_reg.fit(X, y)
    y_pred_linear = lin_reg.predict(X)
    # loss_linear = mean_squared_error(y, y_pred_linear)
    # # new way of doing:
    residuals_krr = y - y_pred
    residuals_lin = y - y_pred_linear

    # # try to just use lin resid
    # return residuals_lin

     # Compute Mutual Information (MI) sum for both residuals
    def compute_mi_sum(X, residuals):
        mi_values = []
        for j in range(X.shape[1]):  # Iterate over each regressor
            mi = ee.mi(X[:, j], residuals)  # Compute MI
            mi_values.append(max(0, mi))  # Ensure non-negative MI
        return np.sum(mi_values)  # Sum over all regressors
    
    mi_sum_krr = compute_mi_sum(X, residuals_krr)
    mi_sum_lin = compute_mi_sum(X, residuals_lin)

    # try using the loss instead of mi sometime?

    if mi_sum_krr < mi_sum_lin:
        return residuals_krr, mi_sum_krr
    else:
        # print("linear")
        return residuals_lin, mi_sum_lin

def get_Pij_L(i, j, ind, features, d):
    """
    Get the set of features that are independent of xi but not independent of xj.
    """
    Pij = []
    for k in range(d):
        if k != i and k != j:
            if k not in ind[i] and k in ind[j]:
                Pij.append(features[k])
    return np.array(Pij).T

def check_PP2_L(i, PRS, d):
    '''Checks whether PP2 criterion holds for i: i must be identified in PP2 relation with at least one j to be a root, and if a j is in PP2 relation with i,
    i cannot be a root.'''
    pot_root = True
    count = 0
    for j in range(d):
        if j!=i:
            if (j,i) in PRS and PRS[(j,i)] == 'PP2':
                pot_root = False
            if (i,j) in PRS and PRS[(i,j)] == 'PP2':
                count = 1
    if count == 0:
       pot_root = False
    return pot_root

def find_v_structures_L(ind_collection):
    dependent_triplets = []
    d = len(ind_collection)

    for i in range(d):
        for j in ind_collection[i]:
            for k in ind_collection[i]:
                if j != k:
                    # Check if j and k are independent
                    if k not in ind_collection[j] and j not in ind_collection[k]:
                        # i is dependent on both j and k, and j and k are independent
                        dependent_triplets.append((i, j, k))
    
    return dependent_triplets

def check_v_structure_L(i, dependent_triplets):
    for triplet in dependent_triplets:
        if triplet[0] == i:
            return True
    return False

# this is the one used in all exps on 12/6
def hierarchical_topological_sort_L(features, ind, v_structures):
    d = len(features)
    PRS = {}
    pi_H = {}
    
    # if the vertex is isolated, make it a root
    for i in range(d):
        if ind[i] == []:
            pi_H[i] = 1

    # Find all vertices in VS
    VS = []
    for i in range(d):
        if i in pi_H and pi_H[i] == 1:
            continue
        if check_v_structure_L(i, v_structures):
            VS.append(i)
        
    pot_roots = set()
    pot_roots_mi_sum = defaultdict(int)

    # Stage 2: Leveraging VS
    for i in range(d):
        # if check_v_structure_L(i, v_strsuctures):
        #     continue
        if i in VS:
            continue
        nonVS = list(set(ind[i]).difference(set(VS)))
        if len(nonVS) == 0:
            pi_H[i] = 1
        else:
            for j in nonVS:
                xj_residual, mi_sum = calculate_residual_L(features[j], features[i].reshape(-1, 1))
                pot_roots_mi_sum[i] += mi_sum
                if check_independence_L(features[i], xj_residual, thresh=0.01):
                    PRS[(i, j)] = 'PP2'
                    pot_roots.add(i)

    try:
        # Stage 3: Root Identification
        for i in range(d):
            # Don't check roots
            if i in pi_H and pi_H[i] == 1:
                continue
            # Don't check vertices in VS
            if i in VS:
                continue
            # Check the cond ind position
            # Don't Check vertices not pot_roots
            if i not in pot_roots:
                continue


            # # This is the cond-ind approach to removing non-roots 
            # # Who are dependents - not i, dependent on i, not a known descendant of i, and k not in VS
            # dependents = [features[k] for k in range(d) if k != i and k in ind[i] and (i,k) not in PRS and k not in VS]
            # if len(dependents) == 0:
            #     pi_H[i] = 1
            #     continue
            # flag = True
            # for xk in dependents:
            #     # thresh = 0.05 or 0.01?
            #     if all(check_conditional_independence_L(features[j], xk, features[i], thresh=0.01) for j in range(d) if ((i, j) in PRS)):
            #         flag = False
            #         # If the above condition holds, i cannot be a root, so we stop immediately
            #         break
            # if flag == True:
            #     pi_H[i] = 1


            # this is the regression-based approach
            # dependents = [j for j in range(d) if j != i and j in ind[i]]
            # try adding not VS as a condition
            dependents = [j for j in range(d) if j != i and j in ind[i] and j not in VS]
            if len(dependents) == 0:
                pi_H[i] = 1
                continue
            flag = True
            for xj in dependents:
                # thresh = 0.05 or 0.01?
                for xk in range(d):
                    if xj != i and xj != xk and (xj, xk) in PRS:
                        # Concatenate the reshaped features horizontally to form a 2D array
                        combined_features = np.hstack([features[i].reshape(-1, 1), features[xj].reshape(-1, 1)])
                        # Pass the concatenated 2D array to the function
                        xk_residual, mi_sum = calculate_residual_L(features[xk], combined_features)
                        # xj_residual = calculate_residual_UC(features[xj], [features[i].reshape(-1, 1), features[xk].reshape(-1, 1)])
                        if check_independence_L(features[xj], xk_residual, thresh=0.01) == False:
                            # If the above condition holds, i cannot be a root, so we stop immediately
                            flag = False
                            break
                if not flag:
                    break
            if flag == True:
                pi_H[i] = 1




    except Exception as e:
        print("error")
        print(e)
    roots = [i for i in range(d) if i in pi_H and pi_H[i] == 1]
    # print(roots)
    # Need to do something if roots are empty
    if roots == []:
        print("empty roots")
        # Select vertex that is maximally independent of other vertices (do this for all the non MRDs)
        if pot_roots_mi_sum:
            roots = [min(pot_roots_mi_sum, key = pot_roots_mi_sum.get)]
        # If no pot_roots were detected, do the same operation but for all of the variables
        else:
            vertices_mi_sum = defaultdict(int)
            for i in range(d):
                for j in range(d):
                    j_residual, mi_sum = calculate_residual_L(features[j], features[i].reshape(-1, 1))
                    vertices_mi_sum[i] += mi_sum
            roots = [min(vertices_mi_sum, key = vertices_mi_sum.get)]

    return roots

def marg_dep_L(data, alpha=0.01):
    d = data.shape[1]
    ind_collection = [[] for _ in range(d)]
    for i in range(d):
        for j in range(i + 1, d):
            if hsic_test(data, i, j, [])['p_value'] < alpha:
                ind_collection[i].append(j)
                ind_collection[j].append(i)
    return ind_collection

def check_v_structure_unsorted_L(i, dependent_triplets, unsorted_list):
    for triplet in dependent_triplets:
        if triplet[0] == i and (triplet[1] in unsorted_list or triplet[2] in unsorted_list):
            return True
    return False


def linearity_check_L(residuals_dict, alpha=0.01):
    keys = list(residuals_dict.keys())
    non_linear_keys = []

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            key_i = keys[i]
            key_j = keys[j]
            
            residual_i = residuals_dict[key_i]
            residual_j = residuals_dict[key_j]
            
            # Regress residual_i onto residual_j
            model = LinearRegression()
            # model = Ridge(alpha=.0001)
            reg_ij = model.fit(residual_j.reshape(-1, 1), residual_i)
            residual_i_given_j = residual_i - reg_ij.predict(residual_j.reshape(-1, 1))
            ind_j = check_independence_L(residual_i_given_j, residual_j, alpha)
            #ind_j = check_independence_pvalue(residual_i_given_j, residual_j)
            
            # Regress residual_j onto residual_i
            # model = LinearRegression()
            # for dense models
            # model = Ridge(alpha=.0001)
            reg_ji = model.fit(residual_i.reshape(-1, 1), residual_j)
            residual_j_given_i = residual_j - reg_ji.predict(residual_i.reshape(-1, 1))
            ind_i = check_independence_L(residual_j_given_i, residual_i, alpha)
            #ind_i = check_independence_pvalue(residual_j_given_i, residual_i)

            # Check independence (not pvalue)
            if ind_j and not ind_i:
                non_linear_keys.append(key_i)
            if ind_i and not ind_j:
                non_linear_keys.append(key_j)
            
    
    return non_linear_keys

def nonlinear_sort_L(sorted_list, unsorted_list, ind, data, v_structures):
    while unsorted_list:
        # Store Residuals
        residual_storage = {}
        measures = np.full(data.shape[1], np.inf)
        for x in unsorted_list:
            anc_x = ind[x]
            features = list(set(anc_x) & set(sorted_list))
            if not features:
                # If no features are found, set measure to a high value (indicating low priority)
                measures[x] = np.inf
                continue
            X = np.array([data[:, y] for y in features]).T
            y = np.array(data[:, x])
            # krr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.01)
            # krr = KernelRidge(kernel='polynomial', alpha=0.1, degree=8, coef0=1)
#             krr = xgb.XGBRegressor(
#     n_estimators=50,           # Fewer trees to prevent overfitting
#     learning_rate=0.1,         # Lower learning rate for stability
#     max_depth=3,               # Shallower trees to avoid overfitting
#     min_child_weight=5,        # Minimum sum of weights in a child node (regularization)
#     subsample=0.8,             # Randomly sample 80% of rows per tree
#     colsample_bytree=0.8,      # Randomly sample 80% of features per tree
#     reg_alpha=1.0,             # L1 regularization (lasso)
#     reg_lambda=1.0,            # L2 regularization (ridge)
#     random_state=42            # For reproducibility
# )
            krr = RandomForestRegressor(
    n_estimators=100,        # Fewer trees suffice for small datasets
    max_depth=10,            # Limit depth to avoid overfitting
    min_samples_split=10,    # Require more samples to split nodes (regularization)
    min_samples_leaf=5,      # Ensure leaf nodes have at least 5 samples
    max_features="sqrt",     # Use a subset of features to make trees diverse
    random_state=42          # For reproducibility
)
#             krr = RandomForestRegressor(
#     n_estimators=200,        # Increase the number of trees for better stability
#     max_depth=20,            # Allow deeper trees but prevent overfitting
#     min_samples_split=5,     # Looser regularization compared to smaller datasets
#     min_samples_leaf=2,      # Smaller leaf size allows more granularity
#     max_features="sqrt"     # Subset of features for diversity
# )
#             krr = RandomForestRegressor(
#     n_estimators=300,        # Increase trees to stabilize predictions
#     max_depth=None,          # Let trees grow fully (Random Forests reduce overfitting via averaging)
#     min_samples_split=2,     # Default splitting threshold
#     min_samples_leaf=1,      # Default for capturing as much information as possible
#     max_features="sqrt",     # Default for random forests
#     random_state=42          # For reproducibility
# )

            # krr = KernelRidge(kernel='polynomial', alpha=0.1, degree=8, coef0=1)
            krr.fit(X, y)
            y_pred = krr.predict(X)
            residuals = y - y_pred
            
            # this block utilizes lin reg too - can remove
            loss_nonlinear = mean_squared_error(y, y_pred)
            lin_reg = LinearRegression()
            # lin_reg = Ridge(alpha = .0001)
            lin_reg.fit(X, y)
            y_pred_linear = lin_reg.predict(X)
            loss_linear = mean_squared_error(y, y_pred_linear)
            if loss_nonlinear > loss_linear:
                residuals = y - y_pred_linear


            # Store Residuals
            residual_storage[x] = residuals
            mi_values = []
            for y in features:
                mi = ee.mi(data[:, y], residuals)
                # this is original command
                mi_values.append(max(0, mi))
                # mi_values.append(abs(mi))
            measures[x] = np.mean(mi_values)
        
        # Check for linearity between residuals for measures not equal to np.inf
        linear_extension = linearity_check_L(residual_storage)
        
        #Set Residuals with linear effects equal to np.inf
        for index in linear_extension:
            measures[index] = np.inf


        # Check if all measures are np.inf
        if np.all(measures == np.inf):
            # If all measures are np.inf, randomly select an element from unsorted_list
            min_index = np.random.choice(unsorted_list)
        else:
            min_index = np.argmin(measures)
        
        sorted_list.append(min_index)
        unsorted_list.remove(min_index)
    return sorted_list

def TDLHD_sort(data, true_roots):
    """
    Nonlinear Hierarchical Topological Sort (NHTS) function.
    
    Parameters:
    data (np.array): Dataset with d variables as columns and n samples as rows.
    
    Returns:
    list: Topological ordering of the variables.
    """
    ind = marg_dep_L(data)
    v_structures = find_v_structures_L(ind)
    # OG Stuff
    # roots = hierarchical_topological_sort_L(data.T, ind, v_structures)
    roots = true_roots
    # NHTS root procedure
    # roots = hierarchical_topological_sort(data.T, ind)
    # print(roots)
    # real_roots = deepcopy(roots)
    unsorted = [i for i in range(data.shape[1]) if i not in roots]
    # OG stuff
    output = nonlinear_sort_L(roots, unsorted, ind, data, v_structures)
    # LoSAM Procedure
    # output = nonlinear_sort_new(roots, unsorted, ind, data)
    return output

def TDLHD(data):
    """
    Nonlinear Hierarchical Topological Sort (NHTS) function.

    Parameters:
    data (np.array): Dataset with d variables as columns and n samples as rows.

    Returns:
    list: Topological ordering of the variables.
    """
    ind = marg_dep_L(data)
    v_structures = find_v_structures_L(ind)
    # OG Stuffs
    roots = hierarchical_topological_sort_L(data.T, ind, v_structures)
    # NHTS root procedure
    # roots = hierarchical_topological_sort(data.T, ind)
    # print(roots)
    real_roots = deepcopy(roots)
    unsorted = [i for i in range(data.shape[1]) if i not in roots]
    # OG stuff
    output = nonlinear_sort_L(roots, unsorted, ind, data, v_structures)
    # LoSAM Procedure
    # output = nonlinear_sort_new(roots, unsorted, ind, data)
    # print(output)
    return output, real_roots



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

#experiment with turning up the weights...

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

import networkx as nx

def topological_sort_from_matrix(matrix):
    n = len(matrix)  # Number of nodes
    G = nx.DiGraph()

    # Add edges to the graph based on the adjacency matrix
    for i in range(n):
        for j in range(n):
            if matrix[i][j] == 1:
                G.add_edge(i, j)

    # Perform topological sort
    topo_sort = list(nx.topological_sort(G))
    
    return topo_sort

import networkx as nx

def roots_from_matrix(matrix):
    n = len(matrix)  # Number of nodes
    G = nx.DiGraph()

    # Add edges to the graph based on the adjacency matrix
    for i in range(n):
        for j in range(n):
            if matrix[i][j] == 1:
                G.add_edge(i, j)

    # Find nodes with no incoming edges (roots)
    roots = [node for node, in_degree in G.in_degree() if in_degree == 0]

    return roots


def convert_to_binary(matrix):
    """
    Convert a matrix into a binary matrix where:
    - 1 represents a non-zero value (including non-NaN values).
    - 0 represents a zero value or NaN.
    
    Parameters:
        matrix (numpy.ndarray): Input matrix.
    
    Returns:
        numpy.ndarray: Binary matrix.
    """
    # Replace NaN with 0 before comparison
    matrix = np.nan_to_num(matrix, nan=0)
    # Create binary matrix
    binary_matrix = (matrix != 0).astype(int)
    return binary_matrix



def topological_sort_from_nogam(X):
    """
    Run NoGAM and return the topological sort of variables.
    
    Args:
        X (np.ndarray): Input dataset.

    Returns:
        list: Topological sort of variables.
    """
    nogam = dodiscover.toporder.NoGAM(n_crossval = 2, prune = False)
    df = pd.DataFrame(X)
    context = dodiscover.make_context().variables(data = df).build()
    nogam.learn_graph(df, context)
    NoGAM_sort = [df.columns[i] for i in nogam.order_]
    return NoGAM_sort

def topological_sort_from_score(X):
    """
    Run Score and return the topological sort of variables.
    
    Args:
        X (np.ndarray): Input dataset.

    Returns:
        list: Topological sort of variables.
    """
    score = dodiscover.toporder.SCORE(prune = False)
    df = pd.DataFrame(X)
    context = dodiscover.make_context().variables(data = df).build()
    score.learn_graph(df, context)
    score_sort = [df.columns[i] for i in score.order_]
    return score_sort




def break_cycles(graph):
    """
    Remove cycles from the graph to make it a DAG.
    Args:
        graph (nx.DiGraph): A directed graph that may contain cycles.
    Returns:
        nx.DiGraph: A graph with cycles removed.
    """
    try:
        # Check if the graph contains a cycle
        cycles = list(nx.simple_cycles(graph))
        if cycles:
            print(f"Adascore - Cycles detected: {cycles}")
            for cycle in cycles:
                # Remove one edge from each cycle
                edge_to_remove = (cycle[0], cycle[1])  # Remove the first edge in the cycle
                print(f"Adascore - Removing edge to break cycle: {edge_to_remove}")
                graph.remove_edge(*edge_to_remove)
    except nx.NetworkXNoCycle:
        print("Adascore - No cycles detected.")
    return graph




def topological_sort_from_adascore(X):
    """
    Run Adascore and return the topological sort of variables.
    
    Args:
        X (np.ndarray): Input dataset.

    Returns:
        list: Topological sort of variables.
    """
    algo = SCAMUV(alpha_orientation=.05, alpha_confounded_leaf=.05, alpha_separations=.05, cv = 1)
    graph = algo.fit(pd.DataFrame(X))

    # Break cycles if any
    graph = break_cycles(graph)

    ada_sort = list(nx.topological_sort(graph))
    return ada_sort

def get_CAM_order(X):
    """
    Computes the topological order of variables using the CAM algorithm.
    
    Parameters:
    - X (np.ndarray or pd.DataFrame): Input data (n x d matrix).
    - prune (bool): Whether to prune edges (default: False).
    - splines_degree (int): Degree of spline basis functions (default: 1).
    - n_splines (int): Number of splines (default: 2).
    
    Returns:
    - List of variable indices in topological order if successful.
    - None if the method fails.
    """
    
    df = pd.DataFrame(X)  # Convert to DataFrame if needed
    # cam = dodiscover.toporder.CAM(prune=prune, splines_degree=splines_degree, n_splines=n_splines)
    cam = dodiscover.toporder.CAM(prune=False)
    context = dodiscover.make_context().variables(data=df).build()
    cam.learn_graph(df, context)
    
    return [df.columns[i] for i in cam.order_]  # Return column names in topological order


def acc_meas(true_parents, predicted_parents, d):
    """
    Calculate accuracy, precision, and false negative rate for the predicted parent sets.
    
    Parameters:
    - true_parents_list: List of sets, where each set contains the true parents of a variable.
    - predicted_parents_list: List of sets, where each set contains the predicted parents of a variable.
    
    Returns:
    A dictionary containing accuracy, precision, and FNR.
    """
    true_flat = []
    predicted_flat = []
    
    
    # Create binary vectors for each set of parents
    true_vector = [1 if node in true_parents else 0 for node in range(d)]
    predicted_vector = [1 if node in predicted_parents else 0 for node in range(d)]
    
    true_flat.extend(true_vector)
    predicted_flat.extend(predicted_vector)
    
    # Calculate metrics
    precision = precision_score(true_flat, predicted_flat, zero_division=0)
    accuracy = accuracy_score(true_flat, predicted_flat)
    recall = recall_score(true_flat, predicted_flat, zero_division=0)
    f1 = f1_score(true_flat, predicted_flat, zero_division=0)
    
    return f1, precision, recall




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



from concurrent.futures import ThreadPoolExecutor, as_completed


def run_trial(trial_index, n_subsample, method_names):
    """
    Execute a single trial, generating data and running methods.
    
    Args:
        trial_index (int): Index of the trial.
        n (int): Number of samples.
        d (int): Number of variables.
        avg_edge (float): Average number of edges.

    Returns:
        dict: Results for the trial, including metrics and runtime.
    """
    results = {}
    try:
        # Generate data for this trial
        # To ensure DGM are uniquely generated even for EC2 instance
        seed = int(time.time() * 1e6) % (2**32)  # Use microseconds for higher precision
        np.random.seed(seed)
        # X, adjacency_matrix, true_topological_order, _ = generate_quadratic_data(n, d, avg_edge, dgm, noise)
        # Generate syn data
        import cdt
        import networkx as nx
        data, true_graph = cdt.data.load_dataset('sachs')

        # Convert data to n x d numpy matrix and true_graph to d x d adjacency matrix
        X = data.to_numpy()
        labels = data.columns.tolist()
        adjacency_matrix = nx.to_numpy_array(true_graph, nodelist=labels)

        # Take a subset of Data to test
        # firrst n_subsample samples
        # X = X[:n_subsample, :]
        # random n_subsample samples
        indices = np.random.choice(X.shape[0], n_subsample, replace = False)
        X = X[indices, :]

        # Standardize X
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # print(type(adjacency_matrix))
        # turn back on
        # np.save(os.path.join(new_wd, f"trial_{trial_index}_adj_matrix.npy"), adjacency_matrix)
        # np.save(os.path.join(new_wd, f"trial_{trial_index}_data.npy"), X)
        
        true_roots = roots_from_matrix(adjacency_matrix)
        # Code to check if DGM are uniquely generated
        # import hashlib
        # data_hash = hashlib.sha256(str(X).encode()).hexdigest()
        # print(f"Trial data hash: {data_hash}")
        
        # Initialize per-method metrics
        metrics = {
            "root_f1": {},
            "root_pre": {},
            "root_rec": {},
            "atop": {},
            "SHD": {},
            "F1": {},
            "Precision": {},
            "Recall": {},
            # "SID": {},
            "times": {},
            "matrix_times": {}
        }
        
        # Define all possible methods to Evaluate
        poss_methods = {
                # Ordering Methods
                "TDLHD": lambda: TDLHD(X),
                "NHTS": lambda: NHTS_old(X),
                "DLiNGAM": lambda: lingam.DirectLiNGAM().fit(pd.DataFrame(X)).causal_order_,
                "SCORE": lambda: topological_sort_from_score(X),
                "NoGAM": lambda: topological_sort_from_nogam(X),
                "RESIT": lambda: lingam.RESIT(RandomForestRegressor(max_depth=4)).fit(X).causal_order_,
                "CAM": lambda: get_CAM_order(X), # default is n_spline 10, degree_3
                    # Heuristic Methods
                "R2Sort": lambda: [index for index, _ in sorted(enumerate(r2coeff(X.T)), key=lambda x: x[1], reverse=False)],
                "VarSort":  lambda: [index for index, _ in sorted(enumerate(np.var(X, axis=0)),  key=lambda x: x[1], reverse=False)],
                "RandSort": lambda: list(np.random.permutation(X.shape[1])),
                # Non-Ordering Methods
                # 
                "CAMUV": lambda: topological_sort_from_matrix(lingam.CAMUV().fit(pd.DataFrame(X)).adjacency_matrix_),
                "RCD": lambda: topological_sort_from_matrix(convert_to_binary(lingam.RCD().fit(X).adjacency_matrix_)),
                "GES": lambda: topological_sort_from_matrix(pdag2dag(ges(X)['G']).graph),
                "GRaSP": lambda: topological_sort_from_matrix(pdag2dag(grasp(X)).graph),
                "Adascore": lambda: topological_sort_from_adascore(X),
                # Weird other methods
                "TDLHD_sort": lambda: TDLHD_sort(X, true_roots),
                "NHTS_sort": lambda: NHTS_old_sort(X, true_roots)
            }

        # Select Methods from Possible Methods
        methods = {}
        for method in method_names:
            methods[method] = poss_methods[method]

        
        # Run each method and capture metrics
        for method, func in methods.items():
            try:
                start_time = time.time()
                if method in ["TDLHD_sort", "NHTS_sort"]:
                    sort_order = func()
                    end_time = time.time()
                    metrics["atop"][method] = count_topological_errors(adjacency_matrix, sort_order)
                    matrix = cam_prune_from_order(sort_order, X)
                    matrix_end_time = time.time()
                    # Adj Matrix Evaluation
                    metrics["SHD"][method] = SHD(adjacency_matrix, matrix)
                    # metrics["SID"][method] = SID(adjacency_matrix, matrix)
                    metrics["F1"][method] = f1_score(adjacency_matrix.flatten(), matrix.flatten())
                    metrics["Precision"][method] = precision_score(adjacency_matrix.flatten(), matrix.flatten())
                    metrics["Recall"][method] =  recall_score(adjacency_matrix.flatten(), matrix.flatten())
                    
                elif method in ["TDLHD", "NHTS", "LoSAMUC"]:
                    # Run Method
                    sort_order, roots = func()
                    end_time = time.time()
                    matrix = cam_prune_from_order(sort_order, X)
                    # print(matrix)
                    matrix_end_time = time.time()
                    # print("")
                    # print(method, "PR", roots)
                    # print(method, "TR", roots_from_matrix(adjacency_matrix))
                    # Sort Evaluation
                    metrics["atop"][method] = count_topological_errors(adjacency_matrix, sort_order)
                    # Root Evaluation
                    f1, precision, recall = acc_meas(roots_from_matrix(adjacency_matrix), roots, adjacency_matrix.shape[1])
                    metrics["root_f1"][method] = f1
                    metrics["root_pre"][method] = precision
                    metrics["root_rec"][method] = recall
                    # Adj Matrix Evaluation
                    metrics["SHD"][method] = SHD(adjacency_matrix, matrix)
                    # metrics["SID"][method] = SID(adjacency_matrix, matrix)
                    metrics["F1"][method] = f1_score(adjacency_matrix.flatten(), matrix.flatten())
                    metrics["Precision"][method] = precision_score(adjacency_matrix.flatten(), matrix.flatten())
                    metrics["Recall"][method] =  recall_score(adjacency_matrix.flatten(), matrix.flatten())
                    
                else:
                    # Sort Evaluation
                    sort_order = func()
                    end_time = time.time()
                    matrix = cam_prune_from_order(sort_order, X)
                    matrix_end_time = time.time()
                    metrics["atop"][method] = count_topological_errors(adjacency_matrix, sort_order)
                    # Adj Matrix Evaluation
                    metrics["SHD"][method] = SHD(adjacency_matrix, matrix)
                    # metrics["SID"][method] = SID(adjacency_matrix, matrix)
                    metrics["F1"][method] = f1_score(adjacency_matrix.flatten(), matrix.flatten())
                    metrics["Precision"][method] = precision_score(adjacency_matrix.flatten(), matrix.flatten())
                    metrics["Recall"][method] =  recall_score(adjacency_matrix.flatten(), matrix.flatten())


                  
                metrics["times"][method] = end_time - start_time
                metrics["matrix_times"][method] = matrix_end_time - start_time

                #turn back on
                # if sort_order is not None:
                #     np.save(
                #         os.path.join(new_wd, f"trial_{trial_index}_{method}_sort_order.npy"),
                #         sort_order
                #     )

            except Exception as e:
                metrics["atop"][method] = None  # Indicate failure
                metrics["root_f1"][method] = None
                metrics["root_pre"][method] = None
                metrics["root_rec"][method] = None
                metrics["times"][method] = None
                print(f"Error in method {method}, trial {trial_index}: {e}")
        
        results = metrics
    except Exception as e:
        print(f"Error in trial {trial_index}: {e}")
        results["error"] = str(e)
    
    return results


import os
import concurrent.futures
from tqdm import tqdm
import numpy as np
# turn back on when on ec2
import boto3
import datetime

# S3 Setup
s3 = boto3.client('s3')

# Function to upload to S3
def upload_to_s3(file_path, s3_key):
    try:
        s3.upload_file(file_path, bucket_name, s3_key)
        print(f"✅ Uploaded {s3_key} to S3 successfully!")
    except Exception as e:
        print(f"❌ Failed to upload {s3_key}: {e}")

# Main execution
if __name__ == "__main__":

    # File Path Save
    new_wd = "/home/ec2-user/"
    # total dataset is 
    n_subsample = 300
    num_blocks = 5
    trials = 6

    # Method Names
    method_names = ["TDLHD", "NHTS", "DLiNGAM", "RESIT", "SCORE", "NoGAM", "CAM", "RandSort", "VarSort", "R2Sort"]
    # method_names = ["TDLHD"]
    # method_names = ["DLiNGAM", "RandSort", "VarSort", "R2Sort"]

    # Initialize aggregated metrics
    aggregated_metrics = {
        "atop": {method: [] for method in method_names},
        "times": {method: [] for method in method_names},
        "root_f1": {method: [] for method in method_names},
        "root_rec": {method: [] for method in method_names},
        "root_pre": {method: [] for method in method_names},
        "SHD": {method: [] for method in method_names},
        "F1": {method: [] for method in method_names},
        "Precision": {method: [] for method in method_names},
        "Recall": {method: [] for method in method_names},
        "matrix_times": {method: [] for method in method_names}
    }

    # Run the experiment block 3 consecutive times
    for run in range(num_blocks):
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(run_trial, i, n_subsample, method_names) for i in range(trials)]

            all_results = []
            for future in tqdm(concurrent.futures.as_completed(futures), desc=f"Run {run + 1}"):
                all_results.append(future.result())

        # Aggregate results
        for result in all_results:
            if "error" in result:
                continue  # Skip trials that failed entirely
            for method in method_names:
                if result.get("atop", {}).get(method) is not None:
                    aggregated_metrics["atop"][method].append(result["atop"][method])
                if method in ["TDLHD", "NHTS", "LoSAMUC"]:
                    if result.get("root_f1", {}).get(method) is not None:
                        aggregated_metrics["root_f1"][method].append(result["root_f1"][method])
                    if result.get("root_pre", {}).get(method) is not None:
                        aggregated_metrics["root_pre"][method].append(result["root_pre"][method])
                    if result.get("root_rec", {}).get(method) is not None:
                        aggregated_metrics["root_rec"][method].append(result["root_rec"][method])
                if result.get("times", {}).get(method) is not None:
                    aggregated_metrics["times"][method].append(result["times"][method])
                if result.get("SHD", {}).get(method) is not None:
                    aggregated_metrics["SHD"][method].append(result["SHD"][method])
                if result.get("F1", {}).get(method) is not None:
                    aggregated_metrics["F1"][method].append(result["F1"][method])
                if result.get("Precision", {}).get(method) is not None:
                    aggregated_metrics["Precision"][method].append(result["Precision"][method])
                if result.get("Recall", {}).get(method) is not None:
                    aggregated_metrics["Recall"][method].append(result["Recall"][method])
                if result.get("matrix_times", {}).get(method) is not None:
                    aggregated_metrics["matrix_times"][method].append(result["matrix_times"][method])

    for method in method_names:
        median_error = np.median(aggregated_metrics["atop"][method]) if aggregated_metrics["atop"][method] else None
        mean_time = np.mean(aggregated_metrics["times"][method]) if aggregated_metrics["times"][method] else None
        median_SHD = np.median(aggregated_metrics["SHD"][method]) if aggregated_metrics["SHD"][method] else None
        median_F1 = np.median(aggregated_metrics["F1"][method]) if aggregated_metrics["F1"][method] else None
        median_Precision = np.median(aggregated_metrics["Precision"][method]) if aggregated_metrics["Precision"][method] else None
        median_Recall = np.median(aggregated_metrics["Recall"][method]) if aggregated_metrics["Recall"][method] else None
        mean_matrix_time = np.mean(aggregated_metrics["matrix_times"][method]) if aggregated_metrics["matrix_times"][method] else None

        print(f"{method}: Median Atop = {median_error}, Median SHD = {median_SHD}, Median F1 = {median_F1}, Median Precision = {median_Precision}, Median Recall = {median_Recall}, Mean Runtime = {mean_time:.2f}s, Mean Matrix Time = {mean_matrix_time:.2f}s")

    # Saving and Uploading to S3
    # lin_prop_folder = f"linear_proportion_{lin_prop}"  # Folder for each linear proportion
    lin_prop_folder = f"rando_standard_sachs_exp_{n_subsample}"

    for method in method_names:
        metrics_to_save = ["atop", "times", "root_f1", "root_pre", "root_rec", "SHD", "F1", "Precision", "Recall", "matrix_times"]
        for metric in metrics_to_save:
            if aggregated_metrics[metric][method]:
                file_path = f"{new_wd}{method}_{metric}.npy"
                np.save(file_path, aggregated_metrics[metric][method])
                upload_to_s3(file_path, f"{lin_prop_folder}/{method}_{metric}.npy")


# print(data_matrix[0:5,:])
# print(sum(sum(true_adj_matrix)))
