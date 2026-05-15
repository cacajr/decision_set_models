import sys, os
if not sys.path[0] == os.path.abspath('.'):
    sys.path.insert(0, os.path.abspath('.'))

from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from itertools import product
from tqdm import tqdm
from joblib import Parallel, delayed
import time
from utils.functions import tree_to_dnf
from utils.binarize import Binarize


''' - Dataset informations '''
database_names = [
    'lung_cancer', 
    'iris', 
    'parkinsons',
    'ionosphere',
    'wdbc', 
    'transfusion', 
    'pima', 
    'titanic', 
    'depressed', 
    'mushroom', 
    'twitter'
]
categorical_columns_indexes = [
    [0, 1], 
    [], 
    [],
    [0],
    [], 
    [],
    [0], 
    [0, 2, 3, 5], 
    [6], 
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    [42, 43, 44, 45, 46, 47, 48]
]

''' - Model configurations '''
seed = 21

max_depths = [1, 2, 3, 4]
max_leaf_nodes_list = [1, 2, 3, 4]
number_quantiles_ordinal_columns_list = [5, 10, 15, 20]

n_jobs = -1  # Define number of parallel jobs for evaluations. -1 uses all available cores.


def evaluate_config_cv(cfg, X_train_full, y_train_full, 
                       n_splits, n_repeats, seed):
    """Evaluate a single configuration for Decision Tree"""
    # X_train_full and y_train_full are already binarized numpy arrays
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    
    accuracies = []
    times = []
    n_nodes = []
    depths = []
    num_rules_list = []
    larger_rule_list = []
    sum_literals_list = []
    
    # Create feature names based on binarized data shape
    feature_names = [f'feature_{i}' for i in range(X_train_full.shape[1])]
    
    for train_idx, val_idx in rskf.split(X_train_full, y_train_full):
        Xtr, Xval = X_train_full[train_idx], X_train_full[val_idx]
        ytr, yval = y_train_full[train_idx], y_train_full[val_idx]
        
        try:
            model = DecisionTreeClassifier(
                max_depth=cfg['max_depth'],
                max_leaf_nodes=cfg['max_leaf_nodes'],
                random_state=seed
            )
            
            # Measure training time
            start_time = time.time()
            model.fit(Xtr, ytr)
            train_time = time.time() - start_time
            
            # Prediction and accuracy
            accuracy = model.score(Xval, yval)
            
            # Calculate DNF metrics
            dnf_clauses = tree_to_dnf(model, feature_names=feature_names)
            num_rules = len(dnf_clauses)
            larger_rule = max(len(clause) for clause in dnf_clauses) if dnf_clauses else 0
            sum_literals = sum(len(clause) for clause in dnf_clauses)
            
            accuracies.append(accuracy)
            times.append(train_time)
            n_nodes.append(model.tree_.node_count)
            depths.append(model.get_depth())
            num_rules_list.append(num_rules)
            larger_rule_list.append(larger_rule)
            sum_literals_list.append(sum_literals)
            
        except Exception as e:
            print(f"Config failed during fold evaluation: {cfg} -- {str(e)[:200]}")
            return None
    
    # Check if any results were collected
    if not accuracies:
        print(f"Config produced no valid folds: {cfg}")
        return None
    
    # Build result dictionary
    res = {
        **cfg,
        "cv_mean_accuracy": float(np.mean(accuracies)),
        "cv_std_accuracy": float(np.std(accuracies, ddof=1)),
        "cv_mean_time": float(np.mean(times)),
        "cv_std_time": float(np.std(times, ddof=1)),
        "cv_mean_n_nodes": float(np.mean(n_nodes)),
        "cv_std_n_nodes": float(np.std(n_nodes, ddof=1)),
        "cv_mean_depth": float(np.mean(depths)),
        "cv_std_depth": float(np.std(depths, ddof=1)),
        "cv_mean_num_rules": float(np.mean(num_rules_list)),
        "cv_std_num_rules": float(np.std(num_rules_list, ddof=1)),
        "cv_mean_larger_rule": float(np.mean(larger_rule_list)),
        "cv_std_larger_rule": float(np.std(larger_rule_list, ddof=1)),
        "cv_mean_sum_literals": float(np.mean(sum_literals_list)),
        "cv_std_sum_literals": float(np.std(sum_literals_list, ddof=1)),
    }
    
    return res


def apply_binarization_to_dataset(X, y, categorical_columns_index, number_quantiles, seed):
    """Apply binarization to dataset and return binarized X and y"""
    binarizer = Binarize(
        data_frame=X,
        series=y,
        multiclass=False,
        categorical_columns_index=categorical_columns_index,
        number_quantiles_ordinal_columns=number_quantiles,
        number_partitions=1,
        balance_instances=True,
        balance_instances_seed=seed
    )
    
    X_binarized = binarizer.get_normal_instances(partition=1)
    y_binarized = binarizer.get_classes()[0]
    
    return X_binarized, y_binarized


for database_name, categorical_columns_index in zip(database_names, categorical_columns_indexes):
    print(f'\n--- Database: {database_name} ---')
    dt_path = f'./drafts/hyperparameters/decision_tree_datasets_results/{database_name}.csv'

    # Load dataset
    Xy = pd.read_csv(f'./databases/{database_name}.csv')
    X = Xy.drop(['Class'], axis=1)
    y = Xy['Class']

    # Search settings (tunable)
    n_splits = 5
    n_repeats = 3

    # Build full grid with all hyperparameter combinations
    all_configs = []
    for max_depth, max_leaf_nodes, number_quantiles in product(max_depths, max_leaf_nodes_list, number_quantiles_ordinal_columns_list):
        cfg = {
            "max_depth": max_depth,
            "max_leaf_nodes": max_leaf_nodes,
            "number_quantiles_ordinal_columns": number_quantiles,
        }
        all_configs.append(cfg)

    configs = all_configs
    print(f'Total configurations to test: {len(configs)}')

    # Group configs by number_quantiles to efficiently binarize
    configs_by_quantiles = {}
    for cfg in configs:
        nq = cfg['number_quantiles_ordinal_columns']
        if nq not in configs_by_quantiles:
            configs_by_quantiles[nq] = []
        configs_by_quantiles[nq].append(cfg)
    
    # Store results for all quantile values
    dt_results_all = []
    
    # Process each quantile value separately
    for number_quantiles in sorted(configs_by_quantiles.keys()):
        print(f'  Applying Binarization with number_quantiles={number_quantiles}...')
        
        # Apply binarization for this quantile value
        X_binarized, y_binarized = apply_binarization_to_dataset(
            X, y, categorical_columns_index, number_quantiles, seed
        )
        
        # Create an outer hold-out split for final evaluation (30%)
        X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
            X_binarized, y_binarized, test_size=0.3, random_state=seed, stratify=y_binarized
        )

        # Get configs for this quantile value
        configs_this_nq = configs_by_quantiles[number_quantiles]
        print(f'  Evaluating {len(configs_this_nq)} configurations in parallel (nq={number_quantiles})...')
        
        # Parallel evaluation of configurations
        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_config_cv)(
                cfg, X_train_full, y_train_full,
                n_splits, n_repeats, seed
            ) for cfg in tqdm(configs_this_nq, desc=f'CV Configs (nq={number_quantiles})', position=1, leave=False)
        )
        
        for res in results:
            if res is not None:
                dt_results_all.append(res)

    # Save full grid results for inspection
    dt_results_df = pd.DataFrame(dt_results_all)

    if not dt_results_df.empty:
        dt_results_df.to_csv(dt_path.replace('.csv', '_grid_results.csv'), index=False)

        # Select top 10 by CV mean accuracy
        dt_top10 = dt_results_df.sort_values(by='cv_mean_accuracy', ascending=False).head(10).copy()

        # Refit top10 on the full training set and evaluate on the holdout set
        dt_holdout_metrics = []
        
        for (_, dt_r) in dt_top10.iterrows():
            dt_cfg = dt_r.to_dict()
            number_quantiles = int(dt_cfg['number_quantiles_ordinal_columns'])
            
            # Apply binarization with the same quantiles used during CV
            X_binarized, y_binarized = apply_binarization_to_dataset(
                X, y, categorical_columns_index, number_quantiles, seed
            )
            
            # Create holdout split
            X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
                X_binarized, y_binarized, test_size=0.3, random_state=seed, stratify=y_binarized
            )

            model = DecisionTreeClassifier(
                max_depth=int(dt_cfg['max_depth']),
                max_leaf_nodes=int(dt_cfg['max_leaf_nodes']),
                random_state=seed
            )

            # Measure training time
            start_time = time.time()
            model.fit(X_train_full, y_train_full)
            holdout_train_time = time.time() - start_time

            # Measure prediction time
            start_time = time.time()
            holdout_acc = model.score(X_holdout, y_holdout)
            holdout_pred_time = time.time() - start_time

            # Calculate DNF metrics for holdout
            feature_names = [f'feature_{i}' for i in range(X_train_full.shape[1])]
            dnf_clauses = tree_to_dnf(model, feature_names=feature_names)
            holdout_num_rules = len(dnf_clauses)
            holdout_larger_rule = max(len(clause) for clause in dnf_clauses) if dnf_clauses else 0
            holdout_sum_literals = sum(len(clause) for clause in dnf_clauses)

            dt_holdout_metrics.append({
                'holdout_accuracy': float(holdout_acc),
                'holdout_train_time': float(holdout_train_time),
                'holdout_pred_time': float(holdout_pred_time),
                'holdout_n_nodes': int(model.tree_.node_count),
                'holdout_depth': int(model.get_depth()),
                'holdout_num_rules': int(holdout_num_rules),
                'holdout_larger_rule': int(holdout_larger_rule),
                'holdout_sum_literals': int(holdout_sum_literals),
            })

        dt_holdout_df = pd.DataFrame(dt_holdout_metrics)

        dt_top10.reset_index(drop=True, inplace=True)
        
        dt_top10 = pd.concat([dt_top10.reset_index(drop=True), dt_holdout_df], axis=1)

        # Reorder columns (make sure config columns first)
        col_order = [
            'max_depth', 'max_leaf_nodes', 'number_quantiles_ordinal_columns',
            'cv_mean_accuracy', 'cv_std_accuracy', 'cv_mean_time', 'cv_std_time',
            'cv_mean_n_nodes', 'cv_std_n_nodes', 'cv_mean_depth', 'cv_std_depth',
            'cv_mean_num_rules', 'cv_std_num_rules', 'cv_mean_larger_rule', 'cv_std_larger_rule',
            'cv_mean_sum_literals', 'cv_std_sum_literals',
            'holdout_accuracy', 'holdout_train_time', 'holdout_pred_time', 'holdout_n_nodes', 'holdout_depth',
            'holdout_num_rules', 'holdout_larger_rule', 'holdout_sum_literals'
        ]

        cols = [c for c in col_order if c in dt_top10.columns]
        dt_top10 = dt_top10[cols]

        # Save the top 10 results as requested
        dt_top10.to_csv(dt_path, index=False)
        print(f"Saved top 10 configs for dataset {database_name} to {dt_path}")
    else:
        print(f"No successful configurations for dataset {database_name}.")
