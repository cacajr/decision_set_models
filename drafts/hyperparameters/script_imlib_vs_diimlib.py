import sys, os
if not sys.path[0] == os.path.abspath('.'):
    sys.path.insert(0, os.path.abspath('.'))

from models.imlib import IMLIB
from models.di_imlib import DI_IMLIB

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from itertools import product
from tqdm import tqdm
from joblib import Parallel, delayed
import time


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

number_lines_per_partition = [4, 8, 16, 32]
max_rule_set_sizes = [1, 2, 3, 4]
max_sizes_each_rule = [1, 2, 3, 4]
rules_size_weight = [1, 2, 5, 10]
rules_accuracy_weights = [1, 2, 5, 10]
number_quantiles_ordinal_columns = [5, 10, 15, 20]
balance_instances = True
balance_instances_seed = seed

n_jobs = -1  # Define number of parallel jobs for evaluations. -1 uses all available cores.


def evaluate_config_cv(cfg, X_train_full, y_train_full, categorical_columns_index, 
                       n_splits, n_repeats, seed, balance_instances, balance_instances_seed):
    """Evaluate a single configuration for both IMLIB and DI-IMLIB"""
    
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    
    imlib_accuracies, diimlib_accuracies = [], []
    imlib_times, diimlib_times = [], []
    imlib_rules_counts, diimlib_rules_counts = [], []
    imlib_max_rules_sizes, diimlib_max_rules_sizes = [], []
    imlib_sum_rules_sizes, diimlib_sum_rules_sizes = [], []
    
    for train_idx, val_idx in rskf.split(X_train_full, y_train_full):
        Xtr, Xval = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        ytr, yval = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
        
        try:
            imlib_model = IMLIB(
                max_rule_set_size=cfg['max_rule_set_size'],
                max_size_each_rule=cfg['max_size_each_rule'],
                rules_size_weight=cfg['rules_size_weight'],
                rules_accuracy_weight=cfg['rules_accuracy_weight'],
                categorical_columns_index=categorical_columns_index,
                number_quantiles_ordinal_columns=cfg['number_quantiles_ordinal'],
                number_lines_per_partition=cfg['number_lines_per_partition'],
                balance_instances=balance_instances,
                balance_instances_seed=balance_instances_seed
            )
            diimlib_model = DI_IMLIB(
                max_rule_set_size=cfg['max_rule_set_size'],
                max_size_each_rule=cfg['max_size_each_rule'],
                rules_size_weight=cfg['rules_size_weight'],
                rules_accuracy_weight=cfg['rules_accuracy_weight'],
                categorical_columns_index=categorical_columns_index,
                number_quantiles_ordinal_columns=cfg['number_quantiles_ordinal'],
                number_lines_per_partition=cfg['number_lines_per_partition'],
                balance_instances=balance_instances,
                balance_instances_seed=balance_instances_seed
            )
            
            imlib_model.fit(Xtr, ytr)
            diimlib_model.fit(Xtr, ytr)
            
            imlib_accuracies.append(imlib_model.score(Xval, yval))
            diimlib_accuracies.append(diimlib_model.score(Xval, yval))
            
            imlib_times.append(imlib_model.get_total_time_solver_solutions())
            diimlib_times.append(diimlib_model.get_total_time_solver_solutions())
            
            imlib_rules_counts.append(imlib_model.get_rule_set_size())
            diimlib_rules_counts.append(diimlib_model.get_rule_set_size())
            
            imlib_max_rules_sizes.append(imlib_model.get_larger_rule_size())
            diimlib_max_rules_sizes.append(diimlib_model.get_larger_rule_size())
            
            imlib_sum_rules_sizes.append(imlib_model.get_sum_rules_size())
            diimlib_sum_rules_sizes.append(diimlib_model.get_sum_rules_size())
            
        except Exception as e:
            print(f"Config failed during fold evaluation: {cfg} -- {str(e)[:200]}")
            return None, None
    
    # Check if any results were collected
    if not imlib_accuracies:
        print(f"Config produced no valid folds: {cfg}")
        return None, None
    
    # Build result dictionaries
    imlib_res = {
        **cfg,
        "cv_mean_accuracy": float(np.mean(imlib_accuracies)),
        "cv_std_accuracy": float(np.std(imlib_accuracies, ddof=1)),
        "cv_mean_time": float(np.mean(imlib_times)),
        "cv_std_time": float(np.std(imlib_times, ddof=1)),
        "cv_mean_num_rules": float(np.mean(imlib_rules_counts)),
        "cv_std_num_rules": float(np.std(imlib_rules_counts, ddof=1)),
        "cv_mean_larger_rule": float(np.mean(imlib_max_rules_sizes)),
        "cv_std_larger_rule": float(np.std(imlib_max_rules_sizes, ddof=1)),
        "cv_mean_sum_literals": float(np.mean(imlib_sum_rules_sizes)),
        "cv_std_sum_literals": float(np.std(imlib_sum_rules_sizes, ddof=1)),
    }
    
    diimlib_res = {
        **cfg,
        "cv_mean_accuracy": float(np.mean(diimlib_accuracies)),
        "cv_std_accuracy": float(np.std(diimlib_accuracies, ddof=1)),
        "cv_mean_time": float(np.mean(diimlib_times)),
        "cv_std_time": float(np.std(diimlib_times, ddof=1)),
        "cv_mean_num_rules": float(np.mean(diimlib_rules_counts)),
        "cv_std_num_rules": float(np.std(diimlib_rules_counts, ddof=1)),
        "cv_mean_larger_rule": float(np.mean(diimlib_max_rules_sizes)),
        "cv_std_larger_rule": float(np.std(diimlib_max_rules_sizes, ddof=1)),
        "cv_mean_sum_literals": float(np.mean(diimlib_sum_rules_sizes)),
        "cv_std_sum_literals": float(np.std(diimlib_sum_rules_sizes, ddof=1)),
    }
    
    return imlib_res, diimlib_res


for database_name, categorical_columns_index in zip(database_names, categorical_columns_indexes):
    print(f'\n--- Database: {database_name} ---')
    imlib_path = f'./drafts/hyperparameters/imlib_datasets_results/{database_name}.csv'
    diimlib_path = f'./drafts/hyperparameters/diimlib_datasets_results/{database_name}.csv'

    # Load dataset
    Xy = pd.read_csv(f'./databases/{database_name}.csv')
    X = Xy.drop(['Class'], axis=1)
    y = Xy['Class']

    # Create an outer hold-out split for final evaluation (30%) and use the training part to search hyperparams
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    # Search settings (tunable)
    n_splits = 5
    n_repeats = 3
    max_evals = 200  # if None -> exhaustive grid; otherwise sample up to this many random configs

    # Build full grid
    all_configs = []
    for n_lines_partition, max_rule_set_size, max_size_each_rule, rules_size_weight_value, rules_accuracy_weight_value, n_quantiles_ordinal in product(
        number_lines_per_partition,
        max_rule_set_sizes,
        max_sizes_each_rule,
        rules_size_weight,
        rules_accuracy_weights,
        number_quantiles_ordinal_columns,
    ):
        cfg = {
            "number_lines_per_partition": n_lines_partition,
            "max_rule_set_size": max_rule_set_size,
            "max_size_each_rule": max_size_each_rule,
            "rules_size_weight": rules_size_weight_value,
            "rules_accuracy_weight": rules_accuracy_weight_value,
            "number_quantiles_ordinal": n_quantiles_ordinal,
        }
        all_configs.append(cfg)

    # Optionally subsample configs to reduce computation
    rng = np.random.RandomState(seed)
    if max_evals is not None and len(all_configs) > max_evals:
        sampled_idx = rng.choice(len(all_configs), size=max_evals, replace=False)
        configs = [all_configs[i] for i in sampled_idx]
    else:
        configs = all_configs

    # Parallel evaluation of configurations
    print(f'  Evaluating {len(configs)} configurations in parallel...')
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_config_cv)(
            cfg, X_train_full, y_train_full, categorical_columns_index,
            n_splits, n_repeats, seed, balance_instances, balance_instances_seed
        ) for cfg in tqdm(configs, desc='CV Configs', position=1, leave=False)
    )

    imlib_results = []
    diimlib_results = []
    
    for imlib_res, diimlib_res in results:
        if imlib_res is not None and diimlib_res is not None:
            imlib_results.append(imlib_res)
            diimlib_results.append(diimlib_res)

    # Save full grid results for inspection
    imlib_results_df = pd.DataFrame(imlib_results)
    diimlib_results_df = pd.DataFrame(diimlib_results)

    if not imlib_results_df.empty and not diimlib_results_df.empty:
        imlib_results_df.to_csv(imlib_path.replace('.csv', '_grid_results.csv'), index=False)
        diimlib_results_df.to_csv(diimlib_path.replace('.csv', '_grid_results.csv'), index=False)

        # Select top 10 by CV mean accuracy
        imlib_top10 = imlib_results_df.sort_values(by='cv_mean_accuracy', ascending=False).head(10).copy()
        diimlib_top10 = diimlib_results_df.sort_values(by='cv_mean_accuracy', ascending=False).head(10).copy()

        # Refit top10 on the full training set and evaluate on the holdout set
        imlib_holdout_metrics = []
        diimlib_holdout_metrics = []
        
        for (_, imlib_r), (_, diimlib_r) in zip(imlib_top10.iterrows(), diimlib_top10.iterrows()):
            imlib_cfg = imlib_r.to_dict()
            diimlib_cfg = diimlib_r.to_dict()

            imlib_model = IMLIB(
                max_rule_set_size=int(imlib_cfg['max_rule_set_size']),
                max_size_each_rule=int(imlib_cfg['max_size_each_rule']),
                rules_size_weight=int(imlib_cfg['rules_size_weight']),
                rules_accuracy_weight=int(imlib_cfg['rules_accuracy_weight']),
                categorical_columns_index=categorical_columns_index,
                number_quantiles_ordinal_columns=int(imlib_cfg['number_quantiles_ordinal']),
                number_lines_per_partition=int(imlib_cfg['number_lines_per_partition']),
                balance_instances=balance_instances,
                balance_instances_seed=balance_instances_seed
            )
            diimlib_model = DI_IMLIB(
                max_rule_set_size=int(diimlib_cfg['max_rule_set_size']),
                max_size_each_rule=int(diimlib_cfg['max_size_each_rule']),
                rules_size_weight=int(diimlib_cfg['rules_size_weight']),
                rules_accuracy_weight=int(diimlib_cfg['rules_accuracy_weight']),
                categorical_columns_index=categorical_columns_index,
                number_quantiles_ordinal_columns=int(diimlib_cfg['number_quantiles_ordinal']),
                number_lines_per_partition=int(diimlib_cfg['number_lines_per_partition']),
                balance_instances=balance_instances,
                balance_instances_seed=balance_instances_seed
            )

            imlib_model.fit(X_train_full, y_train_full)
            diimlib_model.fit(X_train_full, y_train_full)

            # Measure training times
            imlib_train_time = imlib_model.get_total_time_solver_solutions()
            diimlib_train_time = diimlib_model.get_total_time_solver_solutions()
            
            # Measure prediction times
            start_time = time.time()
            imlib_holdout_acc = imlib_model.score(X_holdout, y_holdout)
            imlib_pred_time = time.time() - start_time
            
            start_time = time.time()
            diimlib_holdout_acc = diimlib_model.score(X_holdout, y_holdout)
            diimlib_pred_time = time.time() - start_time

            imlib_holdout_metrics.append({
                'holdout_accuracy': float(imlib_holdout_acc),
                'holdout_train_time': float(imlib_train_time),
                'holdout_pred_time': float(imlib_pred_time),
                'holdout_num_rules': int(imlib_model.get_rule_set_size()),
                'holdout_larger_rule': int(imlib_model.get_larger_rule_size()),
                'holdout_sum_literals': int(imlib_model.get_sum_rules_size()),
            })
            diimlib_holdout_metrics.append({
                'holdout_accuracy': float(diimlib_holdout_acc),
                'holdout_train_time': float(diimlib_train_time),
                'holdout_pred_time': float(diimlib_pred_time),
                'holdout_num_rules': int(diimlib_model.get_rule_set_size()),
                'holdout_larger_rule': int(diimlib_model.get_larger_rule_size()),
                'holdout_sum_literals': int(diimlib_model.get_sum_rules_size()),
            })

        imlib_holdout_df = pd.DataFrame(imlib_holdout_metrics)
        diimlib_holdout_df = pd.DataFrame(diimlib_holdout_metrics)

        imlib_top10.reset_index(drop=True, inplace=True)
        diimlib_top10.reset_index(drop=True, inplace=True)
        
        imlib_top10 = pd.concat([imlib_top10.reset_index(drop=True), imlib_holdout_df], axis=1)
        diimlib_top10 = pd.concat([diimlib_top10.reset_index(drop=True), diimlib_holdout_df], axis=1)

        # Reorder columns (make sure config columns first)
        col_order = [
            'number_lines_per_partition', 'max_rule_set_size', 'max_size_each_rule', 'rules_size_weight', 'rules_accuracy_weight', 'number_quantiles_ordinal',
            'cv_mean_accuracy', 'cv_std_accuracy', 'cv_mean_time', 'cv_std_time',
            'cv_mean_num_rules', 'cv_std_num_rules', 'cv_mean_larger_rule', 'cv_std_larger_rule', 'cv_mean_sum_literals', 'cv_std_sum_literals',
            'holdout_accuracy', 'holdout_train_time', 'holdout_pred_time', 'holdout_num_rules', 'holdout_larger_rule', 'holdout_sum_literals'
        ]

        cols = [c for c in col_order if c in imlib_top10.columns]
        imlib_top10 = imlib_top10[cols]

        cols = [c for c in col_order if c in diimlib_top10.columns]
        diimlib_top10 = diimlib_top10[cols]

        # Save the top 10 results as requested
        imlib_top10.to_csv(imlib_path, index=False)
        print(f"Saved top 10 configs for dataset {database_name} to {imlib_path}")
        diimlib_top10.to_csv(diimlib_path, index=False)
        print(f"Saved top 10 configs for dataset {database_name} to {diimlib_path}")
    else:
        print(f"No successful configurations for dataset {database_name}.")
