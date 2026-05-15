import sys, os
if not sys.path[0] == os.path.abspath('.'):
    sys.path.insert(0, os.path.abspath('.'))

from models.i_imlib import I_IMLIB

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


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

for database_name, categorical_columns_index in zip(database_names, categorical_columns_indexes):
    print(f'\n--- Database: {database_name} ---')
    path = f'./drafts/hyperparameters/iimlib_datasets_results/{database_name}.csv'

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
    from itertools import product
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

    from sklearn.model_selection import RepeatedStratifiedKFold
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    results = []

    for cfg in tqdm(configs, desc=f"Grid search ({len(configs)} configs)"):
        accuracies = []
        times = []
        rules_counts = []
        max_rules_sizes = []
        sum_rules_sizes = []
        failed = False

        # Reuse the same splits for all configs (paired comparisons)
        for train_idx, val_idx in rskf.split(X_train_full, y_train_full):
            Xtr, Xval = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
            ytr, yval = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

            model = I_IMLIB(
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

            try:
                model.fit(Xtr, ytr)
            except Exception as e:
                # log and mark as failed
                print(f"Config failed during fit: {cfg} -- {e}")
                failed = True
                break

            try:
                acc = model.score(Xval, yval)
                accuracies.append(acc)
            except Exception as e:
                print(f"Config failed during score: {cfg} -- {e}")
                failed = True
                break

            # collect structure metrics
            try:
                times.append(model.get_total_time_solver_solutions())
                rules_counts.append(model.get_rule_set_size())
                max_rules_sizes.append(model.get_larger_rule_size())
                sum_rules_sizes.append(model.get_sum_rules_size())
            except Exception as e:
                print(f"Config failed during metric extraction: {cfg} -- {e}")
                failed = True
                break

        if failed:
            continue

        res = {
            **cfg,
            "cv_mean_accuracy": float(np.mean(accuracies)),
            "cv_std_accuracy": float(np.std(accuracies, ddof=1)),
            "cv_mean_time": float(np.mean(times)),
            "cv_mean_num_rules": float(np.mean(rules_counts)),
            "cv_std_num_rules": float(np.std(rules_counts, ddof=1)),
            "cv_mean_larger_rule": float(np.mean(max_rules_sizes)),
            "cv_std_larger_rule": float(np.std(max_rules_sizes, ddof=1)),
            "cv_mean_sum_literals": float(np.mean(sum_rules_sizes)),
            "cv_std_sum_literals": float(np.std(sum_rules_sizes, ddof=1)),
        }
        results.append(res)

    # Save full grid results for inspection
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.to_csv(path.replace('.csv', '_grid_results.csv'), index=False)

        # Select top 10 by CV mean accuracy
        top10 = results_df.sort_values(by='cv_mean_accuracy', ascending=False).head(10).copy()

        # Refit top10 on the full training set and evaluate on the holdout set
        holdout_metrics = []
        for _, r in top10.iterrows():
            cfg = r.to_dict()
            model = I_IMLIB(
                max_rule_set_size=int(cfg['max_rule_set_size']),
                max_size_each_rule=int(cfg['max_size_each_rule']),
                rules_size_weight=int(cfg['rules_size_weight']),
                rules_accuracy_weight=int(cfg['rules_accuracy_weight']),
                categorical_columns_index=categorical_columns_index,
                number_quantiles_ordinal_columns=int(cfg['number_quantiles_ordinal']),
                number_lines_per_partition=int(cfg['number_lines_per_partition']),
                balance_instances=balance_instances,
                balance_instances_seed=balance_instances_seed
            )
            model.fit(X_train_full, y_train_full)
            holdout_acc = model.score(X_holdout, y_holdout)
            holdout_metrics.append({
                'holdout_accuracy': float(holdout_acc),
                'holdout_num_rules': int(model.get_rule_set_size()),
                'holdout_larger_rule': int(model.get_larger_rule_size()),
                'holdout_sum_literals': int(model.get_sum_rules_size()),
            })

        holdout_df = pd.DataFrame(holdout_metrics)
        top10.reset_index(drop=True, inplace=True)
        top10 = pd.concat([top10.reset_index(drop=True), holdout_df], axis=1)

        # Reorder columns (make sure config columns first)
        col_order = [
            'number_lines_per_partition', 'max_rule_set_size', 'max_size_each_rule', 'rules_size_weight', 'rules_accuracy_weight', 'number_quantiles_ordinal',
            'cv_mean_accuracy', 'cv_std_accuracy', 'cv_mean_time',
            'cv_mean_num_rules', 'cv_std_num_rules', 'cv_mean_larger_rule', 'cv_std_larger_rule', 'cv_mean_sum_literals', 'cv_std_sum_literals',
            'holdout_accuracy', 'holdout_num_rules', 'holdout_larger_rule', 'holdout_sum_literals'
        ]
        cols = [c for c in col_order if c in top10.columns]
        top10 = top10[cols]

        # Save the top 10 results as requested
        top10.to_csv(path, index=False)
        print(f"Saved top 10 configs for dataset {database_name} to {path}")
    else:
        print(f"No successful configurations for dataset {database_name}.")
