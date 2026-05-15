import sys, os
if not sys.path[0] == os.path.abspath('.'):
    sys.path.insert(0, os.path.abspath('.'))

from databases.databases_infos import infos

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm
from utils.functions import tree_to_dnf
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')


# Decision Tree configuration
dt_configs = {
    'max_depth': [1, 2, 3, 4],
    'max_leaf_nodes': [1, 2, 3, 4]
}

experiment_configs = {
    'seed': 21,
    'n_splits': 5,       # of folds for CV
    'n_repeats': 3,      # of repeats for CV
    'holdout_size': 0.3, # size of holdout set for final evaluation
    'n_jobs': -1         # number of parallel jobs
}


def generate_hyperparameter_configs(config_dict):
    """Generate all hyperparameter configurations from dict"""
    keys = list(config_dict.keys())
    values = [config_dict[k] for k in keys]
    all_combos = list(product(*values))
    
    configs = []
    for combo in all_combos:
        config = {k: v for k, v in zip(keys, combo)}
        configs.append(config)
    
    return configs


def train_and_evaluate_fold(X_train, y_train, X_val, y_val, config):
    """Train and evaluate a single fold"""
    try:
        model = DecisionTreeClassifier(
            max_depth=config['max_depth'],
            max_leaf_nodes=config['max_leaf_nodes'],
            random_state=experiment_configs['seed']
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Prediction
        y_pred = model.predict(X_val)
        cv_accuracy = accuracy_score(y_val, y_pred)
        
        return {
            'accuracy': cv_accuracy,
            'time': train_time,
            'n_nodes': model.tree_.node_count,
            'depth': model.get_depth()
        }
    except Exception as e:
        print(f"Error in fold evaluation: {e}")
        return {
            'accuracy': 0,
            'time': 0,
            'n_nodes': 0,
            'depth': 0
        }


def evaluate_config_cv(X_train_full, y_train_full, config, n_splits, n_repeats):
    """Evaluate a single configuration using repeated stratified k-fold cross-validation"""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                    random_state=experiment_configs['seed'])
    
    fold_results = []
    for train_idx, val_idx in rskf.split(X_train_full, y_train_full):
        X_train_fold = X_train_full.iloc[train_idx]
        y_train_fold = y_train_full.iloc[train_idx]
        X_val_fold = X_train_full.iloc[val_idx]
        y_val_fold = y_train_full.iloc[val_idx]
        
        result = train_and_evaluate_fold(X_train_fold, y_train_fold,
                                        X_val_fold, y_val_fold, config)
        fold_results.append(result)
    
    # Aggregate results
    accuracies = [r['accuracy'] for r in fold_results]
    times = [r['time'] for r in fold_results]
    n_nodes = [r['n_nodes'] for r in fold_results]
    depths = [r['depth'] for r in fold_results]
    
    return {
        'config': config,
        'cv_mean_accuracy': np.mean(accuracies),
        'cv_std_accuracy': np.std(accuracies),
        'cv_mean_time': np.mean(times),
        'cv_mean_n_nodes': np.mean(n_nodes),
        'cv_std_n_nodes': np.std(n_nodes),
        'cv_mean_depth': np.mean(depths),
        'cv_std_depth': np.std(depths)
    }


def evaluate_on_holdout(X_train_full, y_train_full, X_holdout, y_holdout, config, feature_names):
    """Train on full training set and evaluate on holdout"""
    try:
        model = DecisionTreeClassifier(
            max_depth=config['max_depth'],
            max_leaf_nodes=config['max_leaf_nodes'],
            random_state=experiment_configs['seed']
        )
        
        # Measure training time
        start_time = time.time()
        model.fit(X_train_full, y_train_full)
        holdout_train_time = time.time() - start_time
        
        # Measure prediction time
        start_time = time.time()
        y_pred = model.predict(X_holdout)
        holdout_pred_time = time.time() - start_time
        
        holdout_accuracy = accuracy_score(y_holdout, y_pred)
        
        # Convert tree to DNF formula
        target_names = ['0', '1']  # Classes as strings
        dnf_clauses = tree_to_dnf(model, feature_names, target_names, class_label='1')
        
        # Calculate metrics
        holdout_num_rules = len(dnf_clauses)
        holdout_larger_rule = max([len(clause) for clause in dnf_clauses]) if dnf_clauses else 0
        holdout_sum_literals = sum([len(clause) for clause in dnf_clauses])
        
        return {
            'holdout_accuracy': holdout_accuracy,
            'holdout_n_nodes': model.tree_.node_count,
            'holdout_depth': model.get_depth(),
            'holdout_train_time': holdout_train_time,
            'holdout_pred_time': holdout_pred_time,
            'holdout_num_rules': holdout_num_rules,
            'holdout_larger_rule': holdout_larger_rule,
            'holdout_sum_literals': holdout_sum_literals
        }
    except Exception as e:
        print(f"Error in holdout evaluation: {e}")
        return {
            'holdout_accuracy': 0,
            'holdout_n_nodes': 0,
            'holdout_depth': 0,
            'holdout_train_time': 0,
            'holdout_pred_time': 0,
            'holdout_num_rules': 0,
            'holdout_larger_rule': 0,
            'holdout_sum_literals': 0
        }


# Create output directories
dir_name = 'decision_tree_results'
dir_path = f'./drafts/imlib_vs_iimlib_vs_diimlib/{dir_name}'
os.makedirs(dir_path, exist_ok=True)

print(f'\n{"="*60}')
print(f'Decision Tree Hyperparameter Search')
print(f'{"="*60}')

# Generate hyperparameter configurations
configs_to_test = generate_hyperparameter_configs(dt_configs)
print(f'Total configurations to test: {len(configs_to_test)}')
print(f'Configurations: {dt_configs}')

# Main experiment loop
for dataset_name, categorical_cols in tqdm(zip(infos['database_names'], infos['categorical_columns_indexes']),
                                           total=len(infos['database_names']), desc='Datasets', position=0, leave=True):
    print(f'\n--- Dataset: {dataset_name} ---')

    # Load dataset
    Xy = pd.read_csv(f'./databases/{dataset_name}.csv')
    X = Xy.drop(['Class'], axis=1)
    y = Xy['Class']

    # Create an outer hold-out split for final evaluation (30%) and use the training part to search hyperparams
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X, y, test_size=experiment_configs['holdout_size'], random_state=experiment_configs['seed'], stratify=y
    )

    print(f'  Evaluating {len(configs_to_test)} configurations with CV...')
    
    # ===== CV evaluation =====
    cv_results = Parallel(n_jobs=experiment_configs['n_jobs'])(
        delayed(evaluate_config_cv)(
            X_train_full, y_train_full, config,
            experiment_configs['n_splits'], experiment_configs['n_repeats']
        ) for config in tqdm(configs_to_test, desc='CV Configs', position=1, leave=False)
    )

    # Find best configuration based on CV mean accuracy
    best_config_result = max(cv_results, key=lambda x: x['cv_mean_accuracy'])
    best_config = best_config_result['config']
    print(f'  Best CV accuracy: {best_config_result["cv_mean_accuracy"]:.4f}')
    print(f'  Best config: max_depth={best_config["max_depth"]}, max_leaf_nodes={best_config["max_leaf_nodes"]}')

    # ===== Holdout evaluation =====
    print(f'  Evaluating best config on holdout...')
    holdout_result = evaluate_on_holdout(
        X_train_full, y_train_full, X_holdout, y_holdout, best_config, list(X.columns)
    )

    # ===== Create CSV for all configurations =====
    print(f'  Saving results to CSV...')
    
    csv_data = {
        'max_depth': [],
        'max_leaf_nodes': [],
        'cv_mean_accuracy': [],
        'cv_std_accuracy': [],
        'cv_mean_time': [],
        'cv_mean_n_nodes': [],
        'cv_std_n_nodes': [],
        'cv_mean_depth': [],
        'cv_std_depth': [],
        'holdout_accuracy': [],
        'holdout_n_nodes': [],
        'holdout_depth': [],
        'holdout_train_time': [],
        'holdout_pred_time': [],
        'holdout_num_rules': [],
        'holdout_larger_rule': [],
        'holdout_sum_literals': [],
        'is_best_config': []
    }

    # Add all CV results
    for cv_result in cv_results:
        config = cv_result['config']
        csv_data['max_depth'].append(config['max_depth'])
        csv_data['max_leaf_nodes'].append(config['max_leaf_nodes'])
        csv_data['cv_mean_accuracy'].append(cv_result['cv_mean_accuracy'])
        csv_data['cv_std_accuracy'].append(cv_result['cv_std_accuracy'])
        csv_data['cv_mean_time'].append(cv_result['cv_mean_time'])
        csv_data['cv_mean_n_nodes'].append(cv_result['cv_mean_n_nodes'])
        csv_data['cv_std_n_nodes'].append(cv_result['cv_std_n_nodes'])
        csv_data['cv_mean_depth'].append(cv_result['cv_mean_depth'])
        csv_data['cv_std_depth'].append(cv_result['cv_std_depth'])

        # Add holdout results only for best config
        if config == best_config:
            csv_data['holdout_accuracy'].append(holdout_result['holdout_accuracy'])
            csv_data['holdout_n_nodes'].append(holdout_result['holdout_n_nodes'])
            csv_data['holdout_depth'].append(holdout_result['holdout_depth'])
            csv_data['holdout_train_time'].append(holdout_result['holdout_train_time'])
            csv_data['holdout_pred_time'].append(holdout_result['holdout_pred_time'])
            csv_data['holdout_num_rules'].append(holdout_result['holdout_num_rules'])
            csv_data['holdout_larger_rule'].append(holdout_result['holdout_larger_rule'])
            csv_data['holdout_sum_literals'].append(holdout_result['holdout_sum_literals'])
            csv_data['is_best_config'].append(True)
        else:
            csv_data['holdout_accuracy'].append(np.nan)
            csv_data['holdout_n_nodes'].append(np.nan)
            csv_data['holdout_depth'].append(np.nan)
            csv_data['holdout_train_time'].append(np.nan)
            csv_data['holdout_pred_time'].append(np.nan)
            csv_data['holdout_num_rules'].append(np.nan)
            csv_data['holdout_larger_rule'].append(np.nan)
            csv_data['holdout_sum_literals'].append(np.nan)
            csv_data['is_best_config'].append(False)

    df_results = pd.DataFrame(csv_data)
    csv_path = f'{dir_path}/{dataset_name}.csv'
    df_results.to_csv(csv_path, index=False)
    print(f'  CSV saved to: {csv_path}')

print(f'\n{"="*60}')
print('Decision Tree experiment completed successfully!')
print(f'{"="*60}')
print(f'\nResults saved in: {dir_path}')
