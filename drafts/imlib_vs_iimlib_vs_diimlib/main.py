import sys, os
if not sys.path[0] == os.path.abspath('.'):
    sys.path.insert(0, os.path.abspath('.'))

from databases.databases_infos import infos

from models.imlib import IMLIB
from models.i_imlib import I_IMLIB
from models.di_imlib import DI_IMLIB

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')


models = {
    'IMLIB': IMLIB,
    'I-IMLIB': I_IMLIB,
    'DI-IMLIB': DI_IMLIB
}

models_configs = {
    'max_rule_set_size': [1, 2, 3, 4],
    'max_size_each_rule': [1, 2, 3, 4],
    'rules_size_weight': [1, 2, 5, 10],
    'rules_accuracy_weight': [1, 2, 5, 10],
    'number_lines_per_partition': [4, 8, 16, 32],
    'number_quantiles_ordinal_columns': [5, 10, 15, 20],
    'balance_instances': True,
    'balance_instances_seed': 21
}

experiment_configs = {
    'seed': 21,
    'n_splits': 5,      # of folds for CV
    'n_repeats': 3,     # of repeats for CV
    'max_evals': 200,   # of hyperparameter sets to evaluate
    'holdout_size': 0.3, # size of holdout set for final evaluation
    'n_jobs': -1  # number of parallel jobs
}


def generate_hyperparameter_configs(config_dict, max_evals):
    """Generate hyperparameter configurations using random sampling"""
    # Generate all possible combinations
    keys = [k for k in config_dict.keys() if k not in ['balance_instances', 'balance_instances_seed']]
    values = [config_dict[k] for k in keys]
    all_combos = list(product(*values))
    
    # Sample up to max_evals configurations
    if len(all_combos) > max_evals:
        np.random.seed(experiment_configs['seed'])
        indices = np.random.choice(len(all_combos), max_evals, replace=False)
        sampled_combos = [all_combos[i] for i in sorted(indices)]
    else:
        sampled_combos = all_combos
    
    configs = []
    for combo in sampled_combos:
        config = {k: v for k, v in zip(keys, combo)}
        config['balance_instances'] = config_dict['balance_instances']
        config['balance_instances_seed'] = config_dict['balance_instances_seed']
        configs.append(config)
    
    return configs


def train_and_evaluate_fold(ModelClass, X_train, y_train, X_val, y_val, config, categorical_cols):
    """Train and evaluate a single fold"""
    try:
        model = ModelClass(categorical_columns_index=categorical_cols, **config)
        model.fit(X_train, y_train)
        
        # CV metrics
        y_pred = X_val.apply(lambda row: model.predict(row.values), axis=1)
        cv_accuracy = accuracy_score(y_val, y_pred)
        
        # Get rule statistics
        rules_sizes = model.get_rules_size()
        num_rules = len(rules_sizes)
        larger_rule = model.get_larger_rule_size()
        sum_literals = model.get_sum_rules_size()
        train_time = model.get_total_time_solver_solutions()
        
        return {
            'accuracy': cv_accuracy,
            'time': train_time,
            'num_rules': num_rules,
            'larger_rule': larger_rule,
            'sum_literals': sum_literals
        }
    except Exception as e:
        print(f"Error in fold evaluation: {e}")
        return {
            'accuracy': 0,
            'time': 0,
            'num_rules': 0,
            'larger_rule': 0,
            'sum_literals': 0
        }


def evaluate_config_cv(ModelClass, X_train_full, y_train_full, config, categorical_cols, n_splits, n_repeats):
    """Evaluate a single configuration using repeated stratified k-fold cross-validation"""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, 
                                     random_state=experiment_configs['seed'])
    
    fold_results = []
    for train_idx, val_idx in rskf.split(X_train_full, y_train_full):
        X_train_fold = X_train_full.iloc[train_idx]
        y_train_fold = y_train_full.iloc[train_idx]
        X_val_fold = X_train_full.iloc[val_idx]
        y_val_fold = y_train_full.iloc[val_idx]
        
        result = train_and_evaluate_fold(ModelClass, X_train_fold, y_train_fold, 
                                        X_val_fold, y_val_fold, config, categorical_cols)
        fold_results.append(result)
    
    # Aggregate results
    accuracies = [r['accuracy'] for r in fold_results]
    times = [r['time'] for r in fold_results]
    num_rules = [r['num_rules'] for r in fold_results]
    larger_rules = [r['larger_rule'] for r in fold_results]
    sum_literals = [r['sum_literals'] for r in fold_results]
    
    return {
        'config': config,
        'cv_mean_accuracy': np.mean(accuracies),
        'cv_std_accuracy': np.std(accuracies),
        'cv_mean_time': np.mean(times),
        'cv_mean_num_rules': np.mean(num_rules),
        'cv_std_num_rules': np.std(num_rules),
        'cv_mean_larger_rule': np.mean(larger_rules),
        'cv_std_larger_rule': np.std(larger_rules),
        'cv_mean_sum_literals': np.mean(sum_literals),
        'cv_std_sum_literals': np.std(sum_literals)
    }


def evaluate_on_holdout(ModelClass, X_train_full, y_train_full, X_holdout, y_holdout, config, categorical_cols):
    """Train on full training set and evaluate on holdout"""
    try:
        model = ModelClass(categorical_columns_index=categorical_cols, **config)
        model.fit(X_train_full, y_train_full)
        holdout_train_time = model.get_total_time_solver_solutions()
        
        # Measure prediction time
        start_time = time.time()
        y_pred = X_holdout.apply(lambda row: model.predict(row.values), axis=1)
        holdout_pred_time = time.time() - start_time
        
        holdout_accuracy = accuracy_score(y_holdout, y_pred)
        
        rules_sizes = model.get_rules_size()
        holdout_num_rules = len(rules_sizes)
        holdout_larger_rule = model.get_larger_rule_size()
        holdout_sum_literals = model.get_sum_rules_size()
        
        return {
            'holdout_accuracy': holdout_accuracy,
            'holdout_num_rules': holdout_num_rules,
            'holdout_larger_rule': holdout_larger_rule,
            'holdout_sum_literals': holdout_sum_literals,
            'holdout_train_time': holdout_train_time,
            'holdout_pred_time': holdout_pred_time
        }
    except Exception as e:
        print(f"Error in holdout evaluation: {e}")
        return {
            'holdout_accuracy': 0,
            'holdout_num_rules': 0,
            'holdout_larger_rule': 0,
            'holdout_sum_literals': 0,
            'holdout_train_time': 0,
            'holdout_pred_time': 0
        }


# Create output directories
for model_name in models.keys():
    dir_name = f'{model_name.replace("-", "").lower()}_best_config_results'
    dir_path = f'./drafts/imlib_vs_iimlib_vs_diimlib/{dir_name}'
    os.makedirs(dir_path, exist_ok=True)


# Main experiment loop
for model_name, ModelClass in tqdm(models.items(), desc='Models', position=0, leave=True):
    print(f'\n{"="*60}')
    print(f'Model: {model_name}')
    print(f'{"="*60}')

    dir_name = f'{model_name.replace("-", "").lower()}_best_config_results'
    dir_path = f'./drafts/imlib_vs_iimlib_vs_diimlib/{dir_name}'
    
    # Generate hyperparameter configurations
    configs_to_test = generate_hyperparameter_configs(models_configs, experiment_configs['max_evals'])
    print(f'Total configurations to test: {len(configs_to_test)}')

    # Dictionary to store all models results for all datasets
    all_models_results = {m: {} for m in models.keys()}

    for dataset_name, categorical_cols in tqdm(zip(infos['database_names'], infos['categorical_columns_indexes']), 
                                               total=len(infos['database_names']), desc='Datasets', position=1, leave=False):
        print(f'\n--- Dataset: {dataset_name} ---')

        # Load dataset
        Xy = pd.read_csv(f'./databases/{dataset_name}.csv')
        X = Xy.drop(['Class'], axis=1)
        y = Xy['Class']

        # Create an outer hold-out split for final evaluation (30%) and use the training part to search hyperparams
        X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
            X, y, test_size=experiment_configs['holdout_size'], random_state=experiment_configs['seed'], stratify=y
        )
        
        # ===== CV for current model =====
        print(f'  Evaluating configurations with CV...')
        cv_results = Parallel(n_jobs=experiment_configs['n_jobs'])(
            delayed(evaluate_config_cv)(
                ModelClass, X_train_full, y_train_full, config, categorical_cols,
                experiment_configs['n_splits'], experiment_configs['n_repeats']
            ) for config in tqdm(configs_to_test, desc='CV Configs', position=2, leave=False)
        )
        
        # Find best configuration based on CV mean accuracy
        best_config_result = max(cv_results, key=lambda x: x['cv_mean_accuracy'])
        best_config = best_config_result['config']
        print(f'  Best CV accuracy: {best_config_result["cv_mean_accuracy"]:.4f}')
        
        # ===== Holdout evaluation for current model =====
        print(f'  Evaluating best config on holdout...')
        holdout_result_current = evaluate_on_holdout(
            ModelClass, X_train_full, y_train_full, X_holdout, y_holdout, best_config, categorical_cols
        )
        
        # Store result for current model
        all_models_results[model_name][dataset_name] = {
            'best_config': best_config,
            'cv_results': best_config_result,
            'holdout_results': {model_name: holdout_result_current},
            'X_train_full': X_train_full,
            'y_train_full': y_train_full,
            'X_holdout': X_holdout,
            'y_holdout': y_holdout
        }
        
        print(f'  Evaluating best config on other models holdout...')
        
        # ===== Holdout evaluation for other models using best config of current model (in parallel) =====
        other_models_to_eval = [(name, cls) for name, cls in models.items() if name != model_name]
        other_holdout_results = Parallel(n_jobs=experiment_configs['n_jobs'])(
            delayed(evaluate_on_holdout)(
                OtherModelClass, X_train_full, y_train_full, X_holdout, y_holdout, best_config, categorical_cols
            )
            for other_model_name, OtherModelClass in tqdm(other_models_to_eval, desc='Other Models', position=3, leave=False)
        )
        
        all_models_results[model_name][dataset_name]['holdout_results'] = {model_name: holdout_result_current}
        for (other_model_name, _), result in zip(other_models_to_eval, other_holdout_results):
            all_models_results[model_name][dataset_name]['holdout_results'][other_model_name] = result
        
        # ===== Create CSV for current model and dataset (one row per model) =====
        # Prepare configuration columns
        config_cols = {}
        for param_name, param_value in best_config.items():
            config_cols[param_name] = param_value
        
        # Initialize data for all models
        csv_data = {
            'model': [],
            # Configuration columns
            **{param: [] for param in config_cols.keys()},
            # CV metrics columns (only for current model)
            'cv_mean_accuracy': [],
            'cv_std_accuracy': [],
            'cv_mean_time': [],
            'cv_mean_num_rules': [],
            'cv_std_num_rules': [],
            'cv_mean_larger_rule': [],
            'cv_std_larger_rule': [],
            'cv_mean_sum_literals': [],
            'cv_std_sum_literals': [],
            # Holdout metrics for each model
            'holdout_accuracy': [],
            'holdout_num_rules': [],
            'holdout_larger_rule': [],
            'holdout_sum_literals': [],
            'holdout_train_time': [],
            'holdout_pred_time': []
        }
        
        # Add row for current model (with config and CV metrics)
        csv_data['model'].append(model_name)
        for param, value in config_cols.items():
            csv_data[param].append(value)
        csv_data['cv_mean_accuracy'].append(best_config_result['cv_mean_accuracy'])
        csv_data['cv_std_accuracy'].append(best_config_result['cv_std_accuracy'])
        csv_data['cv_mean_time'].append(best_config_result['cv_mean_time'])
        csv_data['cv_mean_num_rules'].append(best_config_result['cv_mean_num_rules'])
        csv_data['cv_std_num_rules'].append(best_config_result['cv_std_num_rules'])
        csv_data['cv_mean_larger_rule'].append(best_config_result['cv_mean_larger_rule'])
        csv_data['cv_std_larger_rule'].append(best_config_result['cv_std_larger_rule'])
        csv_data['cv_mean_sum_literals'].append(best_config_result['cv_mean_sum_literals'])
        csv_data['cv_std_sum_literals'].append(best_config_result['cv_std_sum_literals'])
        csv_data['holdout_accuracy'].append(holdout_result_current['holdout_accuracy'])
        csv_data['holdout_num_rules'].append(holdout_result_current['holdout_num_rules'])
        csv_data['holdout_larger_rule'].append(holdout_result_current['holdout_larger_rule'])
        csv_data['holdout_sum_literals'].append(holdout_result_current['holdout_sum_literals'])
        csv_data['holdout_train_time'].append(holdout_result_current['holdout_train_time'])
        csv_data['holdout_pred_time'].append(holdout_result_current['holdout_pred_time'])
        
        # Add rows for other models (with empty config and CV metrics, only holdout metrics)
        for other_model_name in models.keys():
            if other_model_name != model_name:
                csv_data['model'].append(other_model_name)
                # Config columns are empty for other models
                for param in config_cols.keys():
                    csv_data[param].append(np.nan)
                # CV metrics are empty for other models
                csv_data['cv_mean_accuracy'].append(np.nan)
                csv_data['cv_std_accuracy'].append(np.nan)
                csv_data['cv_mean_time'].append(np.nan)
                csv_data['cv_mean_num_rules'].append(np.nan)
                csv_data['cv_std_num_rules'].append(np.nan)
                csv_data['cv_mean_larger_rule'].append(np.nan)
                csv_data['cv_std_larger_rule'].append(np.nan)
                csv_data['cv_mean_sum_literals'].append(np.nan)
                csv_data['cv_std_sum_literals'].append(np.nan)
                # Holdout metrics for other models
                other_results = all_models_results[model_name][dataset_name]['holdout_results'][other_model_name]
                csv_data['holdout_accuracy'].append(other_results['holdout_accuracy'])
                csv_data['holdout_num_rules'].append(other_results['holdout_num_rules'])
                csv_data['holdout_larger_rule'].append(other_results['holdout_larger_rule'])
                csv_data['holdout_sum_literals'].append(other_results['holdout_sum_literals'])
                csv_data['holdout_train_time'].append(other_results['holdout_train_time'])
                csv_data['holdout_pred_time'].append(other_results['holdout_pred_time'])
        
        df_results = pd.DataFrame(csv_data)
        csv_path = f'{dir_path}/{dataset_name}.csv'
        df_results.to_csv(csv_path, index=False)
        print(f'  CSV saved to: {csv_path}')

print(f'\n{"="*60}')
print('Experiment completed successfully!')
print(f'{"="*60}')
        