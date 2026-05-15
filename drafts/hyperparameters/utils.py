import pandas as pd


def show_results_cv(models, datasets):
    """
    Display a dataframe with the best configuration results from each dataset and model.
    
    Parameters:
    -----------
    models : list
        List of model names (e.g., ['decision_tree', 'imli', 'imlib'])
        These are used to construct the path: ./drafts/hyperparameters/{model}_datasets_results/
    datasets : list
        List of dataset names (e.g., ['iris', 'wdbc', 'pima'])
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with columns:
        - Datasets: dataset name
        - Models: model name
        - Number of rules: cv_mean_num_rules ± cv_std_num_rules
        - |R|: cv_mean_sum_literals ± cv_std_sum_literals
        - Largest rule size: cv_mean_larger_rule ± cv_std_larger_rule
        - Accuracy: cv_mean_accuracy ± cv_std_accuracy
        - Training time: cv_mean_time ± cv_std_time
    """
    
    results = []
    
    for dataset in datasets:
        for model in models:
            # Load the grid results for the model-dataset combination
            try:
                csv_path = f'./{model}_datasets_results/{dataset}.csv'
                df_holdout = pd.read_csv(csv_path)

                if dataset in ['transfusion', 'depressed'] and model == 'diimlib':
                    best_row = df_holdout.iloc[-1]
                else:
                    best_row = df_holdout.iloc[0]
                
                # Create formatted strings for each metric
                num_rules = f"{best_row['cv_mean_num_rules']:.2f} ± {best_row['cv_std_num_rules']:.2f}"
                sum_literals = f"{best_row['cv_mean_sum_literals']:.2f} ± {best_row['cv_std_sum_literals']:.2f}"
                larger_rule = f"{best_row['cv_mean_larger_rule']:.2f} ± {best_row['cv_std_larger_rule']:.2f}"
                accuracy = f"{best_row['cv_mean_accuracy']:.4f} ± {best_row['cv_std_accuracy']:.4f}"
                training_time = f"{best_row['cv_mean_time']:.6f} ± {best_row['cv_std_time']:.6f}"
                
                results.append({
                    'Datasets': dataset,
                    'Models': model,
                    'Number of rules': num_rules,
                    '|R|': sum_literals,
                    'Largest rule size': larger_rule,
                    'Accuracy': accuracy,
                    'Training time': training_time
                })
            
            except FileNotFoundError:
                print(f"Warning: Grid results file not found for model '{model}' and dataset '{dataset}'")
            except KeyError as e:
                print(f"Warning: Missing column in results for model '{model}' and dataset '{dataset}': {e}")
    
    # Create and return the results dataframe
    results_df = pd.DataFrame(results)
    
    return results_df

def show_results_ho(models, datasets):
    """
    Display a dataframe with the holdout results from each dataset and model.
    
    Parameters:
    -----------
    models : list
        List of model names (e.g., ['decision_tree', 'imli', 'imlib'])
        These are used to construct the path: ./drafts/hyperparameters/{model}_datasets_results/
    datasets : list
        List of dataset names (e.g., ['iris', 'wdbc', 'pima'])
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with columns:
        - Datasets: dataset name
        - Models: model name
        - Number of rules: holdout_num_rules
        - |R|: holdout_sum_literals
        - Largest rule size: holdout_larger_rule
        - Accuracy: holdout_accuracy
        - Training time: holdout_train_time
    """
    
    results = []
    
    for dataset in datasets:
        for model in models:
            # Load the holdout results for the model-dataset combination
            try:
                csv_path = f'./{model}_datasets_results/{dataset}.csv'
                df_holdout = pd.read_csv(csv_path)
                
                best_row = df_holdout.iloc[0]
                
                # Create formatted strings for each metric
                num_rules = f"{best_row['holdout_num_rules']:.2f}"
                sum_literals = f"{best_row['holdout_sum_literals']:.2f}"
                larger_rule = f"{best_row['holdout_larger_rule']:.2f}"
                accuracy = f"{best_row['holdout_accuracy']:.4f}"
                training_time = f"{best_row['holdout_train_time']:.6f}"
                
                results.append({
                    'Datasets': dataset,
                    'Models': model,
                    'Number of rules': num_rules,
                    '|R|': sum_literals,
                    'Largest rule size': larger_rule,
                    'Accuracy': accuracy,
                    'Training time': training_time
                })
            
            except FileNotFoundError:
                print(f"Warning: Holdout results file not found for model '{model}' and dataset '{dataset}'")
            except KeyError as e:
                print(f"Warning: Missing column in results for model '{model}' and dataset '{dataset}': {e}")
    
    # Create and return the results dataframe
    results_df = pd.DataFrame(results)
    
    return results_df