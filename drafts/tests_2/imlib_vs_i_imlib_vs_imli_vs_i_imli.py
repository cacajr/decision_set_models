import sys, os
if not sys.path[0] == os.path.abspath('.'):
    sys.path.insert(0, os.path.abspath('.'))

from models.imlib import IMLIB
from models.i_imlib import I_IMLIB
from models.imli import IMLI
from models.i_imli import I_IMLI

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy
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
number_lines_per_partition = [8, 16]
max_rule_set_sizes = [1, 2, 3]
# max_sizes_each_rule = [1, 2, 3]
rules_size_weight = 1
rules_accuracy_weights = [5, 10]
number_quantiles_ordinal_columns = 5
balance_instances = True
balance_instances_seed = 21

''' - Test realizations number '''
number_realizations = 10

for database_name, categorical_columns_index in zip(database_names, categorical_columns_indexes):
    ''' - Results tables configurations '''
    columns = ['Configuration', 'Rule sizes', 'Average deviation of rule sizes', 'Standard deviation of rule sizes', 'Entropy of rule sizes', 'Number of rules', '|R|', 'Largest rule size', 'Accuracy', 'Training time', 'Confusion matrix']
    imli_results_df = pd.DataFrame([], columns=columns)
    i_imli_results_df = pd.DataFrame([], columns=columns)
    imlib_results_df = pd.DataFrame([], columns=columns)
    i_imlib_results_df = pd.DataFrame([], columns=columns)

    ''' - Test results informations path '''
    imli_results_path = f'./drafts/tests_2/imlib_vs_i_imlib_vs_imli_vs_i_imli_results/imlib_vs_i_imlib_vs_imli_vs_i_imli_results/{database_name}_imli.csv'
    i_imli_results_path = f'./drafts/tests_2/imlib_vs_i_imlib_vs_imli_vs_i_imli_results/imlib_vs_i_imlib_vs_imli_vs_i_imli_results/{database_name}_i_imli.csv'
    imlib_results_path = f'./drafts/tests_2/imlib_vs_i_imlib_vs_imli_vs_i_imli_results/imlib_vs_i_imlib_vs_imli_vs_i_imli_results/{database_name}_imlib.csv'
    i_imlib_results_path = f'./drafts/tests_2/imlib_vs_i_imlib_vs_imli_vs_i_imli_results/imlib_vs_i_imlib_vs_imli_vs_i_imli_results/{database_name}_i_imlib.csv'
    
    ''' - Loading dataset '''
    database_path = f'./databases/{database_name}.csv'
    Xy = pd.read_csv(database_path)
    X = Xy.drop(['Class'], axis=1)
    y = Xy['Class']

    for r in tqdm(range(number_realizations)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

        # validation dataset
        X_train_train, X_test_test, y_train_train, y_test_test = train_test_split(X_train, y_train, test_size = 0.1)

        # best configs in accuracy terms on validation dataset
        imli_best_config = [0, 0, 0, 0]       # [lpp, mrss, raw, accuracy]
        i_imli_best_config = [0, 0, 0, 0]     # [lpp, mrss, raw, accuracy]
        imlib_best_config = [0, 0, 0, 0, 0]   # [lpp, mrss, raw, mser, accuracy]
        i_imlib_best_config = [0, 0, 0, 0, 0] # [lpp, mrss, raw, mser, accuracy]

        for lpp in tqdm(number_lines_per_partition, desc=f'lpp: 0 | mrss: 0 | raw: 0 | mser: 0 '):
            for mrss in tqdm(max_rule_set_sizes, desc=f'lpp: {lpp} | mrss: 0 | raw: 0 | mser: 0'):
                for raw in tqdm(rules_accuracy_weights, desc=f'lpp: {lpp} | mrss: {mrss} | raw: 0 | mser: 0'):
                    imli_model = IMLI(
                        max_rule_set_size = mrss,
                        rules_size_weight= rules_size_weight,
                        rules_accuracy_weight = raw,
                        categorical_columns_index = categorical_columns_index,
                        number_quantiles_ordinal_columns = number_quantiles_ordinal_columns,
                        number_lines_per_partition = lpp,
                        balance_instances = balance_instances,
                        balance_instances_seed = balance_instances_seed
                    )

                    i_imli_model = I_IMLI(
                        max_rule_set_size = mrss,
                        rules_size_weight= rules_size_weight,
                        rules_accuracy_weight = raw,
                        categorical_columns_index = categorical_columns_index,
                        number_quantiles_ordinal_columns = number_quantiles_ordinal_columns,
                        number_lines_per_partition = lpp,
                        balance_instances = balance_instances,
                        balance_instances_seed = balance_instances_seed
                    )

                    imli_model.fit(X_train_train, y_train_train)
                    i_imli_model.fit(X_train_train, y_train_train)

                    imli_accuracy = imli_model.score(X_test_test, y_test_test)
                    i_imli_accuracy = i_imli_model.score(X_test_test, y_test_test)
                    
                    if imli_accuracy > imli_best_config[-1]:
                        imli_best_config = [lpp, mrss, raw, imli_accuracy]
                    if i_imli_accuracy > i_imli_best_config[-1]:
                        i_imli_best_config = [lpp, mrss, raw, i_imli_accuracy]

                    imli_larger_rule_size = imli_model.get_larger_rule_size()
                    i_imli_larger_rule_size = imli_model.get_larger_rule_size()

                    larger_rule_size = imli_larger_rule_size if imli_larger_rule_size <= i_imli_larger_rule_size else i_imli_larger_rule_size
                    # case the larger rule be 1, so iterate mser once
                    larger_rule_size = 2 if larger_rule_size == 1 else larger_rule_size

                    for mser in tqdm(range(1, larger_rule_size), desc=f'lpp: {lpp} | mrss: {mrss} | raw: {raw} | mser: 0'):
                        imlib_model = IMLIB(
                            max_rule_set_size = mrss,
                            max_size_each_rule = mser,
                            rules_accuracy_weight = raw,
                            categorical_columns_index = categorical_columns_index,
                            number_quantiles_ordinal_columns = number_quantiles_ordinal_columns,
                            number_lines_per_partition = lpp,
                            balance_instances = balance_instances,
                            balance_instances_seed = balance_instances_seed
                        )

                        i_imlib_model = I_IMLIB(
                            max_rule_set_size = mrss,
                            max_size_each_rule = mser,
                            rules_accuracy_weight = raw,
                            categorical_columns_index = categorical_columns_index,
                            number_quantiles_ordinal_columns = number_quantiles_ordinal_columns,
                            number_lines_per_partition = lpp,
                            balance_instances = balance_instances,
                            balance_instances_seed = balance_instances_seed
                        )

                        imlib_model.fit(X_train_train, y_train_train)
                        i_imlib_model.fit(X_train_train, y_train_train)

                        imlib_accuracy = imlib_model.score(X_test_test, y_test_test)
                        i_imlib_accuracy = i_imlib_model.score(X_test_test, y_test_test)

                        if imlib_accuracy > imlib_best_config[-1]:
                            imlib_best_config = [lpp, mrss, raw, mser, imlib_accuracy]
                        if i_imlib_accuracy > i_imlib_best_config[-1]:
                            i_imlib_best_config = [lpp, mrss, raw, mser, i_imlib_accuracy]
                    
        imli_model = IMLI(
            max_rule_set_size = imli_best_config[1],
            rules_size_weight= rules_size_weight,
            rules_accuracy_weight = imli_best_config[2],
            categorical_columns_index = categorical_columns_index,
            number_quantiles_ordinal_columns = number_quantiles_ordinal_columns,
            number_lines_per_partition = imli_best_config[0],
            balance_instances = balance_instances,
            balance_instances_seed = balance_instances_seed
        )

        i_imli_model = I_IMLI(
            max_rule_set_size = i_imli_best_config[1],
            rules_size_weight= rules_size_weight,
            rules_accuracy_weight = i_imli_best_config[2],
            categorical_columns_index = categorical_columns_index,
            number_quantiles_ordinal_columns = number_quantiles_ordinal_columns,
            number_lines_per_partition = i_imli_best_config[0],
            balance_instances = balance_instances,
            balance_instances_seed = balance_instances_seed
        )

        imlib_model = IMLIB(
            max_rule_set_size = imlib_best_config[1],
            max_size_each_rule = imlib_best_config[3],
            rules_accuracy_weight = imlib_best_config[2],
            categorical_columns_index = categorical_columns_index,
            number_quantiles_ordinal_columns = number_quantiles_ordinal_columns,
            number_lines_per_partition = imlib_best_config[0],
            balance_instances = balance_instances,
            balance_instances_seed = balance_instances_seed
        )

        i_imlib_model = I_IMLIB(
            max_rule_set_size = i_imlib_best_config[1],
            max_size_each_rule = i_imlib_best_config[3],
            rules_accuracy_weight = i_imlib_best_config[2],
            categorical_columns_index = categorical_columns_index,
            number_quantiles_ordinal_columns = number_quantiles_ordinal_columns,
            number_lines_per_partition = i_imlib_best_config[0],
            balance_instances = balance_instances,
            balance_instances_seed = balance_instances_seed
        )

        imli_model.fit(X_train, y_train)
        i_imli_model.fit(X_train, y_train)
        imlib_model.fit(X_train, y_train)
        i_imlib_model.fit(X_train, y_train)

        imli_rules_size = imli_model.get_rules_size()
        imli_avg_deviation_rules_size = np.average(imli_rules_size)
        imli_std_deviation_rules_size = np.std(imli_rules_size)
        imli_sum_rules_size = sum(imli_rules_size)
        imli_entropy_rules_size = entropy([size/imli_sum_rules_size for size in imli_rules_size], base=2)
        imli_predicts = [imli_model.predict(sample) for sample in X_test.values]
        imli_result = pd.DataFrame([[
            f'lpp: {imli_best_config[0]} | mrss: {imli_best_config[1]} | raw: {imli_best_config[2]}',
            imli_rules_size,
            imli_avg_deviation_rules_size,
            imli_std_deviation_rules_size,
            imli_entropy_rules_size,
            imli_model.get_rule_set_size(),
            imli_model.get_sum_rules_size(),
            imli_model.get_larger_rule_size(),
            imli_model.score(X_test, y_test),
            imli_model.get_total_time_solver_solutions(),
            confusion_matrix(y_test, imli_predicts)
        ]], columns=columns)

        i_imli_rules_size = i_imli_model.get_rules_size()
        i_imli_avg_deviation_rules_size = np.average(i_imli_rules_size)
        i_imli_std_deviation_rules_size = np.std(i_imli_rules_size)
        i_imli_sum_rules_size = sum(i_imli_rules_size)
        i_imli_entropy_rules_size = entropy([size/i_imli_sum_rules_size for size in i_imli_rules_size], base=2)
        i_imli_predicts = [i_imli_model.predict(sample) for sample in X_test.values]
        i_imli_result = pd.DataFrame([[
            f'lpp: {i_imli_best_config[0]} | mrss: {i_imli_best_config[1]} | raw: {i_imli_best_config[2]}',
            i_imli_rules_size,
            i_imli_avg_deviation_rules_size,
            i_imli_std_deviation_rules_size,
            i_imli_entropy_rules_size,
            i_imli_model.get_rule_set_size(),
            i_imli_model.get_sum_rules_size(),
            i_imli_model.get_larger_rule_size(),
            i_imli_model.score(X_test, y_test),
            i_imli_model.get_total_time_solver_solutions(),
            confusion_matrix(y_test, i_imli_predicts)
        ]], columns=columns)

        imlib_rules_size = imlib_model.get_rules_size()
        imlib_avg_deviation_rules_size = np.average(imlib_rules_size)
        imlib_std_deviation_rules_size = np.std(imlib_rules_size)
        imlib_sum_rules_size = sum(imlib_rules_size)
        imlib_entropy_rules_size = entropy([size/imlib_sum_rules_size for size in imlib_rules_size], base=2)
        imlib_predicts = [imlib_model.predict(sample) for sample in X_test.values]
        imlib_result = pd.DataFrame([[
            f'lpp: {imlib_best_config[0]} | mrss: {imlib_best_config[1]} | raw: {imlib_best_config[2]} | mser: {imlib_best_config[3]}',
            imlib_rules_size,
            imlib_avg_deviation_rules_size,
            imlib_std_deviation_rules_size,
            imlib_entropy_rules_size,
            imlib_model.get_rule_set_size(),
            imlib_model.get_sum_rules_size(),
            imlib_model.get_larger_rule_size(),
            imlib_model.score(X_test, y_test),
            imlib_model.get_total_time_solver_solutions(),
            confusion_matrix(y_test, imlib_predicts)
        ]], columns=columns)

        i_imlib_rules_size = i_imlib_model.get_rules_size()
        i_imlib_avg_deviation_rules_size = np.average(i_imlib_rules_size)
        i_imlib_std_deviation_rules_size = np.std(i_imlib_rules_size)
        i_imlib_sum_rules_size = sum(i_imlib_rules_size)
        i_imlib_entropy_rules_size = entropy([size/i_imlib_sum_rules_size for size in i_imlib_rules_size], base=2)
        i_imlib_predicts = [i_imlib_model.predict(sample) for sample in X_test.values]
        i_imlib_result = pd.DataFrame([[
            f'lpp: {i_imlib_best_config[0]} | mrss: {i_imlib_best_config[1]} | raw: {i_imlib_best_config[2]} | mser: {i_imlib_best_config[3]}',
            i_imlib_rules_size,
            i_imlib_avg_deviation_rules_size,
            i_imlib_std_deviation_rules_size,
            i_imlib_entropy_rules_size,
            i_imlib_model.get_rule_set_size(),
            i_imlib_model.get_sum_rules_size(),
            i_imlib_model.get_larger_rule_size(),
            i_imlib_model.score(X_test, y_test),
            i_imlib_model.get_total_time_solver_solutions(),
            confusion_matrix(y_test, i_imlib_predicts)
        ]], columns=columns)

        imli_results_df = pd.concat([imli_results_df, imli_result])
        i_imli_results_df = pd.concat([i_imli_results_df, i_imli_result])
        imlib_results_df = pd.concat([imlib_results_df, imlib_result])
        i_imlib_results_df = pd.concat([i_imlib_results_df, i_imlib_result])

    imli_averages = pd.DataFrame([[
        'Averages',
        '',
        f"{round(imli_results_df['Average deviation of rule sizes'].mean(), 4)} ± {round(imli_results_df['Average deviation of rule sizes'].std(), 4)}",
        f"{round(imli_results_df['Standard deviation of rule sizes'].mean(), 4)} ± {round(imli_results_df['Standard deviation of rule sizes'].std(), 4)}",
        f"{round(imli_results_df['Entropy of rule sizes'].mean(), 4)} ± {round(imli_results_df['Entropy of rule sizes'].std(), 4)}",
        f"{round(imli_results_df['Number of rules'].mean(), 4)} ± {round(imli_results_df['Number of rules'].std(), 4)}",
        f"{round(imli_results_df['|R|'].mean(), 4)} ± {round(imli_results_df['|R|'].std(), 4)}",
        f"{round(imli_results_df['Largest rule size'].mean(), 4)} ± {round(imli_results_df['Largest rule size'].std(), 4)}",
        f"{round(imli_results_df['Accuracy'].mean(), 4)} ± {round(imli_results_df['Accuracy'].std(), 4)}",
        f"{round(imli_results_df['Training time'].mean(), 4)} ± {round(imli_results_df['Training time'].std(), 4)}",
        ''
    ]], columns=columns)
    imli_results_df = pd.concat([imli_results_df, imli_averages])

    i_imli_averages = pd.DataFrame([[
        'Averages',
        '',
        f"{round(i_imli_results_df['Average deviation of rule sizes'].mean(), 4)} ± {round(i_imli_results_df['Average deviation of rule sizes'].std(), 4)}",
        f"{round(i_imli_results_df['Standard deviation of rule sizes'].mean(), 4)} ± {round(i_imli_results_df['Standard deviation of rule sizes'].std(), 4)}",
        f"{round(i_imli_results_df['Entropy of rule sizes'].mean(), 4)} ± {round(i_imli_results_df['Entropy of rule sizes'].std(), 4)}",
        f"{round(i_imli_results_df['Number of rules'].mean(), 4)} ± {round(i_imli_results_df['Number of rules'].std(), 4)}",
        f"{round(i_imli_results_df['|R|'].mean(), 4)} ± {round(i_imli_results_df['|R|'].std(), 4)}",
        f"{round(i_imli_results_df['Largest rule size'].mean(), 4)} ± {round(i_imli_results_df['Largest rule size'].std(), 4)}",
        f"{round(i_imli_results_df['Accuracy'].mean(), 4)} ± {round(i_imli_results_df['Accuracy'].std(), 4)}",
        f"{round(i_imli_results_df['Training time'].mean(), 4)} ± {round(i_imli_results_df['Training time'].std(), 4)}",
        ''
    ]], columns=columns)
    i_imli_results_df = pd.concat([i_imli_results_df, i_imli_averages])

    imlib_averages = pd.DataFrame([[
        'Averages',
        '',
        f"{round(imlib_results_df['Average deviation of rule sizes'].mean(), 4)} ± {round(imlib_results_df['Average deviation of rule sizes'].std(), 4)}",
        f"{round(imlib_results_df['Standard deviation of rule sizes'].mean(), 4)} ± {round(imlib_results_df['Standard deviation of rule sizes'].std(), 4)}",
        f"{round(imlib_results_df['Entropy of rule sizes'].mean(), 4)} ± {round(imlib_results_df['Entropy of rule sizes'].std(), 4)}",
        f"{round(imlib_results_df['Number of rules'].mean(), 4)} ± {round(imlib_results_df['Number of rules'].std(), 4)}",
        f"{round(imlib_results_df['|R|'].mean(), 4)} ± {round(imlib_results_df['|R|'].std(), 4)}",
        f"{round(imlib_results_df['Largest rule size'].mean(), 4)} ± {round(imlib_results_df['Largest rule size'].std(), 4)}",
        f"{round(imlib_results_df['Accuracy'].mean(), 4)} ± {round(imlib_results_df['Accuracy'].std(), 4)}",
        f"{round(imlib_results_df['Training time'].mean(), 4)} ± {round(imlib_results_df['Training time'].std(), 4)}",
        ''
    ]], columns=columns)
    imlib_results_df = pd.concat([imlib_results_df, imlib_averages])

    i_imlib_averages = pd.DataFrame([[
        'Averages',
        '',
        f"{round(i_imlib_results_df['Average deviation of rule sizes'].mean(), 4)} ± {round(i_imlib_results_df['Average deviation of rule sizes'].std(), 4)}",
        f"{round(i_imlib_results_df['Standard deviation of rule sizes'].mean(), 4)} ± {round(i_imlib_results_df['Standard deviation of rule sizes'].std(), 4)}",
        f"{round(i_imlib_results_df['Entropy of rule sizes'].mean(), 4)} ± {round(i_imlib_results_df['Entropy of rule sizes'].std(), 4)}",
        f"{round(i_imlib_results_df['Number of rules'].mean(), 4)} ± {round(i_imlib_results_df['Number of rules'].std(), 4)}",
        f"{round(i_imlib_results_df['|R|'].mean(), 4)} ± {round(i_imlib_results_df['|R|'].std(), 4)}",
        f"{round(i_imlib_results_df['Largest rule size'].mean(), 4)} ± {round(i_imlib_results_df['Largest rule size'].std(), 4)}",
        f"{round(i_imlib_results_df['Accuracy'].mean(), 4)} ± {round(i_imlib_results_df['Accuracy'].std(), 4)}",
        f"{round(i_imlib_results_df['Training time'].mean(), 4)} ± {round(i_imlib_results_df['Training time'].std(), 4)}",
        ''
    ]], columns=columns)
    i_imlib_results_df = pd.concat([i_imlib_results_df, i_imlib_averages])

    # save results in csv file
    imli_results_df.to_csv(imli_results_path, index=False)
    i_imli_results_df.to_csv(i_imli_results_path, index=False)
    imlib_results_df.to_csv(imlib_results_path, index=False)
    i_imlib_results_df.to_csv(i_imlib_results_path, index=False)