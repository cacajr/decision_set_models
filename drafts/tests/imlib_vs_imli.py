import sys, os
if not sys.path[0] == os.path.abspath('...'):
    sys.path.insert(0, os.path.abspath('...'))

from models.imli import IMLI
from models.imlib import IMLIB

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# training configurations
database_name = 'pima'
categorical_columns_index = []
number_lines_per_partition = [8, 16]
max_rule_set_sizes = [1, 2, 3]
# max_sizes_each_rule = [1, 2, 3]
rules_accuracy_weights = [5, 10]
number_quantiles_ordinal_columns = 5
balance_instances = True
balance_instances_seed = 21
number_realizations = 10
database_path = f'./databases/{database_name}.csv'
imli_results_path = f'./drafts/tests/imlib_vs_imli_results/{database_name}_imli.csv'
imlib_results_path = f'./drafts/tests/imlib_vs_imli_results/{database_name}_imlib.csv'

# import dataset
Xy = pd.read_csv(database_path)
X = Xy.drop(['Class'], axis=1)
y = Xy['Class']

# dataframes that will save the imli and imlib results, respectively
columns = ['Configuration', 'Rules size', 'Rule set size', 'Sum rules size', 'Larger rule size', 'Accuracy', 'Training time']
imli_results_df = pd.DataFrame([], columns=columns)
imlib_results_df = pd.DataFrame([], columns=columns)

for lpp in tqdm(number_lines_per_partition, desc=f'lpp: 0 | mrss: 0 | raw: 0 | mser: 0 '):
    for mrss in tqdm(max_rule_set_sizes, desc=f'lpp: {lpp} | mrss: 0 | raw: 0 | mser: 0'):
        for raw in tqdm(rules_accuracy_weights, desc=f'lpp: {lpp} | mrss: {mrss} | raw: 0 | mser: 0'):
            for r in tqdm(range(number_realizations), desc=f'lpp: {lpp} | mrss: {mrss} | raw: {raw} | mser: 0'):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

                imli_model = IMLI(
                    max_rule_set_size = mrss,
                    rules_accuracy_weight = raw,
                    categorical_columns_index = categorical_columns_index,
                    number_quantiles_ordinal_columns = number_quantiles_ordinal_columns,
                    number_lines_per_partition = lpp,
                    balance_instances = balance_instances,
                    balance_instances_seed = balance_instances_seed
                )

                imli_model.fit(X_train, y_train)

                imli_larger_rule_size = imli_model.get_larger_rule_size()
                imli_result = pd.DataFrame([[
                    f'lpp: {lpp} | mrss: {mrss} | raw: {raw}',
                    imli_model.get_rules_size(),
                    imli_model.get_rule_set_size(),
                    imli_model.get_sum_rules_size(),
                    imli_larger_rule_size,
                    imli_model.score(X_test, y_test),
                    imli_model.get_total_time_solver_solutions()
                ]], columns=columns)

                imli_results_df = pd.concat([imli_results_df, imli_result])
            
                # case the larger rule be 1, so iterate mser once
                imli_larger_rule_size = 2 if imli_larger_rule_size == 1 else imli_larger_rule_size
                imlib_best_result = pd.DataFrame([[
                    f'lpp: {lpp} | mrss: {mrss} | raw: {raw} | mser: 0', [], 0, 0, 0, 0.0, 0.0
                ]], columns=columns)

                for mser in tqdm(range(1, imli_larger_rule_size), desc=f'lpp: {lpp} | mrss: {mrss} | raw: {raw} | mser: ?'):
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

                    imlib_model.fit(X_train, y_train)

                    imlib_accuracy = imlib_model.score(X_test, y_test)

                    if imlib_accuracy > imlib_best_result['Accuracy'].iloc[0]:
                        imlib_best_result = pd.DataFrame([[
                            f'lpp: {lpp} | mrss: {mrss} | raw: {raw} | mser: {mser}',
                            imlib_model.get_rules_size(),
                            imlib_model.get_rule_set_size(),
                            imlib_model.get_sum_rules_size(),
                            imlib_model.get_larger_rule_size(),
                            imlib_accuracy,
                            imlib_model.get_total_time_solver_solutions()
                        ]], columns=columns)

                imlib_results_df = pd.concat([imlib_results_df, imlib_best_result])

            imli_averages = pd.DataFrame([[
                'Averages',
                '',
                imli_results_df['Rule set size'].iloc[-1: -number_realizations-1: -1].mean(),
                imli_results_df['Sum rules size'].iloc[-1: -number_realizations-1: -1].mean(),
                imli_results_df['Larger rule size'].iloc[-1: -number_realizations-1: -1].mean(),
                imli_results_df['Accuracy'].iloc[-1: -number_realizations-1: -1].mean(),
                imli_results_df['Training time'].iloc[-1: -number_realizations-1: -1].mean()
            ]], columns=columns)

            imli_results_df = pd.concat([imli_results_df, imli_averages])

            imlib_averages = pd.DataFrame([[
                'Averages',
                '',
                imlib_results_df['Rule set size'].iloc[-1: -number_realizations-1: -1].mean(),
                imlib_results_df['Sum rules size'].iloc[-1: -number_realizations-1: -1].mean(),
                imlib_results_df['Larger rule size'].iloc[-1: -number_realizations-1: -1].mean(),
                imlib_results_df['Accuracy'].iloc[-1: -number_realizations-1: -1].mean(),
                imlib_results_df['Training time'].iloc[-1: -number_realizations-1: -1].mean()
            ]], columns=columns)

            imlib_results_df = pd.concat([imlib_results_df, imlib_averages])

# save results in csv file
imli_results_df.to_csv(imli_results_path, index=False)
imlib_results_df.to_csv(imlib_results_path, index=False)