import sys, os
if not sys.path[0] == os.path.abspath('.'):
    sys.path.insert(0, os.path.abspath('.'))

# from models.imlib import IMLIB
from models.imli import IMLI
from models.ikkrr import IKKRR
from models.iminds3 import IMINDS3

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# training configurations
database_name = 'mushroom'
categorical_columns_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
number_lines_per_partition = [8, 16]
max_rule_set_sizes = [1, 2, 3]
# max_sizes_each_rule = [1, 2, 3]
rules_size_weight = 1
rules_accuracy_weights = [5, 10]
number_quantiles_ordinal_columns = 5
balance_instances = True
balance_instances_seed = 21
number_realizations = 10
database_path = f'./databases/{database_name}.csv'
imli_results_path = f'./drafts/tests/imli_vs_ikkrr_vs_iminds3_results/imli_vs_ikkrr_vs_iminds3_results/{database_name}_imli.csv'
ikkrr_results_path = f'./drafts/tests/imli_vs_ikkrr_vs_iminds3_results/imli_vs_ikkrr_vs_iminds3_results/{database_name}_ikkrr.csv'
iminds3_results_path = f'./drafts/tests/imli_vs_ikkrr_vs_iminds3_results/imli_vs_ikkrr_vs_iminds3_results/{database_name}_iminds3.csv'

# import dataset
Xy = pd.read_csv(database_path)
X = Xy.drop(['Class'], axis=1)
y = Xy['Class']

# dataframes that will save the imli, ikkrr and iminds3 results, respectively
columns = ['Configuration', 'Rules size', 'Number of rules', '|R|', 'Largest rule size', 'Accuracy', 'Training time']
imli_results_df = pd.DataFrame([], columns=columns)
ikkrr_results_df = pd.DataFrame([], columns=columns)
iminds3_results_df = pd.DataFrame([], columns=columns)

for lpp in tqdm(number_lines_per_partition, desc=f'lpp: 0 | mrss: 0 | raw: 0 | mser: 0 '):
    for mrss in tqdm(max_rule_set_sizes, desc=f'lpp: {lpp} | mrss: 0 | raw: 0 | mser: 0'):
        for raw in tqdm(rules_accuracy_weights, desc=f'lpp: {lpp} | mrss: {mrss} | raw: 0 | mser: 0'):
            for r in tqdm(range(number_realizations), desc=f'lpp: {lpp} | mrss: {mrss} | raw: {raw} | mser: 0'):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

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
                ikkrr_model = IKKRR(
                    max_rule_set_size = mrss,
                    rules_size_weight= rules_size_weight,
                    rules_accuracy_weight = raw,
                    categorical_columns_index = categorical_columns_index,
                    number_quantiles_ordinal_columns = number_quantiles_ordinal_columns,
                    number_lines_per_partition = lpp,
                    balance_instances = balance_instances,
                    balance_instances_seed = balance_instances_seed
                )
                iminds3_model = IMINDS3(
                    max_rule_set_size = mrss,
                    rules_size_weight= rules_size_weight,
                    rules_accuracy_weight = raw,
                    categorical_columns_index = categorical_columns_index,
                    number_quantiles_ordinal_columns = number_quantiles_ordinal_columns,
                    number_lines_per_partition = lpp,
                    balance_instances = balance_instances,
                    balance_instances_seed = balance_instances_seed
                )

                imli_model.fit(X_train, y_train)
                ikkrr_model.fit(X_train, y_train)
                iminds3_model.fit(X_train, y_train)

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

                ikkrr_larger_rule_size = ikkrr_model.get_larger_rule_size()
                ikkrr_result = pd.DataFrame([[
                    f'lpp: {lpp} | mrss: {mrss} | raw: {raw}',
                    ikkrr_model.get_rules_size(),
                    ikkrr_model.get_rule_set_size(),
                    ikkrr_model.get_sum_rules_size(),
                    ikkrr_larger_rule_size,
                    ikkrr_model.score(X_test, y_test),
                    ikkrr_model.get_total_time_solver_solutions()
                ]], columns=columns)

                iminds3_larger_rule_size = iminds3_model.get_larger_rule_size()
                iminds3_result = pd.DataFrame([[
                    f'lpp: {lpp} | mrss: {mrss} | raw: {raw}',
                    iminds3_model.get_rules_size(),
                    iminds3_model.get_rule_set_size(),
                    iminds3_model.get_sum_rules_size(),
                    iminds3_larger_rule_size,
                    iminds3_model.score(X_test, y_test),
                    iminds3_model.get_total_time_solver_solutions()
                ]], columns=columns)

                imli_results_df = pd.concat([imli_results_df, imli_result])
                ikkrr_results_df = pd.concat([ikkrr_results_df, ikkrr_result])
                iminds3_results_df = pd.concat([iminds3_results_df, iminds3_result])

            imli_averages = pd.DataFrame([[
                'Averages',
                '',
                imli_results_df['Number of rules'].iloc[-1: -number_realizations-1: -1].mean(),
                imli_results_df['|R|'].iloc[-1: -number_realizations-1: -1].mean(),
                imli_results_df['Largest rule size'].iloc[-1: -number_realizations-1: -1].mean(),
                imli_results_df['Accuracy'].iloc[-1: -number_realizations-1: -1].mean(),
                imli_results_df['Training time'].iloc[-1: -number_realizations-1: -1].mean()
            ]], columns=columns)
            imli_results_df = pd.concat([imli_results_df, imli_averages])

            ikkrr_averages = pd.DataFrame([[
                'Averages',
                '',
                ikkrr_results_df['Number of rules'].iloc[-1: -number_realizations-1: -1].mean(),
                ikkrr_results_df['|R|'].iloc[-1: -number_realizations-1: -1].mean(),
                ikkrr_results_df['Largest rule size'].iloc[-1: -number_realizations-1: -1].mean(),
                ikkrr_results_df['Accuracy'].iloc[-1: -number_realizations-1: -1].mean(),
                ikkrr_results_df['Training time'].iloc[-1: -number_realizations-1: -1].mean()
            ]], columns=columns)
            ikkrr_results_df = pd.concat([ikkrr_results_df, ikkrr_averages])

            iminds3_averages = pd.DataFrame([[
                'Averages',
                '',
                iminds3_results_df['Number of rules'].iloc[-1: -number_realizations-1: -1].mean(),
                iminds3_results_df['|R|'].iloc[-1: -number_realizations-1: -1].mean(),
                iminds3_results_df['Largest rule size'].iloc[-1: -number_realizations-1: -1].mean(),
                iminds3_results_df['Accuracy'].iloc[-1: -number_realizations-1: -1].mean(),
                iminds3_results_df['Training time'].iloc[-1: -number_realizations-1: -1].mean()
            ]], columns=columns)
            iminds3_results_df = pd.concat([iminds3_results_df, iminds3_averages])

# save results in csv file
imli_results_df.to_csv(imli_results_path, index=False)
ikkrr_results_df.to_csv(ikkrr_results_path, index=False)
iminds3_results_df.to_csv(iminds3_results_path, index=False)
