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
imli_results_path = f'./drafts/tests/imli_vs_ikkrr_vs_iminds3_results/imli_vs_ikkrr_vs_iminds3_2_results/{database_name}_imli.csv'
ikkrr_results_path = f'./drafts/tests/imli_vs_ikkrr_vs_iminds3_results/imli_vs_ikkrr_vs_iminds3_2_results/{database_name}_ikkrr.csv'
iminds3_results_path = f'./drafts/tests/imli_vs_ikkrr_vs_iminds3_results/imli_vs_ikkrr_vs_iminds3_2_results/{database_name}_iminds3.csv'

# import dataset
Xy = pd.read_csv(database_path)
X = Xy.drop(['Class'], axis=1)
y = Xy['Class']

# dataframes that will save the imli, ikkrr and iminds3 results, respectively
columns = ['Configuration', 'Rules size', 'Number of rules', '|R|', 'Largest rule size', 'Accuracy', 'Training time']
imli_results_df = pd.DataFrame([], columns=columns)
ikkrr_results_df = pd.DataFrame([], columns=columns)
iminds3_results_df = pd.DataFrame([], columns=columns)

# TODO
# ...
