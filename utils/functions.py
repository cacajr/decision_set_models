import pandas as pd
import numpy as np


def findIndexUniqueValues(series, unique_values):
    indexes = []

    for i_series_value, series_value in enumerate(series):
        for i_unique_value, unique_value in enumerate(unique_values):
            if series_value == unique_value:
                indexes.append(i_series_value)
                unique_values = np.delete(unique_values, i_unique_value)

                if unique_values.size == 0:
                    return np.array(indexes)

    return np.array(indexes)

def unique_abs_numbers_ordered_by_appearance(list_of_lists):
    seen = set()
    unique_numbers = []
    
    for sublist in list_of_lists:
        for num in sublist:
            if abs(num) not in seen:
                seen.add(abs(num))
                unique_numbers.append(abs(num))
    
    return unique_numbers

# list_dict_vars_vals
# if columns are 1 and 2, then combinations are:
# [{1:0, 2:0}, {1:0, 2:1}, {1:1, 2:0}, {1:1, 2:1}]
# columns_range
# if are three columns with three columns binarized, then:
# [3, 6, 9]
def remove_invalid_combinations_categorical_variables(
        categorical_index_original_columns, 
        list_dict_vars_vals, 
        columns_range,
        dict_original_to_binarized_values
    ):
    list_bin_cols_same_range = get_bin_cols_by_orig_cols_range(categorical_index_original_columns, list_dict_vars_vals, columns_range)
    new_list_dict_vars_vals = []

    for bin_cols in list_bin_cols_same_range:
        if len(bin_cols) > 1:
            i_bin_cols = [n - 1 for n in bin_cols]

            for dict_vars_vals in list_dict_vars_vals:
                comb_vals = [dict_vars_vals[bin] for bin in bin_cols]

                for key in dict_original_to_binarized_values:
                    # in this moment, we verify if the combination exists
                    if np.all(
                        dict_original_to_binarized_values[key][i_bin_cols] == comb_vals
                    ):
                        new_list_dict_vars_vals.append(dict_vars_vals)

    return new_list_dict_vars_vals

# list_dict_vars_vals
# if columns are 1 and 2, then combinations are:
# [{1:0, 2:0}, {1:0, 2:1}, {1:1, 2:0}, {1:1, 2:1}]
# columns_range
# if are three columns with three columns binarized, then:
# [3, 6, 9]
def remove_invalid_combinations_ordinal_variables(ordinal_index_columns, list_dict_vars_vals, columns_range):
    pass

def get_bin_cols_by_orig_cols_range(index_original_columns, list_dict_vars_vals, columns_range):
    list_bin_cols_same_range = [[] for _ in index_original_columns]
    for i_cat_col in index_original_columns:
        cols_in_range = []
        for bin_col in list_dict_vars_vals[0]:
            if bin_col >= columns_range[i_cat_col][0] and bin_col <= columns_range[i_cat_col][1]:
                list_bin_cols_same_range[i_cat_col].append(bin_col)
                cols_in_range.append(bin_col)
        
        # if the col is in some range, so we can to remove, because it cant be in another range
        for col in cols_in_range:
            del list_dict_vars_vals[0][col]
    
    for list in list_bin_cols_same_range:
        list.sort()

    return list_bin_cols_same_range