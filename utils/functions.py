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