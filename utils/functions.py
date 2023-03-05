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