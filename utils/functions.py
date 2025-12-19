import numpy as np
from itertools import product as _product


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

def generate_consistent_assignments(vars_list, binarized_columns_positions, categorical_columns_index):
    # Generates all assignments for vars_list that respect categorical (one-hot)
    # and ordinal monotonic constraints.
    if not vars_list:
        yield {}
        return

    positions = binarized_columns_positions
    vars_set = set(vars_list)

    # build groups per original feature that intersect vars_set
    groups = []
    for i, pos in enumerate(positions):
        if not pos:
            continue
        inter = [p for p in pos if p in vars_set]
        if not inter:
            continue
        if i in categorical_columns_index and len(pos) > 1:
            groups.append(('onehot', pos, inter))
        elif len(pos) > 1:
            groups.append(('ordinal', pos, inter))
        else:
            groups.append(('single', pos, inter))

    # any vars not in any group -> singletons
    grouped = set()
    for _, _, inter in groups:
        grouped.update(inter)
    remaining = [v for v in vars_list if v not in grouped]
    for v in remaining:
        groups.append(('single', [v], [v]))

    # for each group, build list of possible assignments for the subset inter
    group_assignments = []
    for gtype, full_pos, inter in groups:
        assignments = []
        if gtype == 'single':
            v = inter[0]
            assignments = [{v: 0}, {v: 1}]
        elif gtype == 'onehot':
            # one-hot on full_pos: at most one 1 among full_pos
            # enumerate choices of which position is 1 or none, then project to inter
            opts = [None] + full_pos
            seen = set()
            for chosen in opts:
                assign = {}
                for p in inter:
                    val = 1 if p == chosen else 0
                    assign[p] = val
                key = tuple(assign.items())
                if key not in seen:
                    seen.add(key)
                    assignments.append(assign)
        elif gtype == 'ordinal':
            # full patterns are suffixes of ones (zeros -> ones) on full_pos
            seen = set()
            n = len(full_pos)
            for k in range(n + 1):
                # k = number of trailing ones
                full_pattern = {p: (1 if idx >= n - k else 0) for idx, p in enumerate(full_pos)}
                assign = {p: full_pattern[p] for p in inter}
                key = tuple(assign.items())
                if key not in seen:
                    seen.add(key)
                    assignments.append(assign)
        group_assignments.append(assignments)

    # cartesian product across group assignments
    for combo in _product(*group_assignments):
        merged = {}
        for part in combo:
            merged.update(part)
        yield merged