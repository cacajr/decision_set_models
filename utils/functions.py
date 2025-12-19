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

def rule_is_feasible(rule, categorical_columns_index, pos2feat):
    # rule: list of signed ints (remaining free literals)
    # pos2feat: mapping from position to (feature_idx, index_within_feature, full_pos_list)
    # collect requirements per feature
    reqs_by_feat = {}
    for col in rule:
        p = abs(col)
        req_val = 1 if col > 0 else 0
        if p not in pos2feat:
            # unknown position -> treat as infeasible
            return False
        feat_idx, idx_in_feat, full_pos = pos2feat[p]
        if feat_idx not in reqs_by_feat:
            reqs_by_feat[feat_idx] = {'full_pos': full_pos, 'reqs': {}}
        # conflict check for same position
        if p in reqs_by_feat[feat_idx]['reqs'] and reqs_by_feat[feat_idx]['reqs'][p] != req_val:
            return False
        reqs_by_feat[feat_idx]['reqs'][p] = req_val

    # check each group's feasibility independently
    for feat_idx, info in reqs_by_feat.items():
        full_pos = info['full_pos']
        reqs = info['reqs']  # pos -> required value (0/1)

        # single position group
        if len(full_pos) == 1:
            # if there is any requirement on this pos, it's fine (either 0 or 1)
            # no further group constraint
            continue

        # one-hot categorical group (at most one 1)
        if feat_idx in categorical_columns_index and len(full_pos) > 1:
            ones = [p for p, v in reqs.items() if v == 1]
            # cannot require more than one position equal to 1
            if len(ones) > 1:
                return False
            # otherwise feasible
            continue

        # ordinal group (suffix of ones: zeros -> ones)
        # positions are ordered in full_pos
        pos_index = {p: idx for idx, p in enumerate(full_pos)}
        ones_idx = [pos_index[p] for p, v in reqs.items() if v == 1]
        zeros_idx = [pos_index[p] for p, v in reqs.items() if v == 0]
        # for zeros->ones the last zero must come before the first one
        if ones_idx and zeros_idx and max(zeros_idx) >= min(ones_idx):
            return False
        # otherwise feasible
    return True
    