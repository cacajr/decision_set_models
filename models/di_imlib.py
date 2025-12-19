import pandas as pd
from utils.binarize import Binarize
from pysat.formula import IDPool
from pysat.formula import WCNF
import numpy as np
from pysat.examples.rc2 import RC2
import time
from utils.functions import unique_abs_numbers_ordered_by_appearance, generate_consistent_assignments
import re


class DI_IMLIB:
    ''' Description of Params

        max_rule_set_size: must be a integer and represents the maximum number of 
        rules/clauses that the model will to generate

        max_size_each_rule: must be a integer and represents the maximum number of
        literals per rule/clause

        rules_size_weight: must be a integer and represents the desired level of 
        rule size. A higher value indicates a smaller expected rule size

        rules_accuracy_weight: must be a integer and represents the desired level of 
        accuracy for the rule. A higher value indicates a higher level of accuracy 
        expected from the rule

        time_out_each_partition: must be an integer and represents the maximum
        time in seconds that Solver has to solve one partition

        categorical_columns_index: must be a list with columns index that have 
        categorical data

        number_quantiles_ordinal_columns: must be an integer that represents the 
        number of quantiles/(columns = quantiles - 1) that the new representation will have

        number_lines_per_partition: must be an integer that represents the number of 
        lines for each partitions. Depending on the number of instances in the training 
        dataset, this number can be modified to maintain balance between the number of 
        instances in the partitions

        balance_instances: must be a boolean that represents whether each 
        partition of the dataset should have balanced classes

        balance_instances_seed: must be a integer that represents the random seed
        number for balancing

    '''
    def __init__(self,
            max_rule_set_size = 2,
            max_size_each_rule = 3,
            rules_size_weight = 1,
            rules_accuracy_weight = 10,
            time_out_each_partition = 1024,
            categorical_columns_index=[],
            number_quantiles_ordinal_columns=5,
            number_lines_per_partition = 8,
            balance_instances = True,
            balance_instances_seed = None
        ):

        self.__validate_init_params(
            max_rule_set_size,
            max_size_each_rule,
            rules_size_weight,
            rules_accuracy_weight,
            time_out_each_partition,
            categorical_columns_index,
            number_quantiles_ordinal_columns,
            number_lines_per_partition,
            balance_instances,
            balance_instances_seed
        )

        self.__max_rule_set_size = max_rule_set_size
        self.__max_size_each_rule = max_size_each_rule
        self.__rules_size_weight = rules_size_weight
        self.__rules_accuracy_weight = rules_accuracy_weight
        self.__time_out_each_partition = time_out_each_partition
        self.__categorical_columns_index = categorical_columns_index
        self.__number_quantiles_ordinal_columns = number_quantiles_ordinal_columns
        self.__number_lines_per_partition = number_lines_per_partition
        self.__balance_instances = balance_instances
        self.__balance_instances_seed = balance_instances_seed

        self.__dataset_binarized = Binarize

        self.__literals = IDPool()
        self.__solver_solution = list([])
        self.__total_time_solver_solutions = 0.0

        self.__rules_features = list([])
        self.__rules_columns = list([])
        self.__rules_features_string = str('')

    def __validate_init_params(self,
            max_rule_set_size,
            max_size_each_rule,
            rules_size_weight,
            rules_accuracy_weight,
            time_out_each_partition,
            categorical_columns_index,
            number_quantiles_ordinal_columns,
            number_lines_per_partition,
            balance_instances,
            balance_instances_seed
        ):

        if type(max_rule_set_size) is not int:
            raise Exception('Param max_rule_set_size must be an int')
        if type(max_size_each_rule) is not int:
            raise Exception('Param max_size_each_rule must be an int')
        if type(rules_size_weight) is not int:
            raise Exception('Param rules_size_weight must be an int')
        if type(rules_accuracy_weight) is not int:
            raise Exception('Param rules_accuracy_weight must be an int')
        if type(time_out_each_partition) is not int:
            raise Exception('Param time_out_each_partition must be an int')
        if type(categorical_columns_index) is not list:
            raise Exception('Param categorical_columns_index must be a list')
        if type(number_quantiles_ordinal_columns) is not int:
            raise Exception('Param number_quantiles_ordinal_columns must be an int')
        if number_quantiles_ordinal_columns <= 1:
            raise Exception('Param number_quantiles_ordinal_columns must be greater than 1')
        if type(number_lines_per_partition) is not int:
            raise Exception('Param number_lines_per_partition must be an int')
        if type(balance_instances) is not bool:
            raise Exception('Param balance_instances must be a bool')
        if balance_instances_seed != None and type(balance_instances_seed) is not int:
            raise Exception('Param balance_instances_seed must be an int or None')

    def fit(self, X, y):
        number_partitions = int(np.ceil(X.index.size/self.__number_lines_per_partition))

        self.__dataset_binarized = Binarize(
            data_frame=X,
            series=y,
            categorical_columns_index=self.__categorical_columns_index,
            number_quantiles_ordinal_columns=self.__number_quantiles_ordinal_columns,
            number_partitions=number_partitions,
            balance_instances=self.__balance_instances,
            balance_instances_seed=self.__balance_instances_seed
        )

        X_normal_partitions = self.__dataset_binarized.get_normal_instances()
        X_opposite_partitions = self.__dataset_binarized.get_opposite_instances()
        y_partitions = self.__dataset_binarized.get_classes()

        for _ in range(self.__max_rule_set_size):

            for index_partition, (X_normal_partition, y_partition) in enumerate(
                    zip(
                        X_normal_partitions,
                        y_partitions
                    )
                ):

                wcnf_formula = self.__create_wcnf_formula(
                    self.__solver_solution,
                    X_normal_partition,
                    y_partition
                )

                # TODO: add a new MaxSAT solver option

                # WARNING: this line is used just to debug --------------------------------
                # wcnf_formula.to_file('./models/wcnf_formula.wcnf')
                # -------------------------------------------------------------------------

                solver = RC2(wcnf_formula)

                start = time.time()
                self.__solver_solution = solver.compute()   # TODO: add time out calculate
                end = time.time()

                self.__total_time_solver_solutions += end - start

                if self.__solver_solution == None:
                    raise Exception(f'Partition {index_partition + 1} unsatisfiable')

                if index_partition == number_partitions - 1:
                    self.__create_rules(X_normal_partition)

                self.__reset_literals()

            self.__solver_solution = list([])

            X_normal_partitions, X_opposite_partitions, y_partitions, has_covered_sample = self.__remove_covered_samples(
                X_normal_partitions,
                X_opposite_partitions,
                y_partitions
            )

            # remove rule that did not cover new samples and stops generating new rules
            if not has_covered_sample:
                self.__rules_features.pop()
                self.__rules_columns.pop()

                break

        self.__prune_rules()
        self.__rules_features_string = self.__create_rules_features_string(
            self.__rules_features
        )

    def __create_wcnf_formula(self, previous_solution, X_norm, y):
        features = self.__dataset_binarized.get_normal_features_label()
        wcnf_formula = WCNF()

        # (7.5) (15)
        for i in range(1):    # i ∈ {1, ..., m}
            for j in range(self.__max_size_each_rule):  # j ∈ {1, ..., l}
                clause = []
                for t in range(len(features)):  # t ∈ Φ ...
                    clause.append(self.__x(i,j,t))
                clause.append(self.__x(i,j))    # ... U {*}

                wcnf_formula.append(clause)
        
        # (7.5.1) (17)
        if len(previous_solution) == 0:
            for i in range(1):
                for j in range(self.__max_size_each_rule):
                    for t in range(len(features)):
                        wcnf_formula.append([-self.__x(i,j,t)], weight=self.__rules_size_weight)
                    wcnf_formula.append([self.__x(i,j)], weight=self.__rules_size_weight)
        else:
            x_literals = self.__get_x_literals(features)
            for literal in x_literals:
                wcnf_formula.append([literal], weight=self.__rules_size_weight)
        
        # (7.6) (16)
        for i in range(1):
            for j in range(self.__max_size_each_rule):
                for t in range(len(features)):
                    for tl in range(t+1, len(features)):
                        wcnf_formula.append([-self.__x(i,j,t), -self.__x(i,j,tl)])
                    wcnf_formula.append([-self.__x(i,j,t), -self.__x(i,j)])
        
        # (7.7) (18)
        for i in range(1):
            clause = []
            for j in range(self.__max_size_each_rule):
                clause.append(-self.__x(i,j))
            
            wcnf_formula.append(clause)

        # (7.8) (19)
        for i in range(1):
            for j in range(self.__max_size_each_rule):
                for t in range(len(features)):
                    for w, instance in enumerate(X_norm):    # w ∈ P U N
                        literal_y = int
                        if instance[t] == 0:
                            literal_y = -self.__y(i,j,w)
                        else:
                            literal_y = self.__y(i,j,w)

                        wcnf_formula.append(
                            [-self.__x(i,j,t), -self.__p(i,j), literal_y]
                        )
                        wcnf_formula.append(
                            [-self.__x(i,j,t), self.__p(i,j), -literal_y]
                        )

        # (7.9) (20)
        for i in range(1):
            for j in range(self.__max_size_each_rule):
                for w in range(len(X_norm)):
                    wcnf_formula.append([-self.__x(i,j), self.__y(i,j,w)])

        # (7.10) (21)
        for i in range(1):
            for w in range(len(X_norm)):
                clauses = []
                clause = [self.__z(i,w)]
                for j in range(self.__max_size_each_rule):
                    clauses.append([-self.__z(i,w), self.__y(i,j,w)])
                    clause.append(-self.__y(i,j,w))
                clauses.append(clause)

                wcnf_formula.extend(clauses)

        # (7.11) (22)
        for u in np.where(y == 1)[0]:    # u ∈ P
            clause = []
            for i in range(1):
                clause.append(self.__z(i,u))
            
            wcnf_formula.append(clause, weight= self.__rules_accuracy_weight)

        # (7.12) (23)
        for v in np.where(y == 0)[0]:    # v ∈ N
            for i in range(1):
                wcnf_formula.append(
                    [-self.__z(i,v)], 
                    weight= self.__rules_accuracy_weight
                )

        # (7.5.2) restrictions that ensure that the rules are inconsistent
        if len(self.__rules_columns) > 0:
            n = 0
            for rule in self.__rules_columns:
                clause = []
                for column in rule:
                    t = abs(column) - 1
                    for j in range(self.__max_size_each_rule):
                        wcnf_formula.append(
                            [-self.__a(n), self.__x(0,j,t)]
                        )
                        wcnf_formula.append(
                            [-self.__a(n), self.__p(0,j) if column < 0 else -self.__p(0,j)]
                        )

                        clause.append(self.__a(n))
                        n += 1

                wcnf_formula.append(clause)

        return wcnf_formula

    def __x(self, i, j, t = None):
        if t != None:
            return self.__literals.id(f'x{i}{j}{t}')
        
        return self.__literals.id(f'x{i}{j}*')
    
    def __p(self, i, j):
        return self.__literals.id(f'p{i}{j}')
    
    def __y(self, i, j, w):
        return self.__literals.id(f'y{i}{j}{w}')
    
    def __z(self, i, w):
        return self.__literals.id(f'z{i}{w}')
    
    # additional variables of the tseytin transformation
    def __a(self, n):
        return self.__literals.id(f'a{n}')

    def __reset_literals(self):
        self.__literals = IDPool()

    def __create_rules(self, X_norm):
        normal_features = self.__dataset_binarized.get_normal_features_label()
        opposite_features = self.__dataset_binarized.get_opposite_features_label()

        x_literals = self.__get_x_literals(normal_features)
        p_literals = self.__get_p_literals(normal_features, X_norm)

        rules_features = [[] for _ in range(1)]
        rules_columns = [[] for _ in range(1)]
        
        for i in range(1):    # i ∈ {1, ..., m}
            for j in range(self.__max_size_each_rule):  # j ∈ {1, ..., l}
                for t in range(len(normal_features)):  # t ∈ Φ U {*}
                    if self.__x(i,j,t) in x_literals:
                        if self.__p(i,j) in p_literals:
                            rules_features[i].append(normal_features[t])
                            rules_columns[i].append(t+1)
                        else:
                            rules_features[i].append(opposite_features[t])
                            rules_columns[i].append(-(t+1))

        self.__rules_features += rules_features
        self.__rules_columns += rules_columns
    
    def __get_x_literals(self, features):
        number_features = len(features)
        start = 0
        end = (number_features + 1) * 1 * self.__max_size_each_rule
        
        return self.__solver_solution[start:end]

    def __get_p_literals(self, features, X_norm):
        number_features = len(features)
        number_instances = len(X_norm)
        start = (number_features + 1) * 1 * self.__max_size_each_rule
        end = start + ((number_instances + 1) * 1 * self.__max_size_each_rule)

        p_literals_region = self.__solver_solution[start:end]

        return p_literals_region[1::number_instances + 1]

    def __prune_rules(self):
        normal_features = self.__dataset_binarized.get_normal_features_label()
        opposite_features = self.__dataset_binarized.get_opposite_features_label()

        # removing repeated literal in the same rule: (... A ∧ A ...)
        for i_rule, rule in enumerate(self.__rules_columns):
            self.__rules_columns[i_rule] = list(set(rule))

        # removing normal and opposite literals in the same rule: (... A ∧ ¬A ...)
        rules_falsy = []
        for rule in self.__rules_columns:
            for column in rule:
                if -column in rule:
                    rules_falsy.append(rule)
                    break
        for rule in rules_falsy:
            self.__rules_columns.remove(rule)

        # removing redundances in the same rule: (A <= 2 ∧ A <= 3) and (A > 2 ∧ A > 3)
        ordinal_normal_index_columns = np.where([
            feat.__contains__('<=') 
            for feat in normal_features
        ])[0]
        for i in range(0, len(ordinal_normal_index_columns), self.__number_quantiles_ordinal_columns-1):
            for rule in self.__rules_columns:
                ordinal_normal_columns = []
                ordinal_opposite_columns = []
                for column in rule:
                    if column > 0:
                        if column - 1 in ordinal_normal_index_columns[i:i+self.__number_quantiles_ordinal_columns-1]:
                            ordinal_normal_columns.append(column)
                    else:
                        if abs(column) - 1 in ordinal_normal_index_columns[i:i+self.__number_quantiles_ordinal_columns-1]:
                            ordinal_opposite_columns.append(column)

                if len(ordinal_normal_columns) > 0:
                    for column in ordinal_normal_columns:
                        rule.remove(column)
                    rule.append(min(ordinal_normal_columns))
                if len(ordinal_opposite_columns) > 0:
                    for column in ordinal_opposite_columns:
                        rule.remove(column)
                    rule.append(min(ordinal_opposite_columns))

        # update self.__rules_features
        rules_features = []
        for rule in self.__rules_columns:
            rule_features = []
            for column in rule:
                if column > 0:
                    rule_features.append(normal_features[column-1])
                else:
                    rule_features.append(opposite_features[abs(column)-1])
            rules_features.append(rule_features)

        self.__rules_features = rules_features

    def __create_rules_features_string(self, rules_features):
        rules_features_string = ''
        for i in range(len(rules_features)):
            rules_features_string += '('
            for j in range(len(rules_features[i])):
                rules_features_string += str(rules_features[i][j])

                if j < len(rules_features[i]) - 1:
                    rules_features_string += ' and '
                else:
                    rules_features_string += ')'
            
            if i < len(rules_features) - 1:
                rules_features_string += ' or '
        
        return rules_features_string

    def __remove_covered_samples(self, X_norm, X_oppo, y):
        new_X_norm, new_X_oppo, new_y, has_covered_sample = [], [], [], False

        for X_normal_partition, X_opposite_partition, y_partition in zip(X_norm, X_oppo, y):
            
            index_not_covered_samples = []
            for index_sample, (normal_sample, opposite_sample, predict) in enumerate(
                    zip(
                        X_normal_partition, 
                        X_opposite_partition, 
                        y_partition
                    )
                ):

                partial_predict = self.__aplicate_DNF_rules(normal_sample, opposite_sample)

                # we consider rules in DNF in this verifications
                if predict == 1 and partial_predict == predict: # in this case, this sample was covered
                    has_covered_sample = True
                    continue
                # if predict == 0 and partial_predict == 1:   # in this case, this sample will never be covered
                #     continue
                    
                index_not_covered_samples.append(index_sample)

            new_X_norm.append(X_normal_partition[index_not_covered_samples])
            new_X_oppo.append(X_opposite_partition[index_not_covered_samples])
            new_y.append(y_partition[index_not_covered_samples])
        
        return new_X_norm, new_X_oppo, new_y, has_covered_sample

    def get_rules(self):
        return self.__rules_features_string

    def predict(self, instance):
        self.__validate_instance(instance)

        binarized_to_original_class = self.__dataset_binarized.get_original_to_binarized_values()[-1]
        normal_instance_binarized, opposite_instance_binarized = self.__binarize_instance(instance)
        
        predict = binarized_to_original_class[
            self.__aplicate_DNF_rules(normal_instance_binarized, opposite_instance_binarized)
        ]

        return predict

    def __validate_instance(self, instance):
        qtts_binarized_feat = self.__dataset_binarized.get_qtts_binarized_feat_per_original_feat()

        if type(instance) not in [list, pd.array, np.array, np.ndarray]:
            raise Exception('Param instance must be a list, pd.array, np.array or np.ndarray')
        if len(instance) != len(qtts_binarized_feat):
            raise Exception('Param instance with number of features invalid')
        
        return True

    def __binarize_instance(self, instance):
        normal_instance_binarized = []

        original_to_binarized = self.__dataset_binarized.get_original_to_binarized_values()[:-1]
        qtts_binarized_feat = self.__dataset_binarized.get_qtts_binarized_feat_per_original_feat()

        for i_num_feat, num_feat in enumerate(qtts_binarized_feat):
            if num_feat == 0:
                continue
            elif num_feat == 1:
                if instance[i_num_feat] in original_to_binarized[i_num_feat].keys():
                    normal_instance_binarized.append(original_to_binarized[i_num_feat][instance[i_num_feat]])
                else:
                    normal_instance_binarized.append(0)
            elif i_num_feat in self.__categorical_columns_index:
                if instance[i_num_feat] in original_to_binarized[i_num_feat].keys():
                    for num in original_to_binarized[i_num_feat][instance[i_num_feat]]:
                        normal_instance_binarized.append(num)
                else:
                    last_key = list(original_to_binarized[i_num_feat].keys())[-1]
                    for num in original_to_binarized[i_num_feat][last_key]:
                        normal_instance_binarized.append(0)
            elif type(instance[i_num_feat]) in [
                    int, np.int16, np.int32, np.int64, float, 
                    np.float16, np.float32, np.float64
                ]:

                for quantis in original_to_binarized[i_num_feat].values():
                    if instance[i_num_feat] <= quantis:
                        normal_instance_binarized.append(1)
                    else:
                        normal_instance_binarized.append(0)
            else:
                raise Exception(f'Feature with value {instance[i_num_feat]} invalid')

        opposite_instance_binarized = [
            0 if num == 1 else 1
            for num in normal_instance_binarized
        ]

        return normal_instance_binarized, opposite_instance_binarized

    def __aplicate_DNF_rules(self, normal_instance_binarized, opposite_instance_binarized):
        predict = 0
        for rule_columns in self.__rules_columns:
            for column in rule_columns:
                if column < 0:
                    if opposite_instance_binarized[abs(column) - 1] == 0:
                        predict = 0
                        break
                    else:
                        predict = 1
                else:
                    if normal_instance_binarized[abs(column) - 1] == 0:
                        predict = 0
                        break
                    else:
                        predict = 1

            if predict == 1:
                break

        return predict

    def score(self, X_test = pd.DataFrame, y_test = pd.Series):
        if type(X_test) != pd.DataFrame or type(y_test) != pd.Series:
            raise Exception(
                'Params X_test and y_test must be a pd.DataFrame and pd.Series, respectively'
            )

        hits_count = 0
        for i_line in range(X_test.index.size):
            predict = self.predict(X_test.iloc[i_line].values)

            if predict == y_test.values[i_line]:
                hits_count += 1

        return hits_count/y_test.size

    def get_sufficient_reasons(self, instance):
        self.__validate_instance(instance)

        normal_instance_binarized, opposite_instance_binarized = self.__binarize_instance(instance)        
        predict = self.__aplicate_DNF_rules(normal_instance_binarized, opposite_instance_binarized)

        # just to debugg -----------------------------------------------------------------------------------------------
        # print()
        # instance_values = []
        # for rule in self.__rules_columns:
        #     r = []
        #     for col in rule:
        #         r.append({
        #             col: opposite_instance_binarized[abs(col) - 1] if col < 0 else normal_instance_binarized[col - 1]
        #         })
        #     instance_values.append(r)
        # print('Rules and Values:')
        # print(instance_values)
        # --------------------------------------------------------------------------------------------------------------

        unique_cols = unique_abs_numbers_ordered_by_appearance(self.__rules_columns)
        # print()
        # print('Unique Columns:', unique_cols)
        removed_cols = []
        for col in unique_cols.copy():
            simplify_rules = self.__conditioner(normal_instance_binarized, opposite_instance_binarized, removed_cols + [col])

            # print()
            # print('Simplify Rules with Variables:')
            # print(removed_cols + [col])
            # print('Rules simplified:')
            # print(simplify_rules)

            if ((predict == 0 and not self.__isConsistent(simplify_rules)) or 
                (predict == 1 and self.__isValid(simplify_rules, removed_cols + [col]))):
                unique_cols.remove(col)
                removed_cols.append(col)

        sufficient_reasons = self.__create_sufficient_reasons_feature_string(
            normal_instance_binarized, opposite_instance_binarized, 
            unique_cols, 
            predict
        )

        return sufficient_reasons
    
    def __conditioner(self, normal_instance, opposite_instance, vars):
        new_rules_columns = []

        for rule in self.__rules_columns:
            new_rule = []
            for col in rule:
                if abs(col) not in vars:
                    if col < 0:
                        new_rule.append(opposite_instance[abs(col) - 1])
                    else:
                        new_rule.append(normal_instance[col - 1])
                else:
                    new_rule.append(col)
            
            if 0 in new_rule:
                continue

            new_rule = [col for col in new_rule if col != 1]

            new_rules_columns.append(new_rule)
        
        return new_rules_columns
    
    def __isConsistent(self, rules_columns):
        # in this case, all cols in some rule (term) was removed (had all constants 1)
        if [] in rules_columns:
            return True

        # fast per-rule feasibility check (avoid enumerating all assignments)
        positions = self.__dataset_binarized.get_binarized_columns_positions()
        categorical_idx = set(self.__categorical_columns_index)

        # build mapping position -> (feature_idx, index_within_feature, full_pos_list)
        pos2feat = {}
        for feat_idx, pos_list in enumerate(positions):
            for idx_in_feat, p in enumerate(pos_list):
                pos2feat[p] = (feat_idx, idx_in_feat, pos_list)

        def rule_is_feasible(rule):
            # rule: list of signed ints (remaining free literals)
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
                if feat_idx in categorical_idx and len(full_pos) > 1:
                    ones = [p for p, v in reqs.items() if v == 1]
                    # cannot require more than one position equal to 1
                    if len(ones) > 1:
                        return False
                    # otherwise feasible
                    continue

                # ordinal group (prefix of ones then zeros)
                # positions are ordered in full_pos
                pos_index = {p: idx for idx, p in enumerate(full_pos)}
                ones_idx = [pos_index[p] for p, v in reqs.items() if v == 1]
                zeros_idx = [pos_index[p] for p, v in reqs.items() if v == 0]
                if ones_idx and zeros_idx and max(ones_idx) >= min(zeros_idx):
                    return False
                # otherwise feasible
            return True

        for rule in rules_columns:
            if rule_is_feasible(rule):
                return True
        return False

    def __isValid(self, rules_columns, vars):
        # in this case, all rules (terms) was removed (had constant 0)
        if rules_columns == []:
            return False

        # in this case, all cols in some rule (term) was removed (had all constants 1)
        if [] in rules_columns:
            return True

        # iterate only assignments consistent with dependencies
        for assignment in generate_consistent_assignments(vars, self.__dataset_binarized.get_binarized_columns_positions(), self.__categorical_columns_index):
            predict = 0
            for rule in rules_columns:
                sat = True
                for col in rule:
                    val = assignment.get(abs(col), None)
                    if val is None:
                        sat = False
                        break
                    if col < 0:
                        lit = 1 - val
                    else:
                        lit = val
                    if lit != 1:
                        sat = False
                        break
                if sat:
                    predict = 1
                    break
            if predict == 0:
                return False

        return True

    def __create_sufficient_reasons_feature_string(self, normal_instance, opposite_instance, cols, clss):
        sufficient_reasons_features = set()
        for i_r, rule in enumerate(self.__rules_columns):
            for i_c, col in enumerate(rule):
                if abs(col) not in cols:
                    continue
                
                if col < 0:
                    # only those features that contribute to the classification will go to the sufficient reason
                    if opposite_instance[abs(col) - 1] == clss:
                        sufficient_reasons_features.add(self.__rules_features[i_r][i_c])
                else:
                    # only those features that contribute to the classification will go to the sufficient reason
                    if normal_instance[col - 1] == clss:
                        sufficient_reasons_features.add(self.__rules_features[i_r][i_c])

        # removing redundances in the reasons: (A <= 2 ∧ A <= 3) and (A > 2 ∧ A > 3)
        sufficient_reasons_features = self.__remove_reasons_redundances(sufficient_reasons_features)

        sufficient_reasons_string = '('
        for i_f, feat in enumerate(sufficient_reasons_features):
            sufficient_reasons_string += feat

            if i_f < (len(sufficient_reasons_features) - 1):
                sufficient_reasons_string += ' and '
        sufficient_reasons_string += ')'

        return sufficient_reasons_string

    def __remove_reasons_redundances(self, reasons):
        parsed = []
        for literal in reasons:
            match = re.match(r"(\w+)\s*([<>]=?)\s*(-?\d+)", literal)
            if match:
                var, op, value = match.groups()
                value = int(value)
                parsed.append((var, op, value, literal))

        reduced = {}
        
        for var, op, value, literal in parsed:
            if var not in reduced:
                reduced[var] = []
            reduced[var].append((op, value, literal))

        final_literals = set(reasons)
        
        for var, conditions in reduced.items():
            conditions.sort(key=lambda x: x[1])
            
            to_remove = set()
            for i in range(len(conditions) - 1):
                op1, val1, lit1 = conditions[i]
                op2, val2, lit2 = conditions[i + 1]

                if op1 == "<=" and op2 == "<=":
                    to_remove.add(lit2)
                elif op1 == ">=" and op2 == ">=":
                    to_remove.add(lit1)
                elif op1 == ">" and op2 == ">":
                    to_remove.add(lit1)
                elif op1 == "<" and op2 == "<":
                    to_remove.add(lit2)

            final_literals -= to_remove
        
        return list(final_literals)

    # Utility functions -------------------------------------------------------------------

    def get_dataset_binarized(self):
        return self.__dataset_binarized

    def get_total_time_solver_solutions(self):
        return self.__total_time_solver_solutions

    def get_rules_size(self):
        rules_size = []
        for rule in self.__rules_columns:
            rules_size.append(len(rule))

        return rules_size

    def get_rule_set_size(self):
        return len(self.__rules_columns)

    def get_larger_rule_size(self):
        larger_rule_size = 0
        for rule in self.__rules_columns:
            if larger_rule_size < len(rule):
                larger_rule_size = len(rule)

        return larger_rule_size
    
    def get_sum_rules_size(self):
        sum_rules_size = 0
        for rule in self.__rules_columns:
            sum_rules_size += len(rule)

        return sum_rules_size