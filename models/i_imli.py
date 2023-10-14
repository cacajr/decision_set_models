import pandas as pd
from utils.binarize import Binarize
from pysat.formula import IDPool
from pysat.formula import WCNF
import numpy as np
from pysat.examples.rc2 import RC2
import time


class I_IMLI:
    ''' Description of Params

        max_rule_set_size: must be a integer and represents the maximum number of 
        rules/clauses that the model will to generate

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
        number of quantiles/columns that the new representation will have

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
        if number_quantiles_ordinal_columns <= 2:
            raise Exception('Param number_quantiles_ordinal_columns must be greater than 2')
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
                    1 - y_partition     # invert y to generate DNF rules
                )

                # TODO: add a new MaxSAT solver option

                # WARNING: this line is used just to debug --------------------------------
                # wcnf_formula.to_file('./wcnf_formula.wcnf')
                # -------------------------------------------------------------------------

                solver = RC2(wcnf_formula)

                start = time.time()
                self.__solver_solution = solver.compute()   # TODO: add time out calculate
                end = time.time()

                self.__total_time_solver_solutions += end - start

                if self.__solver_solution == None:
                    raise Exception(f'Partition {index_partition + 1} unsatisfiable')

                if index_partition == number_partitions - 1:
                    self.__create_DNF_rules()

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
        normal_features = self.__dataset_binarized.get_normal_features_label()
        wcnf_formula = WCNF()

        if len(previous_solution) == 0:
            # (8)
            for l in range(1):   # l ∈ {1, ..., k}
                for j in range(2 * len(normal_features)):  # j ∈ {1, ..., m}
                    wcnf_formula.append([-self.__b(l,j)], weight=self.__rules_size_weight)
        else:
            # creating b literals in idpool
            for l in range(1):   # l ∈ {1, ..., k}
                for j in range(2 * len(normal_features)):  # j ∈ {1, ..., m}
                    self.__b(l,j)
            # (11)
            b_literals = self.__get_b_literals(normal_features)
            for literal in b_literals:
                wcnf_formula.append([literal], weight=self.__rules_size_weight)
        
        # (7)
        for i in range(len(X_norm)):    # i ∈ {1, ..., n}
            wcnf_formula.append([-self.__n(i)], weight=self.__rules_accuracy_weight)

        # (9)
        next_li0 = 0    # next index z literal
        for i in range(len(X_norm)):
            if y[i] == 1:
                for l in range(1):
                    clause = [self.__n(i)]
                    for j in range(len(normal_features)):
                        if X_norm[i][j] == 1:
                            clause.append(self.__b(l,j))
                        else:
                            clause.append(self.__b(l,j+len(normal_features)))
                    wcnf_formula.append(clause)
            else:
                clause = [self.__n(i)]
                for l in range(1):
                    clause.append(self.__z(next_li0))
                    next_li0 += 1
                wcnf_formula.append(clause)

                # li0 will iterate over the last k z's
                for li0, l in zip(
                        range(next_li0 - 1, next_li0),
                        range(1)
                    ):
                    
                    for j in range(len(normal_features)):
                        if X_norm[i][j] == 1:
                            wcnf_formula.append([
                                -self.__z(li0), -self.__b(l,j)
                            ])
                        else:
                            wcnf_formula.append([
                                -self.__z(li0), -self.__b(l,j+len(normal_features))
                            ])

        return wcnf_formula

    def __n(self, i):
        return self.__literals.id(f'n{i}')
    
    def __b(self, l, j):
        return self.__literals.id(f'b{l}{j}')
    
    def __z(self, li0):
        return self.__literals.id(f'z{li0}')

    def __reset_literals(self):
        self.__literals = IDPool()

    def __create_DNF_rules(self):
        normal_features = self.__dataset_binarized.get_normal_features_label()
        opposite_features = self.__dataset_binarized.get_opposite_features_label()

        b_literals = self.__get_b_literals(normal_features)

        rules_features = [[] for _ in range(1)]
        rules_columns = [[] for _ in range(1)]
        
        for l in range(1):
            for j in range(2 * len(normal_features)):
                if self.__b(l,j) in b_literals:
                    if j < len(normal_features):    # if is normal feature, put opposite feature
                        rules_features[l].append(opposite_features[j])
                        rules_columns[l].append(-(j+1))
                    else:
                        rules_features[l].append(normal_features[j-len(normal_features)])
                        rules_columns[l].append(j-len(normal_features)+1)

        self.__rules_features += rules_features
        self.__rules_columns += rules_columns

    def __get_b_literals(self, features):
        number_features = len(features)
        start = 0
        end = 2 * number_features * 1
        
        return self.__solver_solution[start:end]

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

    def get_rules(self):
        return self.__rules_features_string

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
                if predict == 0 and partial_predict == 1:   # in this case, this sample will never be covered
                    continue
                    
                index_not_covered_samples.append(index_sample)

            new_X_norm.append(X_normal_partition[index_not_covered_samples])
            new_X_oppo.append(X_opposite_partition[index_not_covered_samples])
            new_y.append(y_partition[index_not_covered_samples])
        
        return new_X_norm, new_X_oppo, new_y, has_covered_sample

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
