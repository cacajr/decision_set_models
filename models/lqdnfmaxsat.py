import pandas as pd
from utils.binarize import Binarize
from pysat.formula import IDPool
from pysat.formula import WCNF
import numpy as np
from pysat.examples.rc2 import RC2


class LQDNFMaxSAT:
    ''' Description of Params

        number_rules: must be a integer and represents the number of rules/clauses
        that the model will to generate

        max_size_each_rule: must be a integer and represents the maximum number of
        literals per rule/clause

        rules_accuracy_weight: must be a integer and represents how accurate should
        the rule be. The higher, the more accurate the rule should be

        time_out_each_partition: must be an integer and represents the maximum
        time in seconds that Solver has to solve one partition

        categorical_columns_index: must be a list with columns index that have 
        categorical data

        number_quantiles_ordinal_columns: must be an integer that represents the 
        number of quantiles/columns that the new representation will have

        number_partitions: must be an integer that represents the number of 
        partitions

        balance_instances: must be a boolean that represents whether each 
        partition of the dataset should have balanced classes

    '''
    def __init__(self,
            number_rules = 1,
            max_size_each_rule = 5,
            rules_accuracy_weight = 10,
            time_out_each_partition = 1024,
            categorical_columns_index=[],
            number_quantiles_ordinal_columns=5,
            number_partitions = 1,
            balance_instances = True
        ):

        self.__validate_init_params(
            number_rules,
            max_size_each_rule,
            rules_accuracy_weight,
            time_out_each_partition,
            categorical_columns_index,
            number_quantiles_ordinal_columns,
            number_partitions,
            balance_instances
        )

        self.__number_rules = number_rules
        self.__max_size_each_rule = max_size_each_rule
        self.__rules_accuracy_weight = rules_accuracy_weight
        self.__time_out_each_partition = time_out_each_partition
        self.__categorical_columns_index = categorical_columns_index
        self.__number_quantiles_ordinal_columns = number_quantiles_ordinal_columns
        self.__number_partitions = number_partitions
        self.__balance_instances = balance_instances

        self.__dataset_binarized = Binarize

        self.__literals = IDPool()
        self.__solver_solution = list([])
        self.__total_time_solver_solutions = 0.0

        self.__rules_features = list([])
        self.__rules_columns = list([])
        self.__rules_features_string = str('')

    def __validate_init_params(self,
            number_rules,
            max_size_each_rule,
            rules_accuracy_weight,
            time_out_each_partition,
            categorical_columns_index,
            number_quantiles_ordinal_columns,
            number_partitions,
            balance_instances
        ):

        if type(number_rules) is not int:
            raise Exception('Param number_rules must be an int')
        if type(max_size_each_rule) is not int:
            raise Exception('Param max_size_each_rule must be an int')
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
        if type(number_partitions) is not int:
            raise Exception('Param number_partitions must be an int')
        if type(balance_instances) is not bool:
            raise Exception('Param balance_instances must be a bool')

    def fit(self, X, y):
        self.__dataset_binarized = Binarize(
            data_frame=X,
            series=y,
            categorical_columns_index=self.__categorical_columns_index,
            number_quantiles_ordinal_columns=self.__number_quantiles_ordinal_columns,
            number_partitions=self.__number_partitions,
            balance_instances=self.__balance_instances
        )

        X_normal_partitions = self.__dataset_binarized.get_normal_instances()
        X_opposite_partitions = self.__dataset_binarized.get_opposite_instances()
        y_partitions = self.__dataset_binarized.get_classes()

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
            wcnf_formula.to_file('./models/wcnf_formula.wcnf')
            # -------------------------------------------------------------------------

            solver = RC2(wcnf_formula)
            self.__solver_solution = solver.compute()   # TODO: add time out calculate
            self.__total_time_solver_solutions += solver.oracle_time()

            if self.__solver_solution == None:
                raise Exception(f'Partition {index_partition + 1} unsatisfiable')

            if index_partition == self.__number_partitions - 1:
                self.__create_rules(X_normal_partition)
                self.__prune_rules()
                self.__rules_features_string = self.__create_rules_features_string(
                    self.__rules_features
                )

            self.__reset_literals()

    def __create_wcnf_formula(self, previous_solution, X_norm, y):
        features = self.__dataset_binarized.get_normal_features_label()
        wcnf_formula = WCNF()

        # (7.5)
        for i in range(self.__number_rules):    # i ∈ {1, ..., m}
            for j in range(self.__max_size_each_rule):  # j ∈ {1, ..., l}
                clause = []
                for t in range(len(features)):  # t ∈ Φ ...
                    clause.append(self.__x(i,j,t))
                clause.append(self.__x(i,j))    # ... U {*}

                wcnf_formula.append(clause)
        
        # (7.5.1)
        if len(previous_solution) == 0:
            for i in range(self.__number_rules):
                for j in range(self.__max_size_each_rule):
                    for t in range(len(features)):
                        wcnf_formula.append([-self.__x(i,j,t)], weight=1)
                    wcnf_formula.append([self.__x(i,j)], weight=1)
        else:
            x_literals = self.__get_x_literals(features)
            for literal in x_literals:
                wcnf_formula.append([literal], weight=1)

        # (7.6)
        for i in range(self.__number_rules):
            for j in range(self.__max_size_each_rule):
                for t in range(len(features)):
                    for tl in range(t+1, len(features)):
                        wcnf_formula.append([-self.__x(i,j,t), -self.__x(i,j,tl)])
                    wcnf_formula.append([-self.__x(i,j,t), -self.__x(i,j)])
        
        # (7.7)
        for i in range(self.__number_rules):
            clause = []
            for j in range(self.__max_size_each_rule):
                clause.append(-self.__x(i,j))
            
            wcnf_formula.append(clause)

        # (7.8)
        for i in range(self.__number_rules):
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

        # (7.9)
        for i in range(self.__number_rules):
            for j in range(self.__max_size_each_rule):
                for w in range(len(X_norm)):
                    wcnf_formula.append([-self.__x(i,j), self.__y(i,j,w)])

        # (7.10)
        for i in range(self.__number_rules):
            for w in range(len(X_norm)):
                clauses = []
                weights = []
                clause = [self.__z(i,w)]
                for j in range(self.__max_size_each_rule):
                    clauses.append([-self.__z(i,w), self.__y(i,j,w)])
                    weights.append(self.__rules_accuracy_weight)
                    clause.append(-self.__y(i,j,w))
                clauses.append(clause)
                weights.append(self.__rules_accuracy_weight)

                wcnf_formula.extend(clauses, weights=weights)

        # (7.11)
        for u in np.where(y == 1)[0]:    # u ∈ P
            clause = []
            for i in range(self.__number_rules):
                clause.append(self.__z(i,u))
            
            wcnf_formula.append(clause, weight= self.__rules_accuracy_weight)

        # (7.12)
        for v in np.where(y == 0)[0]:    # v ∈ N
            for i in range(self.__number_rules):
                wcnf_formula.append(
                    [-self.__z(i,v)], 
                    weight= self.__rules_accuracy_weight
                )

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

    def __reset_literals(self):
        self.__literals = IDPool()

    def __create_rules(self, X_norm):
        normal_features = self.__dataset_binarized.get_normal_features_label()
        opposite_features = self.__dataset_binarized.get_opposite_features_label()

        x_literals = self.__get_x_literals(normal_features)
        p_literals = self.__get_p_literals(normal_features, X_norm)

        rules_features = [[] for _ in range(self.__number_rules)]
        rules_columns = [[] for _ in range(self.__number_rules)]
        
        for i in range(self.__number_rules):    # i ∈ {1, ..., m}
            for j in range(self.__max_size_each_rule):  # j ∈ {1, ..., l}
                for t in range(len(normal_features)):  # t ∈ Φ U {*}
                    if self.__x(i,j,t) in x_literals:
                        if self.__p(i,j) in p_literals:
                            rules_features[i].append(normal_features[t])
                            rules_columns[i].append(t+1)
                        else:
                            rules_features[i].append(opposite_features[t])
                            rules_columns[i].append(-(t+1))

        self.__rules_features = rules_features
        self.__rules_columns = rules_columns
    
    def __get_x_literals(self, features):
        number_features = len(features)
        start = 0
        end = (number_features + 1) * self.__number_rules * self.__max_size_each_rule
        
        return self.__solver_solution[start:end]

    def __get_p_literals(self, features, X_norm):
        number_features = len(features)
        number_instances = len(X_norm)
        start = (number_features + 1) * self.__number_rules * self.__max_size_each_rule
        end = start + ((number_instances + 1) * self.__number_rules * self.__max_size_each_rule)

        p_literals_region = self.__solver_solution[start:end]

        return p_literals_region[1::number_instances + 1]

    def __prune_rules(self):
        normal_features = self.__dataset_binarized.get_normal_features_label()
        opposite_features = self.__dataset_binarized.get_opposite_features_label()

        # removing repeated literal in the same rule: (... A ∧ A ...)
        for i_rule, rule in enumerate(self.__rules_columns):
            self.__rules_columns[i_rule] = list(set(rule))

        # removing normal and opposite literals in the same rule: (... A ∧ ¬A ...)
        for rule in self.__rules_columns:
            for column in rule:
                if -column in rule:
                    rule.remove(column)
                    rule.remove(-column)

        # removing redundances in the same rule: (A <= 2 ∧ A <= 3) and (A > 2 ∧ A > 3)
        ordinal_normal_index_columns = np.where([
            feat.__contains__('<=') 
            for feat in normal_features
        ])[0]
        ordinal_opposite_index_columns = np.where([
            feat.__contains__('>') 
            for feat in opposite_features
        ])[0]
        for i in range(0, len(ordinal_normal_index_columns), self.__number_quantiles_ordinal_columns-1):
            for rule in self.__rules_columns:
                ordinal_normal_columns = []
                ordinal_opposite_columns = []
                for column in rule:
                    if column > 0:
                        if column - 1 in ordinal_normal_index_columns[i:i+self.__number_quantiles_ordinal_columns-1]:
                            ordinal_normal_columns.append(column)
                            rule.remove(column)
                    else:
                        if abs(column) - 1 in ordinal_opposite_index_columns[i:i+self.__number_quantiles_ordinal_columns-1]:
                            ordinal_opposite_columns.append(column)
                            rule.remove(column)

                if len(ordinal_normal_columns) > 0:
                    rule.append(max(ordinal_normal_columns))
                if len(ordinal_opposite_columns) > 0:
                    rule.append(max(ordinal_opposite_columns))

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
                normal_instance_binarized.append(original_to_binarized[i_num_feat][instance[i_num_feat]])
            elif i_num_feat in self.__categorical_columns_index:
                # TODO: adding key "Other" in the values map (Binarize) and add here
                # the case that the map do not have the value binarized
                for num in original_to_binarized[i_num_feat][instance[i_num_feat]]:
                    normal_instance_binarized.append(num)
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

            if predict == y_test[i_line]:
                hits_count += 1

        return hits_count/y_test.size

    def get_dataset_binarized(self):
        return self.__dataset_binarized

    def get_total_time_solver_solutions(self):
        return self.__total_time_solver_solutions
