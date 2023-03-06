import pandas as pd
from utils.binarize import Binarize
from pysat.formula import IDPool
from pysat.formula import WCNF
import numpy as np
from pysat.examples.rc2 import RC2


class LQDNFMaxSAT:
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

        # TODO: init params validation

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

        self.__rules_features = list([])
        self.__rules_columns = list([])
        self.__rules_features_string = str('')

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
            # WARNING: this line is used just if the solver is a binary or to
            # test
            wcnf_formula.to_file('./models/wcnf_formula.wcnf')

            self.__solver_solution = RC2(wcnf_formula).compute()

            if self.__solver_solution == None:
                raise Exception(f'Partition {index_partition + 1} unsatisfiable')

            # TODO: add time and time out calculate

            if index_partition == self.__number_partitions - 1:
                self.__create_rules(X_normal_partition)

            self.__reset_literals()

    def __create_wcnf_formula(self, previous_solution, X_norm, y):
        features = self.__dataset_binarized.get_normal_features_label()
        wcnf_formula = WCNF()

        # (7.5)
        for i in range(self.__number_rules):    # i ∈ {1, ..., m}
            for j in range(self.__max_size_each_rule):  # j ∈ {1, ..., l}
                clause = []
                for t in range(len(features)):  # t ∈ Φ U {*}
                    clause.append(self.__x(i,j,t))
                clause.append(self.__x(i,j))

                wcnf_formula.append(clause)
        
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
                clause = [self.__z(i,w)]
                for j in range(self.__max_size_each_rule):
                    clauses.append([-self.__z(i,w), self.__y(i,j,w)])
                    clause.append(-self.__y(i,j,w))
                clauses.append(clause)

                wcnf_formula.extend(clauses)

        # (7.11)
        for u in np.where(y == 1)[0]:    # u ∈ P
            clause = []
            for i in range(self.__number_rules):
                clause.append(self.__z(i,u))
            
            wcnf_formula.append(clause)

        # (7.12)
        for v in np.where(y == 0)[0]:    # v ∈ N
            for i in range(self.__number_rules):
                wcnf_formula.append([-self.__z(i,v)])

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
        self.__rules_features_string = self.__create_rules_features_string(rules_features)

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
        qtt_binarized_feat = self.__dataset_binarized.get_qtt_binarized_feat_per_original_feat()

        if type(instance) not in [list, pd.array, np.array, np.ndarray]:
            raise Exception('Param instance must be a list, pd.array, np.array or np.ndarray')
        if len(instance) != len(qtt_binarized_feat):
            raise Exception('Param instance with number of features invalid')
        
        return True

    def __binarize_instance(self, instance):
        normal_instance_binarized = []

        original_to_binarized = self.__dataset_binarized.get_original_to_binarized_values()[:-1]
        qtt_binarized_feat = self.__dataset_binarized.get_qtt_binarized_feat_per_original_feat()

        for i_num_feat, num_feat in enumerate(qtt_binarized_feat):
            if num_feat == 0:
                continue
            elif num_feat == 1:
                normal_instance_binarized.append(original_to_binarized[i_num_feat][instance[i_num_feat]])
            elif i_num_feat in self.__categorical_columns_index:
                for num in original_to_binarized[i_num_feat][instance[i_num_feat]]:
                    normal_instance_binarized.append(num)
            elif type(instance[i_num_feat]) in [
                    int, np.int16, np.int32, np.int64, float, 
                    np.float16, np.float32, np.float64
                ]:

                for quantis in original_to_binarized[i_num_feat].keys():
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
