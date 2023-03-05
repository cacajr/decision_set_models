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
        self.__solver_solution = []

        self.__rules_features = []
        self.__rules_indexes = []
        self.__rules_features_string = ''

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
        rules_indexes = [[] for _ in range(self.__number_rules)]
        
        for i in range(self.__number_rules):    # i ∈ {1, ..., m}
            for j in range(self.__max_size_each_rule):  # j ∈ {1, ..., l}
                for t in range(len(normal_features)):  # t ∈ Φ U {*}
                    if self.__x(i,j,t) in x_literals:
                        if self.__p(i,j) in p_literals:
                            rules_features[i].append(normal_features[t])
                            rules_indexes[i].append(t)
                        else:
                            rules_features[i].append(opposite_features[t])
                            rules_indexes[i].append(-t)

        self.__rules_indexes = rules_indexes
        self.__rules_features = rules_features
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
        pass

    def score(self):
        pass
