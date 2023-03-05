import pandas as pd
from utils.binarize import Binarize
from pysat.formula import IDPool
from pysat.formula import WCNF
import numpy as np


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

        for X_normal_partition, y_partition in zip(
                X_normal_partitions,
                y_partitions
            ):

            wcnf_formula = self.__create_wcnf_formula(
                self.__solver_solution,
                X_normal_partition,
                y_partition
            )

            # WARNING: this line is used just if the solver is a binary
            wcnf_formula.to_file('./models/wcnf_formula.wcnf')

            self.__reset_literals()

            # TODO: update the self.__solver_solution variable with the 
            # MAXSat Solver response passing wcnf_formula

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

    def get_rules(self):
        pass

    def predict(self, instance):
        pass

    def score(self):
        pass
