import pandas as pd
from utils.binarize import Binarize
from pysat.formula import IDPool
from pysat.formula import WCNF


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

        solver_solution = []
        for X_normal_partition, X_opposite_partition, y_partition in zip(
                X_normal_partitions,
                X_opposite_partitions,
                y_partitions
            ):

            wcnf_formula = self.__create_wcnf_formula(
                solver_solution,
                X_normal_partition,
                X_opposite_partition,
                y_partition
            )

            # TODO: update the solver_solution variable with the MAXSat 
            # Solver response passing wcnf_formula

    def __create_wcnf_formula(self, previous_solution, X_norm, X_opp, y):
        if len(previous_solution) == 0:
            pass
        else:
            pass

        # TODO: restrictions created in both cases: len(previous_solution)
        # equal 0 or len(previous_solution) different 0

        self.__reset_literals()

    def __x(self, i, j, t):
        return self.__literals.id(f'x{i}{j}{t}')
    
    def __x(self, i, j):
        return self.__literals.id(f'x{i}{j}*')
    
    def __p(self, i, j):
        return self.__literals.id(f'p{i}{j}')

    def __reset_literals(self):
        self.__literals = IDPool()

    def get_rules(self):
        pass

    def predict(self, instance):
        pass

    def score(self):
        pass