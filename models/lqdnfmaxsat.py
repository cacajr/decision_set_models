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

        for X_normal_partition, X_opposite_partition, y_partition in zip(
                X_normal_partitions,
                X_opposite_partitions,
                y_partitions
            ):

            wcnf_formula = self.__create_wcnf_formula(
                self.__solver_solution,
                X_normal_partition,
                X_opposite_partition,
                y_partition
            )

            # WARNING: this line is used just if the solver is a binary
            wcnf_formula.to_file('./models/wcnf_formula.wcnf')
            
            self.__reset_literals()

            # TODO: update the self.__solver_solution variable with the 
            # MAXSat Solver response passing wcnf_formula

    def __create_wcnf_formula(self, previous_solution, X_norm, X_opp, y):
        features = self.__dataset_binarized.get_normal_features_label()
        wcnf_formula = WCNF()

        # (7.5)
        for i in range(self.__number_rules):    # i ∈ {1, ..., m}
            for j in range(self.__max_size_each_rule):  # j ∈ {1, ..., l}
                clause = []
                for t in features:  # t ∈ Φ U {*}
                    clause.append(self.__x(i,j,t))
                clause.append(self.__x(i,j))

                wcnf_formula.append(clause)
        
        # (7.6)
        for i in range(self.__number_rules):    # i ∈ {1, ..., m}
            for j in range(self.__max_size_each_rule):  # j ∈ {1, ..., l}
                clauses = []
                for i_t, t in enumerate(features):  # t ∈ Φ U {*}
                    for tl in features[i_t+1:]:  # t' ∈ Φ U {*}
                        clauses.append([-self.__x(i,j,t), -self.__x(i,j,tl)])
                    clauses.append([-self.__x(i,j,t), -self.__x(i,j)])
                
                wcnf_formula.extend(clauses)
        
        # (7.7)
        for i in range(self.__number_rules):    # i ∈ {1, ..., m}
            clause = []
            for j in range(self.__max_size_each_rule):  # j ∈ {1, ..., l}
                clause.append(-self.__x(i,j))

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

    def __reset_literals(self):
        self.__literals = IDPool()

    def get_rules(self):
        pass

    def predict(self, instance):
        pass

    def score(self):
        pass
