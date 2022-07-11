import pandas as pd


class Data:
    def __init__(self, data_frame = None, categorical_columns = None, 
                 number_partitions = 1, number_quantiles_ordinal_columns = 1,
                 has_balanced_instances = False):

        self.__normal_header = pd.array([])
        self.__opposite_header = pd.array([])

        self.__binarized_normal_values = pd.Series([])
        self.__binarized_opposite_values = pd.Series([])

        self.__number_partitions = number_partitions

        self.__binarize(data_frame, categorical_columns,
                        number_quantiles_ordinal_columns)

        if has_balanced_instances:
            self.__balance_instance()


    def __binarize(self, data_frame, categorical_columns, 
                   number_quantiles_ordinal_columns):
        pass

    def __balance_instance(self):
        pass

    def get_normal_header(self):
        return self.__normal_header

    def get_opposite_header(self):
        return self.__opposite_header

    def get_normal_values(self, partition = 1):
        pass

    def get_opposite_values(self, partition = 1):
        pass
