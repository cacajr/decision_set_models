import pandas as pd


class Binarize:
    ''' Description of Params

        data_frame: must be a dataframe that save the dataset.

        categorical_columns_index: must be a array with columns index that have 
        categorical data.

        number_quantiles_ordinal_columns: must be an integer that represents the 
        number of quantiles/columns that the new representation will have.

        balance_instances: must be a boolean that represents whether each 
        partition of the dataset should have balanced classes

        number_partitions: must be an integer that represents the number of 
        partitions.

    '''
    def __init__(self, 
            data_frame = pd.DataFrame([]), 
            categorical_columns_index = pd.array([]), 
            number_quantiles_ordinal_columns = 1, 
            balance_instances = True,
            number_partitions = 1
        ):

        self.__normal_header_labels = pd.array([])
        self.__opposite_header_labels = pd.array([])
        self.__qtts_columns_per_header_label = pd.array([])

        self.__binarized_normal_values = pd.Series([])
        self.__binarized_opposite_values = pd.Series([])

        self.__number_partitions = number_partitions

        self.__binarize(data_frame, categorical_columns_index,
                        number_quantiles_ordinal_columns)

        if balance_instances:
            self.__balance_instance()


    def __binarize(self, data_frame, categorical_columns_index, 
                   number_quantiles_ordinal_columns):
        pass

    def __balance_instance(self):
        pass

    def get_normal_header_label(self):
        return self.__normal_header_labels

    def get_opposite_header_label(self):
        return self.__opposite_header_labels
    
    def get_quantities_columns_per_header_label(self):
        return self.__qtts_columns_per_header_label

    def get_normal_values(self, partition = 1):
        pass

    def get_opposite_values(self, partition = 1):
        pass

    def get_number_partitions(self):
        return self.__number_partitions
