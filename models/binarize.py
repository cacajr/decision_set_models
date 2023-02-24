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
            number_quantiles_ordinal_columns = 5, 
            balance_instances = True,
            number_partitions = 1
        ):

        self.__normal_features_labels = pd.array([])
        self.__opposite_features_labels = pd.array([])
        self.__qtts_columns_per_feature_label = pd.array([])

        self.__binarized_normal_instances = pd.DataFrame([])
        self.__binarized_opposite_instances = pd.DataFrame([])

        self.__number_partitions = number_partitions

        self.__binarize(
            data_frame, 
            categorical_columns_index,
            number_quantiles_ordinal_columns
        )

        if balance_instances:
            self.__balance_instances()


    def __binarize(self, 
            data_frame, 
            categorical_columns_index, 
            number_quantiles_ordinal_columns
        ):

        qtts_columns_per_feature_label = []

        for index_feat, feat in enumerate(data_frame):
            unique_values = data_frame[feat].unique()
            unique_values_dtype = str(unique_values.dtype)

            if unique_values.size == 1:
                continue

            elif unique_values.size == 2:
                binarized_column = pd.get_dummies(data_frame[feat])

                self.__binarized_normal_instances = pd.concat(
                    [
                        self.__binarized_normal_instances, 
                        binarized_column[unique_values[0]]
                    ], 
                    axis=1
                )

                self.__binarized_opposite_instances = pd.concat(
                    [
                        self.__binarized_opposite_instances, 
                        binarized_column[unique_values[1]]
                    ], 
                    axis=1
                )

                qtts_columns_per_feature_label.append(1)

            elif index_feat in categorical_columns_index:
                binarized_columns = pd.get_dummies(data_frame[feat])

                self.__binarized_normal_instances = pd.concat(
                    [
                        self.__binarized_normal_instances, 
                        binarized_columns
                    ], 
                    axis=1
                )

                self.__binarized_opposite_instances = pd.concat(
                    [
                        self.__binarized_opposite_instances, 
                        binarized_columns.replace({0: 1, 1: 0}).set_axis(
                            ['¬' + label for label in unique_values], 
                            axis='columns'
                        )
                    ], 
                    axis=1
                )

                qtts_columns_per_feature_label.append(len(unique_values))

            elif 'float' in unique_values_dtype or 'int' in unique_values_dtype:
                categories = pd.qcut(
                    pd.Series(data_frame[feat]), 
                    number_quantiles_ordinal_columns
                )



            else:
                raise Exception('Dataset with some invalid column')
                
        self.__qtts_columns_per_feature_label = pd.array(
            qtts_columns_per_feature_label
        )

    def __balance_instances(self):
        pass

    def get_normal_features_label(self):
        return self.__normal_features_labels

    def get_opposite_features_label(self):
        return self.__opposite_features_labels
    
    def get_quantities_columns_per_feature_label(self):
        return self.__qtts_columns_per_feature_label

    def get_normal_values(self, partition = 1):
        pass

    def get_opposite_values(self, partition = 1):
        pass

    def get_number_partitions(self):
        return self.__number_partitions
