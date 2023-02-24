import pandas as pd
import numpy as np


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
                self.__make_binary_columns(feat, data_frame[feat])

                qtts_columns_per_feature_label.append(1)

            elif index_feat in categorical_columns_index:
                new_feats = self.__make_categorical_columns(feat, data_frame[feat])

                qtts_columns_per_feature_label.append(len(new_feats))

            elif 'float' in unique_values_dtype or 'int' in unique_values_dtype:
                new_feats = self.__make_ordinal_columns(
                    feat,
                    data_frame[feat], 
                    number_quantiles_ordinal_columns
                )

                qtts_columns_per_feature_label.append(len(new_feats))

            else:
                raise Exception(
                    f'Dataset with column {feat} invalid. ' +
                    "Make sure this column really isn't a categorical column"
                )


        self.__normal_features_labels = pd.array(
            self.__binarized_normal_instances.columns
        )

        self.__opposite_features_labels = pd.array(
            self.__binarized_opposite_instances.columns
        )

        self.__qtts_columns_per_feature_label = pd.array(
            qtts_columns_per_feature_label
        )

    def __make_binary_columns(self, feature, column):
        binarized_columns = pd.get_dummies(column)
        new_feats = binarized_columns.columns
        binarized_columns = binarized_columns.set_axis(
            [f'{feature} {label}' for label in new_feats],
            axis='columns'
        )

        self.__binarized_normal_instances = pd.concat(
            [
                self.__binarized_normal_instances, 
                binarized_columns.iloc[:, 0]
            ], 
            axis=1
        )

        self.__binarized_opposite_instances = pd.concat(
            [
                self.__binarized_opposite_instances, 
                binarized_columns.iloc[:, 1]
            ], 
            axis=1
        )

        return new_feats

    def __make_categorical_columns(self, feature, column):
        binarized_columns = pd.get_dummies(column)
        new_feats = binarized_columns.columns

        self.__binarized_normal_instances = pd.concat(
            [
                self.__binarized_normal_instances, 
                binarized_columns.set_axis(
                    [f'{feature} {label}' for label in new_feats], 
                    axis='columns'
                )
            ], 
            axis=1
        )

        self.__binarized_opposite_instances = pd.concat(
            [
                self.__binarized_opposite_instances, 
                binarized_columns.replace({0: 1, 1: 0}).set_axis(
                    [f'Not {feature} {label}' for label in new_feats], 
                    axis='columns'
                )
            ], 
            axis=1
        )

        return new_feats

    def __make_ordinal_columns(self, feature, column, number_quantiles_ordinal_columns):
        new_feats = column.quantile(
            [
                n/number_quantiles_ordinal_columns 
                for n in range(1, number_quantiles_ordinal_columns)
            ]
        )

        binarized_columns = pd.DataFrame(
            [
                [0 for _ in range(len(new_feats))] 
                for _ in range(len(column))
            ], 
            columns=new_feats
        )
        for col in binarized_columns:
            indexs_one = column.index[
                np.where(column <= col)
            ]
            indexs_zero = column.index[
                np.where(column > col)
            ]
            binarized_columns.loc[indexs_one, col] = 1
            binarized_columns.loc[indexs_zero, col] = 0

        self.__binarized_normal_instances = pd.concat(
            [
                self.__binarized_normal_instances, 
                binarized_columns.set_axis(
                    [f'{feature} <= {label}' for label in new_feats], 
                    axis='columns'
                )
            ], 
            axis=1
        )

        self.__binarized_opposite_instances = pd.concat(
            [
                self.__binarized_opposite_instances, 
                binarized_columns.replace({0: 1, 1: 0}).set_axis(
                    [f'{feature} > {label}' for label in new_feats], 
                    axis='columns'
                )
            ], 
            axis=1
        )

        return new_feats

    def __balance_instances(self):
        pass

    def get_normal_features_label(self):
        return self.__normal_features_labels

    def get_opposite_features_label(self):
        return self.__opposite_features_labels
    
    def get_quantities_columns_per_feature_label(self):
        return self.__qtts_columns_per_feature_label

    def get_normal_instances(self, partition = 1):
        return self.__binarized_normal_instances.values

    def get_opposite_instances(self, partition = 1):
        return self.__binarized_opposite_instances.values

    def get_number_partitions(self):
        return self.__number_partitions
