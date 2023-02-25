import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Binarize:
    ''' Description of Params

        data_frame: must be a dataframe that save the dataset without class/target/y
        column.

        series: must be a binary series (yes/no, 0/1, mas/fem, etc) that save the 
        class/target/y of the dataset

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
            series = pd.Series([], dtype='object'),
            categorical_columns_index = pd.array([]),
            number_quantiles_ordinal_columns = 5,
            balance_instances = True,
            number_partitions = 1
        ):

        self.__normal_features_labels = pd.array([])
        self.__opposite_features_labels = pd.array([])
        self.__qtts_columns_per_feature_label = pd.array([])

        # this variables will be a list with arrays that will save each partition
        self.__binarized_normal_instances = pd.DataFrame([])
        self.__binarized_opposite_instances = pd.DataFrame([])
        self.__binarized_classes = pd.Series([], dtype='object')
        # -----------------------------------------------------------------------

        self.__number_partitions = number_partitions

        self.__binarize(
            data_frame,
            series,
            categorical_columns_index,
            number_quantiles_ordinal_columns
        )

        self.__separate_partitions(balance_instances)


    def __binarize(self, 
            data_frame,
            series, 
            categorical_columns_index, 
            number_quantiles_ordinal_columns
        ):

        qtts_columns_per_feature_label = []

        for index_feat, feat in enumerate(data_frame):
            unique_values = data_frame[feat].unique()
            unique_values_dtype = str(unique_values.dtype)

            if unique_values.size == 1:
                qtts_columns_per_feature_label.append(0)

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

        unique_values = np.sort(series.unique())
        if unique_values.size != 2:
            raise Exception('The param series (column class/target/y) must be binary')

        self.__binarized_classes = series.replace(
            {
                unique_values[0]: 0, 
                unique_values[1]: 1
            }
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

    def __separate_partitions(self, balance_instances):
        # if the number of partitions is greater than the number of instances, empty
        # partitions will be created. TODO: validation to compare if number of partitions
        # is greater than number of instances
        if self.__number_partitions < 1 or type(self.__number_partitions) is not int:
            raise Exception('Number of partitions is invalid')

        data_frame_binarized = self.__binarized_normal_instances
        series_binarized = self.__binarized_classes

        self.__binarized_normal_instances = np.array_split(
            self.__binarized_normal_instances.values, 
            self.__number_partitions
        )

        self.__binarized_opposite_instances = np.array_split(
            self.__binarized_opposite_instances.values, 
            self.__number_partitions
        )

        self.__binarized_classes = np.array_split(
            self.__binarized_classes.values, 
            self.__number_partitions
        )

        if balance_instances:
            self.__balance_instances(data_frame_binarized, series_binarized)

    def __balance_instances(self, data_frame_binarized, series_binarized):
        normal_instances_balanced = []
        classes_balanced = []

        partitions = self.__binarized_normal_instances
        if self.__number_partitions > 1:
            partitions = self.__binarized_normal_instances[:-1]

        X_aux, y_aux = data_frame_binarized.values, series_binarized.values
        for partition in partitions:
            X1, X2, y1, y2 = train_test_split(X_aux, y_aux, train_size=len(partition))
            normal_instances_balanced.append(X1)
            classes_balanced.append(y1)
            X_aux, y_aux = X2, y2
        normal_instances_balanced.append(X_aux)
        classes_balanced.append(y_aux)

        opposite_instances_balanced = [
            np.select([instance == 0, instance == 1], [1, 0], instance)
            for instance in normal_instances_balanced
        ]

        self.__binarized_normal_instances = normal_instances_balanced
        self.__binarized_opposite_instances = opposite_instances_balanced
        self.__binarized_classes = classes_balanced

    def get_normal_features_label(self):
        return self.__normal_features_labels

    def get_opposite_features_label(self):
        return self.__opposite_features_labels
    
    def get_qtt_binarized_feat_per_original_feat(self):
        return self.__qtts_columns_per_feature_label

    def get_normal_instances(self, partition = 1):
        self.__partition_validate(partition)
        
        return self.__binarized_normal_instances[partition - 1]

    def get_classes(self, partition = 1):
        self.__partition_validate(partition)
        
        return self.__binarized_classes[partition - 1]

    def get_opposite_instances(self, partition = 1):
        self.__partition_validate(partition)

        return self.__binarized_opposite_instances[partition - 1]

    def __partition_validate(self, partition):
        if partition < 1 or partition > self.__number_partitions: 
            raise Exception(f'Partition {partition} is out of range')
        
        if type(partition) not in [int, float]: 
            raise Exception(f'Partition {partition} is invalid')
        
        pass

    def get_number_partitions(self):
        return self.__number_partitions
