import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.functions import findIndexUniqueValues


class Binarize:
    ''' Description of Params

        data_frame: must be a dataframe that save the dataset without class/target/y
        column

        series: must be a series that save the class/target/y of the dataset
        
        multiclass: must be a boolean that represents if series param is multiclass.
        If False, the series must be binary (0/1, 'fem'/'mal', true/false, etc).
        If True, the series must be categorical (0/1/2, 'easy'/'normal'/'hard', etc)

        categorical_columns_index: must be a list with columns index that have 
        categorical data in data_frame param

        number_quantiles_ordinal_columns: must be an integer that represents the 
        number of quantiles/columns that the new representation will have

        number_partitions: must be an integer that represents the number of 
        partitions

        balance_instances: must be a boolean that represents whether each 
        partition of the dataset should have balanced classes

        balance_instances_seed: must be a integer that represents the random seed
        number for balancing

    '''
    def __init__(self, 
            data_frame = pd.DataFrame([]),
            series = pd.Series([], dtype='object'),
            multiclass = False,
            categorical_columns_index = [],
            number_quantiles_ordinal_columns = 5,
            number_partitions = 1,
            balance_instances = True,
            balance_instances_seed = None
        ):

        self.__validate_init_params(
            data_frame,
            series,
            multiclass,
            categorical_columns_index,
            number_quantiles_ordinal_columns,
            number_partitions,
            balance_instances,
            balance_instances_seed
        )

        self.__normal_features_labels = pd.array([])
        self.__opposite_features_labels = pd.array([])
        self.__qtts_columns_per_feature_label = pd.array([])
        self.__classes_labels = np.array([])
        self.__qtt_classes = int

        # this variables will be a list with arrays that will save each partition
        self.__binarized_normal_instances = pd.DataFrame([])
        self.__binarized_opposite_instances = pd.DataFrame([])
        self.__binarized_classes = pd.Series([], dtype='object')
        # -----------------------------------------------------------------------

        self.__original_to_binarized_values = list([])

        self.__number_partitions = number_partitions

        self.__binarize(
            data_frame,
            series,
            multiclass,
            categorical_columns_index,
            number_quantiles_ordinal_columns
        )

        self.__separate_partitions(balance_instances, balance_instances_seed, multiclass)

    def __validate_init_params(self,
            data_frame,
            series,
            multiclass,
            categorical_columns_index,
            number_quantiles_ordinal_columns,
            number_partitions,
            balance_instances,
            balance_instances_seed
        ):

        if type(data_frame) != pd.DataFrame:
            raise Exception('Param data_frame must be a pandas.DataFrame')
        if type(series) != pd.Series:
            raise Exception('Param series must be a pandas.Series')
        if type(multiclass) is not bool:
            raise Exception('Param multiclass must be a bool')
        if type(categorical_columns_index) is not list:
            raise Exception('Param categorical_columns_index must be a list')
        if type(number_quantiles_ordinal_columns) is not int:
            raise Exception('Param number_quantiles_ordinal_columns must be an int')
        if type(number_partitions) is not int:
            raise Exception('Param number_partitions must be an int')
        if type(balance_instances) is not bool:
            raise Exception('Param balance_instances must be a bool')
        if balance_instances_seed != None and type(balance_instances_seed) is not int:
            raise Exception('Param balance_instances_seed must be an int or None')
        
        if series.size != data_frame.index.size:
            raise Exception('Param series must be the same size as the data_frame.index')
        if series.unique().size > 2 and not multiclass:
            raise Exception(
                f'Invalid series param. If the multiclass param is False, ' +
                "the series must be binary (0/1, 'fem'/'mal', true/false, etc)"
            )
        if series.unique().size < 2:
            raise Exception(
                f'Invalid series param. The Series has less than two values'
            )
        if number_partitions < 1 or number_partitions > data_frame.index.size:
            raise Exception("Param number_partitions is out of range data_frame's size")

    def __binarize(self, 
            data_frame,
            series, 
            multiclass,
            categorical_columns_index, 
            number_quantiles_ordinal_columns
        ):

        qtts_columns_per_feature_label = []

        for index_feat, feat in enumerate(data_frame):
            unique_values = data_frame[feat].unique()
            unique_values_dtype = str(unique_values.dtype)

            if unique_values.size == 1:
                self.__original_to_binarized_values.append({})

                qtts_columns_per_feature_label.append(0)

                continue

            elif unique_values.size == 2:
                self.__create_binary_columns(feat, data_frame[feat], unique_values)

                qtts_columns_per_feature_label.append(1)

            elif index_feat in categorical_columns_index:
                new_feats = self.__create_categorical_columns(feat, data_frame[feat], unique_values)

                qtts_columns_per_feature_label.append(len(new_feats))

            elif 'float' in unique_values_dtype or 'int' in unique_values_dtype:
                new_feats = self.__create_ordinal_columns(
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

        series_unique_values = np.sort(series.unique())
        self.__qtt_classes = len(series_unique_values)

        if not multiclass:
            self.__binarized_classes = series.replace(
                {
                    series_unique_values[0]: 0, 
                    series_unique_values[1]: 1
                }
            )

            self.__classes_labels = np.array([series_unique_values[1]])

            # adding binarized to original values map in the last column (class) ----
            unique_values_indexes = findIndexUniqueValues(series, series_unique_values)
            values_map = {}
            for index in unique_values_indexes:
                values_map[self.__binarized_classes.values[index]] = series.values[index]
            self.__original_to_binarized_values.append(values_map)
            # -----------------------------------------------------------------------
        else:
            self.__binarized_classes = pd.get_dummies(series).astype(int)

            self.__classes_labels = self.__binarized_classes.columns.values

            # adding binarized to original values map in the last column (class) ----
            for column in self.__binarized_classes.columns:
                unique_values_indexes = findIndexUniqueValues(series, series_unique_values)
                values_map = {}
                for index in unique_values_indexes:
                    values_map[self.__binarized_classes[column].values[index]] = series.values[index]
                self.__original_to_binarized_values.append(values_map)
            # -----------------------------------------------------------------------

    def __create_binary_columns(self, feature, column, unique_values):
        binarized_columns = pd.get_dummies(column).astype(int)
        new_feats = binarized_columns.columns
        binarized_columns = binarized_columns.set_axis(
            [f'{feature} {label}' for label in new_feats],
            axis='columns'
        )

        # adding original to binarized values map --------------------------------
        unique_values_indexes = findIndexUniqueValues(column, unique_values)
        values_map = {}
        for index in unique_values_indexes:
            values_map[column.values[index]] = binarized_columns.iloc[index].values[0]
        self.__original_to_binarized_values.append(values_map)
        # -----------------------------------------------------------------------

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

    def __create_categorical_columns(self, feature, column, unique_values):
        binarized_columns = pd.get_dummies(column).astype(int)
        new_feats = binarized_columns.columns

        # adding original to binarized values map --------------------------------
        unique_values_indexes = findIndexUniqueValues(column, unique_values)
        values_map = {}
        for index in unique_values_indexes:
            values_map[column.values[index]] = binarized_columns.iloc[index].values
        self.__original_to_binarized_values.append(values_map)
        # -----------------------------------------------------------------------

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

    def __create_ordinal_columns(self, feature, column, number_quantiles_ordinal_columns):
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
            columns=new_feats,
            index=column.index
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

        # adding original to binaryzed valus map --------------------------------
        values_map = {}
        for i_quantile, quantile in enumerate(new_feats):
            values_map[i_quantile] = quantile
        self.__original_to_binarized_values.append(values_map)
        # -----------------------------------------------------------------------

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

    def __separate_partitions(self, balance_instances, balance_instances_seed, multiclass):
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

        if not multiclass:
            self.__binarized_classes = np.array_split(
                self.__binarized_classes.values, 
                self.__number_partitions
            )
        else:
            self.__binarized_classes = [
                np.array_split(
                    self.__binarized_classes[b_clss].values, 
                    self.__number_partitions
                )
                for b_clss in self.__binarized_classes
            ]

        if balance_instances:
            self.__balance_instances(
                data_frame_binarized, 
                series_binarized,
                balance_instances_seed,
                multiclass
            )

    def __balance_instances(self, 
            data_frame_binarized, 
            series_binarized, 
            balance_instances_seed,
            multiclass
        ):

        normal_instances_balanced = []
        if not multiclass:
            classes_balanced = []
        else:
            classes_balanced = [[] for _ in range(self.__qtt_classes)]

        X_aux, y_aux = data_frame_binarized.values, series_binarized.values
        
        if self.__number_partitions == 1:
            X1, X2, y1, y2 = train_test_split(
                X_aux, 
                y_aux, 
                train_size=0.5, 
                stratify=y_aux
            )
            normal_instances_balanced.append(np.concatenate((X1, X2)))
            if not multiclass:
                classes_balanced.append(np.concatenate((y1, y2)))
            else:
                for i in range(self.__qtt_classes):
                    classes_balanced[i].append(np.concatenate((y1[:,i], y2[:,i])))
        else:
            for partition in self.__binarized_normal_instances[:-1]:
                X1, X2, y1, y2 = train_test_split(
                    X_aux, 
                    y_aux, 
                    train_size=len(partition),
                    stratify=y_aux,
                    random_state=balance_instances_seed
                )
                normal_instances_balanced.append(X1)
                if not multiclass:
                    classes_balanced.append(y1)
                else:
                    for i in range(self.__qtt_classes):
                        classes_balanced[i].append(y1[:,i])
                X_aux, y_aux = X2, y2

            normal_instances_balanced.append(X_aux)
            if not multiclass:
                classes_balanced.append(y_aux)
            else:
                for i in range(self.__qtt_classes):
                    classes_balanced[i].append(y_aux[:,i])

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
    
    def get_qtts_binarized_feat_per_original_feat(self):
        return self.__qtts_columns_per_feature_label

    def get_normal_instances(self, partition = 0):
        if not self.__partition_validate(partition):
            return self.__binarized_normal_instances
        
        return self.__binarized_normal_instances[partition - 1]

    def get_opposite_instances(self, partition = 0):
        if not self.__partition_validate(partition):
            return self.__binarized_opposite_instances

        return self.__binarized_opposite_instances[partition - 1]
    
    def get_classes(self, clss = None, partition = None):
        if not self.__class_validate(clss):
            raise Exception(
                    f'Invalid clss param. The class does not exist'
                )

        if not self.__partition_validate(partition) and len(self.__classes_labels) == 1:
            return self.__binarized_classes
        
        if not self.__partition_validate(partition) and len(self.__classes_labels) > 1:
            if clss == None:
                return self.__binarized_classes
            return self.__binarized_classes[np.where(self.__classes_labels == clss)[0][0]]
        
        if clss == None:
            raise Exception(
                    f'Invalid clss param. Tell which class the partition is'
                )
        return self.__binarized_classes[np.where(self.__classes_labels == clss)[0][0]][partition - 1]

    def __class_validate(self, clss):
        if clss != None and clss not in self.__classes_labels: 
            return False
        
        return True

    def __partition_validate(self, partition):
        if type(partition) not in [int, float]: 
            return False
        
        if partition < 1 or partition > self.__number_partitions: 
            return False
        
        return True

    def get_original_to_binarized_values(self):
        return self.__original_to_binarized_values

    def get_number_partitions(self):
        return self.__number_partitions
    
    def get_classes_label(self):
        return self.__classes_labels

    def get_qtt_classes(self):
        return self.__qtt_classes
