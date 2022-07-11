import pandas as pd


class Data:
    def __init__(self, data_frame = None, num_partition = 1, balanced_instance = False):
        self.__normal_header, self.__opposite_header = pd.array([])
        self.__normal_data, self.__opposite_data = pd.Series([])

        self.__binarized_normal_data, self.__binarized_opposite_data = pd.Series([])

        self.__load_header_and_data(data_frame)

        self.__binarize_data()

    
    def __load_header_and_data(self, data_frame):
        pass


    def __binarize_data(self):
        pass
