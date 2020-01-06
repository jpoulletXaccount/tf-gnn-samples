
import pandas as pd
import numpy as np

from tasks.sparse_graph_task import DataFold
from utils.number_vehicles.stops.manager import stops_manager_cvrptw

class CreatedDatasetUtils(object):
    """
    Class useful to parse the data corresponding to the created data set
    """

    def __init__(self,path, number_type):
        path = path + "/created_dataset.csv"
        self.df_data = pd.read_csv(path)
        self.df_data = self.df_data.sample(frac=1)

        # to be filled later on
        self.list_dist_matrix = []
        self.list_type_num = []
        self.list_features = []
        self.list_labels = []

        # Parameters
        self.MAX_VEHI = 5
        self._number_type = number_type     # Determine the number of type of link
        self._scale = [5,10,15,25,50,75,100,150,200,500]
        assert len(self._scale) == self._number_type


    @property
    def number_features(self):
        return 4

    @property
    def number_labels(self):
        return self.MAX_VEHI

    def load_data(self):
        """
        Main function, load the data in self.df_data
        :return: a dict[train] = list dist matrix, a dict[train] = list_feature, a dict[train] = list label
        """
        assert len(self.df_data['time_cst'].unique()) == 1, self.df_data['time_cst'].unique()
        assert len(self.df_data['capacity_cst'].unique()) == 1,self.df_data['capacity_cst'].unique()

        list_filename = list(self.df_data['input_file'].unique())

        for filename in list_filename:
            self.parse_file(filename)

        return self.split_data()

    def _find_type(self,time):
        """
        Given a certain time, find the category to which it belongs.
        :param time: the time needed to travel
        :return: a type, or -1 if none
        """
        for i,test in enumerate(self._scale):
            if time <= test:
                return i

        return -1


    def parse_file(self,filename):
        """
        Parse one file. Update corresponding self attribute
        :param filename: the correpsonding filename
        """
        df_file = self.df_data[self.df_data['input_file'] == filename]
        ref_manager = stops_manager_cvrptw.StopsManagerCVRPTW.from_cvrpTW_file(filename)
        for i,row in df_file.iterrows():
            conversion_matrix = {}
            # Add the label
            true_label = min(self.MAX_VEHI, int(row['number_vehilce'])) - 1
            # one_hot_encoded = [0.0 for i in range(0, self.MAX_VEHI)]
            # one_hot_encoded[true_label] = 1.0
            self.list_labels.append(true_label)   # we have to put them between 0 and 4

            list_of_list = row['stops_per_vehicle'].split('--')
            list_of_stops = []
            for l in list_of_list:
                list_of_stops.extend(l.split('_'))
            stop_manager = stops_manager_cvrptw.StopsManagerCVRPTW.from_sublist(list_of_stops, ref_manager)

            # Create features and dist matrix, including the depot at position 0
            features = []
            matrix_type_num = np.zeros(shape=(self._number_type,len(stop_manager) +1))
            dist_matrix = [[] for _ in range(0,self._number_type)]

            # depot
            features.append(ref_manager.depot.features)
            for stop in stop_manager.values():
                conversion_matrix[stop.guid] = len(conversion_matrix) +1
                test_dist = stop.get_distance_to_another_stop(ref_manager.depot)
                edge_type = self._find_type(test_dist)
                if edge_type != -1:
                    dist_matrix[edge_type].append((0, conversion_matrix[stop.guid]))
                    matrix_type_num[edge_type, conversion_matrix[stop.guid]] +=1

            for stopId in stop_manager:
                stop = stop_manager[stopId]
                features.append(stop.features)
                dist_stop = stop.get_distance_to_another_stop(ref_manager.depot)
                edge_type = self._find_type(dist_stop)
                if edge_type != -1:
                    dist_matrix[edge_type].append((conversion_matrix[stopId],0))
                    matrix_type_num[edge_type, 0] +=1

                for stopId_2 in stop_manager:
                    dist_stop = stop.get_distance_to_another_stop(ref_manager[stopId_2])
                    edge_type = self._find_type(dist_stop)
                    if edge_type != -1:
                        dist_matrix[edge_type].append((conversion_matrix[stopId],conversion_matrix[stopId_2]))
                        matrix_type_num[edge_type, conversion_matrix[stopId_2]] +=1

            type_to_adj_list = [np.array(sorted(adj_list), dtype=np.int32) if len(adj_list) > 0 else np.zeros(shape=(0,2), dtype=np.int32)
                            for adj_list in dist_matrix]

            assert np.array(type_to_adj_list).shape[0] == self._number_type, print(np.array(type_to_adj_list).shape[0])

            self.list_features.append(features)
            self.list_dist_matrix.append(type_to_adj_list)
            self.list_type_num.append(matrix_type_num)


    def split_data(self):
        """
        Split the data into three parts, train, valid, and test
        :return: a dict[train] = list dist matrix, a dict[train] = list_feature, a dict[train] = list label
        """
        total_data = len(self.list_labels)
        first_bound = int(0.7 * total_data)
        second_bound = int(0.85 * total_data)

        dict_matrix = {DataFold.TRAIN : self.list_dist_matrix[0:first_bound],
                       DataFold.VALIDATION : self.list_dist_matrix[first_bound:second_bound],
                       DataFold.TEST : self.list_dist_matrix[second_bound:]}

        dict_type_enum = {DataFold.TRAIN : self.list_type_num[0:first_bound],
                       DataFold.VALIDATION : self.list_type_num[first_bound:second_bound],
                       DataFold.TEST : self.list_type_num[second_bound:]}


        dict_features = {DataFold.TRAIN : self.list_features[0:first_bound],
                       DataFold.VALIDATION : self.list_features[first_bound:second_bound],
                       DataFold.TEST : self.list_features[second_bound:]}

        dict_labels = {DataFold.TRAIN : self.list_labels[0:first_bound],
                       DataFold.VALIDATION : self.list_labels[first_bound:second_bound],
                       DataFold.TEST : self.list_labels[second_bound:]}

        return dict_matrix, dict_type_enum, dict_features, dict_labels



