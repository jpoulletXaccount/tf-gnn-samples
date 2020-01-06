
from utils.number_vehicles.stops.manager import stops_manager_cvrptw
from utils.number_vehicles.stops.objects import stops_cvrptw_ups
from utils.number_vehicles.stops import classDepot

import pandas as pd
import math

class StopsManagerCVRPTW_UPS(stops_manager_cvrptw.StopsManagerCVRPTW):
    """
    Class of manager for stops_cvrptw
    """

    def __init__(self,depot = None):
        super(StopsManagerCVRPTW_UPS,self).__init__(depot)


    @classmethod
    def from_ups_file(cls,filename,date):
        """
        Create a stop manager filled
        :param filename: the file from which we should read the stops
        :param date: the data considered
        :return: an object of this class
        """
        manager = cls()
        manager._create_depot(line=None)
        df = pd.read_csv(filename,converters={'Date':int})

        df = df[df['Date'] == date]

        for i,row in df.iterrows():
            manager._create_stop(row)

        manager._check_time_windows()
        manager._check_feasibility_demand()
        return manager


    def _create_stop(self,row):
        """
        From the line of the file, create a stop with the corresponding
        :param row: a row of the df
        :return:
        """
        guid = row['ID']
        self[guid] = stops_cvrptw_ups.Stop_ups(guid, row['Latitude'], row['Longitude'], row['Npackage'], row['Start Commit Time'],row['End Commit Time'], row['Stop type'],row['Date'])

    def _create_depot(self,line):
        """
        From the line of the file create the corresponding depot
        :return:
        """

        self.depot = classDepot.Depot(42.3775, -71.0796, 1200)  # we set up a shift of 12h


    def _check_time_windows(self):
        """
        Check that all stops are feasible: if not then remove them from the manager
        :return:
        """
        depot_due_date = self.depot.due_date
        list_remove = []
        for stop in self.values():
            dist = stop.get_distance_to_another_stop(self.depot)
            dist = 1.2* 1.609 * dist
            speed = 4.115 + 13.067 * (1 - math.exp(-4.8257 * dist))
            # Convert to km per click
            speed = speed / 100
            time_needed = dist / speed

            # Check if we have time
            if time_needed >= stop.endTW:
                list_remove.append(stop.guid)

            if time_needed + stop.beginTW + stop.service_time >= depot_due_date:
                list_remove.append(stop.guid)

        for stop_guid in list_remove:
            del self[stop_guid]

        print("Number of infeasible stops remove ", len(list_remove))

    def _check_feasibility_demand(self):
        """
        Check that the demand of the stops is lower than the total capacity of the truck,
        so far set up to 350. Remove them if infeasible
        :return:
        """
        max_cap = 350
        list_remove = []
        for stop in self.values():
            if stop.demand >= max_cap:
                list_remove.append(stop.guid)

        for stop_guid in list_remove:
            del self[stop_guid]

        print("Number of infeasible stops remove ", len(list_remove))










