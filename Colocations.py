##################################################################################
# Class: Colocations.py
# Description: The class generates a co-location.csv file from a mobility   
#              traces dataset
# 
# Created on 01/dec/2020
# @author: Dimitri Belli             
# License: GPLv3
# Web: https://github.com/Lafcadio79/Locations_Colocations
##################################################################################
# This program is free software; you can redistribuite it and/or modify it under
# the terms of the GNU/General Pubblic License as published the Free software
# Foundation; either version 3 of the License, or (at your opinion) any later 
# version
##################################################################################


# import libraries
import numpy     as np
import pandas    as pd
import datetime  as dt
import geopandas as gp
import warnings  as wn

from tqdm        import tqdm

wn.filterwarnings("ignore")


# define the Colocation class with assignment
class Colocation:
    """
    A class used to convert a dataset of location information into a dataset of co-locations
    
    ........................................................................................
    
    Variables
    ----------
    
    tw    : time_window (in seconds, default 150s, i.e., 2.5 minutes)
    
    limit : accuracy threshold (in meters, default 50m)
    
    dst   : the short range communication distance of each device (in meters, default 500m)


    Methods
    -------
    
    print_dataframe
    
    from_date_string_to_datetime
    
    from_datetime_to_timestamp
    
    accuracy_threshold
    
    add_short_range_communication_area
    
    dataframe_split
    
    get_intersections
    
    """
    
    def __init__(self, data):
        """
        Parameters
        ----------
        data : str
           a csv filename - The data within the csv file must have at least the following header information:
  
                           ["user_id", "latitude", "longitude", "sampletimestamp", "accuracy"]
        """
        
        self.data = pd.read_csv(data, header=0).filter(["user_id", "latitude", "longitude", "sampletimestamp", "accuracy"])

    def users_mapping(self):
        """
        Anonymize the n user identifiers by re-labelling the user_id from 0 to n 
        """
        
        users = self.data.user_id.unique()
        
        map_uid = dict(zip(users, np.arange(0, len(users))))
        
        self.data = self.data.replace({"user_id" : map_uid})
        
    def from_date_string_to_datetime(self):
        """
        Converts all date string to datetime format
        """
        
        self.data.sampletimestamp = pd.to_datetime(self.data.sampletimestamp)
        
    def from_datetime_to_timestamp(self):
        """
        Converts all datetime to timestamp
        """
        
        self.data.sampletimestamp = self.data.sampletimestamp.values.astype(np.int64) // 10 ** 9
    
    def accuracy_threshold(self, limit=50):
        """
        Deletes all the dataframe observations that exceed a given accuracy
        
        Parameter
        ----------
        limit : int
            the maximum accuracy limit (default 50m) 
            
        """
        
        self.data = self.data.loc[self.data.accuracy <= limit]
        
    def add_short_range_communication_area(self, dst=100):
        """
        Adds the short range communication area information as a
        polygon (circle) to the dataframe  
        
        Parameter
        ---------
        dst : int
            the distance from the center in meters (default 100m)            
        """

        self.data["short_range_communication_area"] = gp.points_from_xy(self.data.latitude, self.data.longitude).buffer(dst / 100000)
    
    def dataframe_split(self, tw=150):
        """
        Splits the whole mobility traces into a series of dataframes on
        the basis of a given time window (in seconds)
        
        Parameter
        ---------
        tw : int
           the time window whihc provides the interval for slicing the 
           mobility traces dataframe (default 150s, i.e. 2.5 minutes)
        """
        
        # list of dataframes
        df_split_list = []
        
        # get the time limits
        min_stp = self.data.sampletimestamp.min()
        max_stp = self.data.sampletimestamp.max()
        
        # get the number of dataframes + 1 ()
        df_s = int((max_stp - min_stp) / tw) + 1        
        
        for i in tqdm(range(df_s), desc="Splitting the dataframe:\t"):
            
            # split the mobility traces into a partition (dataframe)
            df_temp = self.data[(self.data.sampletimestamp >= min_stp) & (\
                                    self.data.sampletimestamp <= min_stp + tw)]
            # append the partition (dataframe) to the list
            df_split_list.append(df_temp)
            # increase the minimum time limit for the following interval
            min_stp = min_stp + tw
        
        return df_split_list
    
    def get_intersections(self):
        """
        Splits the mobility traces into several datasets and for each one seeks for intersections
        between nodes (contact spatial information)

        """

        df_dfs_int = []

        # split the mobility traces dataset in partitions
        df_split = self.dataframe_split()
        
        # seek for each partition the nodes' intersections (spatial information)
        for single_df_split in tqdm(df_split, desc="Seeking for intersections:\t"):
            df_dfs = {}
            intersections = []
            users = single_df_split.user_id.unique()
            for user in users:
                df_dfs[user] = single_df_split.loc[single_df_split.user_id == user].reset_index().filter(["user_id", \
                            "latitude", "longitude", "sampletimestamp", "short_range_communication_area"])

            for k in range(len(users)-1):
                subject = users[k]
                other_users = users[k+1:]
                for i in df_dfs[subject].iterrows():
                    for usr in other_users:
                        for j in df_dfs[usr].iterrows():
                            if(i[1].short_range_communication_area.intersects(j[1].short_range_communication_area)):
                                intersections.append([i[1].user_id, i[1].sampletimestamp, j[1].user_id, j[1].sampletimestamp])

            # insert all the intersection information of a partition into a dataframe                    
            df_int = pd.DataFrame(intersections, columns=["uid_1", "timestamp_uid_1", "uid_2", "timestamp_uid_2"])
    
            df_dfs_int.append(df_int)
        
        return df_dfs_int
    
    def get_colocations(self):
        """
        Generates the co-location information (performing the above splitting procedure)
        
        """
        # empty dataframe to store all the co-locations
        df_ac = pd.DataFrame()
        
        # get all the intersections for all the mobility traces (splitted)
        df_dfs_int = self.get_intersections()

        # getting the colocations for all the dataset's partition
        for df in tqdm(df_dfs_int, desc="Generating the co-locations\t"):

            # get all the uniques users' couples
            users_couples = [list(c) for c in set(tuple(c) \
                                for c in [[user[1].uid_1, user[1].uid_2] for user in df.iterrows()])]
            # get co-locations of a single partition
            coloc = []
            for couple in users_couples:
                # filter to the rows of interest
                sc = df[(df.uid_1 == couple[0]) & (df.uid_2 == couple[1])]
                # collect the temporal information and generate the co-locations
                df_time = pd.concat([sc.timestamp_uid_1, sc.timestamp_uid_2])
                coloc.append([couple[0], couple[1], df_time.min(), "up"])
                coloc.append([couple[0], couple[1], df_time.max(), "down"])
                coloc.append([couple[1], couple[0], df_time.min(), "up"])
                coloc.append([couple[1], couple[0], df_time.max(), "down"])
    
            # order the co-locations into a dataframe
            df_coloc = pd.DataFrame(coloc, columns=["node_1", "node_2", "timestamp", "connection"])
    
            # concatenate the dataframe
            df_ac = pd.concat([df_ac, df_coloc])
            
        # reset the index of the co-locations
        df_ac = df_ac.reset_index().filter(["node_1", "node_2", "timestamp", "connection"])
        
        # save and return the co-locations
        df_ac.to_csv("output//co_locations.csv", index=False)
        
        print("Done!")
        return df_ac
    

# execute
if __name__ == '__main__':
    # the mobility traces' dataset
    obj = Colocation("input//test_dataset.csv")
    # anonymization of the data
    obj.users_mapping()
    # filter for accuracy (none = 50m)
    obj.accuracy_threshold()
    # it takes a minute for the computation (none = 100m)
    obj.add_short_range_communication_area(500)
    # convert to datetime
    obj.from_date_string_to_datetime()
    # convert to timestamp
    obj.from_datetime_to_timestamp()
    # get the co-locations and save them into a "co_locations.csv" file
    co_locations = obj.get_colocations()
