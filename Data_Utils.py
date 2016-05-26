from netCDF4 import Dataset
import numpy as np
import matplotlib.dates as md
import matplotlib.pyplot as plt
from matplotlib import dates
from datetime import datetime
import math
import logging.config
from logging_configuration import *
import time
import pickle

__author__ = 'akrietemeyer'

"""
ignore invalid values in numpy (NaN) operations
set Logger
"""
np.seterr(invalid='ignore')
logging.config.dictConfig(LOGGING_CONFIGURATION_NETCDF_LAYER)
logger = logging.getLogger("root")
logging.getLogger("root").setLevel(logging.WARNING)

# functions basically static, so used globally


def get_converted_time(stamp):
    """
    converts a float unix timestamp to datetime str
        format: YYYY-MM-DD HH:MM:SS
    :param stamp: float (optional: numpy.float64)
    :return: str
    example: get_converted_time(1441065600.0)
        --> '2015-09-01 00:00:00'
    """
    return str(datetime.utcfromtimestamp(stamp))


def get_data_array(data_array):
    """
    returns pure data in NetCDF variable (without mask)
    :param data_array: NetCDF Variable
    :return: data array (just [xxx])
    """
    if type(data_array.__array__()) is np.ma.masked_array:
        return data_array.__array__().data
    else:
        return data_array.__array__()


def get_mean(data):
    return np.nanmean(data)


class NCDFDataRoot:
    """
    #1 TODO: implement general gradient test in NCDFDataRoot. (actually in Moorings)
    """

    def __init__(self, url, start_time, end_time):
        """
        url: link / path to netcdf data file (openDAP link)
        reads / creates / calls:
            whole data a netCDF dataset from the url        as  self.root
            time data numpy.ndarray from time variable      as  self.time
            empty dict() to store own qc + methods applied  as  self.qc_dict
            array xticks for plotting in 3 steps [0, len/2, end]    as  self.xTicks
            array xtickslabel iterates over xticks and converts time (self.get_converted_time) to readable date
                as self.xTicksLabel
            empty array to store label names    as  self.legend_names
            last update interval (self.get_update_interval) from the NetCDF file    as  self.update_interval
            calls (self.check_update)
            empty array to store the base axis for plotting     as  self.base_axis
        :param url: str()
        :return: void
        example usage: NCDFDataRoot("http://thredds.socib.es/thredds/dodsC/mooring/weather_station/  ..
            station_parcbit-scb_met004/L1/dep0002_station-parcbit_scb-met004_L1_latest.nc")
        """
        self.url = url
        self.root = Dataset(url)
        self.return_flag = False

        self.start_time = start_time
        self.end_time = end_time
        self.corrected_starting_idx = 1

        self.time = get_data_array(self.get_variable('time'))
        idx_this_timestamp = np.where(self.time <= self.end_time)[0][-1]
        self.end_idx = idx_this_timestamp + 1
        idx_last_timestamp = np.where(self.time >= self.start_time)[0]
        if idx_last_timestamp.any():
            self.corrected_starting_idx = np.where(self.time >= (self.start_time - 180000))[0][0]
            self.time = self.time[self.corrected_starting_idx:self.end_idx]
            idx_last_timestamp = np.where(self.time >= self.start_time)[0][0]
            self.start_idx = idx_last_timestamp
        else:
            self.return_flag = True
            self.start_idx = -2

        self.qc_dict = dict()

        self.x_ticks = [self.time[0], self.time[round(len(self.time) / 2)], self.time[len(self.time) - 1]]
        self.x_ticks_label = map(get_converted_time, self.x_ticks)

        logger.info('First entry from ' + get_converted_time(self.time[0]))
        logger.info('Last entry from ' + get_converted_time(self.time[len(self.time) - 1]))

        self.legend_names = []

        self.update_interval = self.get_update_interval()
        self.last_sample_interval = self.get_last_sample_interval()

        logger.info('Update interval (from NetCDF): ' + str(self.update_interval))
        logger.info('Last sample interval: ' + str(self.last_sample_interval))

        self.temp_boolean_list = []
        self.temp_var_data = np.array

        self.check_update()
        self.base_axis = []

        self.date_converted = [datetime.fromtimestamp(ts) for ts in self.time]
        self.date_converted_backward = md.date2num(self.date_converted)

        self.differences_log = dict()
        self.set_up_differences_log()

    def set_up_differences_log(self):
        """
        Should / Could be wrapped into a class...
        :return:
        """
        self.differences_log['netcdf'] = dict()
        self.differences_log['netcdf']['bads'] = 0
        self.differences_log['netcdf']['nans'] = 0
        self.differences_log['own'] = dict()
        self.differences_log['own']['bads'] = dict()
        self.differences_log['own']['bads']['validRange'] = 0
        self.differences_log['own']['bads']['stationary'] = 0
        self.differences_log['own']['bads']['spike'] = 0
        self.differences_log['own']['bads']['gradient'] = 0
        self.differences_log['own']['bads']['nans'] = 0
        self.differences_log['own']['bads']['sum_excluding_nans'] = 0
        self.differences_log['diffs'] = dict()
        self.differences_log['diffs']['bads'] = 0
        self.differences_log['diffs']['nans'] = 0

    def append_qc_list(self, qc_name):
        """
        Appends a QC Dict entry (self.qc_dict) to manage the own different QC lists (e.g. AIR_PRE, etc)
        which have to go through different QC methods.
        As length dimension, the len(self.time) is used.
        Inside the dict entry (qc_dict[qc_name]), a 2D Array is created (first dimension filled with ones, second
            with zeros). The first dimension (qc_dict[qc_name][0]) is meant to be used as QC flags as specified in
            socib data QC. The second dimension (qc_dict[qc_name][1]) is meant to be used to store, which QC method
            has been applied.
        :param qc_name: str()
        :return: void
        example: append_qc_list("AIR_TEM")
            --> [[1, 1, 1, ...],[0, 0, 0, ...],[True, True, ...]dtype=bool]
        """
        # change AK 15.10.2015
        # self.qc_dict[qc_name] = np.ones(len(self.time)), np.zeros(len(self.time))
        self.qc_dict[qc_name] = np.ones(len(self.time)), np.zeros(len(self.time)), np.ones((len(self.time)), dtype=bool)

    def get_variables(self):
        """
        Returns a (collections.OrderedDict) from self.root.variables.
        Has the same structure as in the NetCDF document.
        Access single variables by using get_variable(str)
        :return: collections.OrderedDict (basically a dict)
        example: all_variables = self.get_variables()
            --> OrderedDict([(u'station_name', <type 'netCDF4._netCDF4.Variable'>
                    |S1 station_name(name_strlen)
                    ...
                float64 time(time)
                    standard_name: time
                    ...
        """
        return self.root.variables

    def get_variable(self, var_name):
        """
        Returns a NetCDF variable from self.root.variables[var_name].
        Used to access a variable.
        :param var_name: str
        :return: netCDF4._netCDF4.Variable
        example: self.time = get_variable('time')
            --> float64 time(time)
                    standard_name: time
                    units: seconds since 1970-01-01 00:00:00
                    ...
        """
        try:
            return self.root.variables.get(var_name)
        except (RuntimeError, TypeError, NameError) as detail:
            logger.error(detail)

    def print_variable(self, var_name):
        """
        prints the content of the variable form netcdf to debug handler.
        :param var_name: str
        :return: void
        example: self.print_variable('AIR_TEM')
        """
        logger.debug(var_name + ':\r\n' + str(self.get_variable(var_name)))

    def print_all_variables(self):
        """
        prints all variables from NetCDF to debug handler.
        :return: void
        example: self.print_all_variables()
        """
        logger.debug(str(self.root.variables))

    def print_all_variables_names(self):
        """
        Prints a list of all variable names to debug handler.
        :return: void
        example: self.print_all_variables_names()
        """
        for v in self.root.variables:
            logger.debug(str(v))

    def compute_qc_valid_range(self, var_name, min_range, max_range, flag):
        """
        Performs a valid range test as specified in SOCIB QC document.
        reads / creates / calls:
            reads _netCDF4.Variable (self.get_variable(var_name))       as  var
            accesses __array__().data to acquire numpy ndarray (calls method get_data_array)          as  var
            creates temporary list to store QC flags (ones actually)    as  my_list
            find NaN values by searching np.isnan values in var         as  nan_ind
            calls np.less & greater to get bool list ([False, True, False, ...]) where QC is applied    as  less and
                greater
            using logical or (|) to merge less and greater to get a complete bool list of entries to be marked
                as merged
            flags the entries in my_list[merged] with either flag mark or 9 for NaN values      as my_list[merged]=...
            calls self.fill_qc_list to set flags in self.qc_dict[0]: self.fill_qc_list(var_name, my_list, 1). The last
                parameter defines the flag to be set in the self.qc_dict[1] (flag which QC was applied) 1--> range
        :param var_name: str
        :param min_range: number
        :param max_range: number
        :param flag: int
        :return: void
        example: self.compute_qc_valid_range('AIR_TEM', -30, 60, 4)
        """
        var = self.get_variable(var_name)
        var = get_data_array(var)[self.corrected_starting_idx:self.end_idx]

        my_list = np.ones(len(self.time))

        nan_ind = np.where(np.isnan(var))[0]
        my_list[nan_ind] = 9
        self.fill_qc_list(var_name, my_list, 9)

        my_list = np.ones(len(self.time))

        less = np.less(var, min_range)
        greater = np.greater(var, max_range)
        merged = less | greater

        my_list[merged] = flag

        logger.info(str(len(np.argwhere(my_list == flag))) + " entries marked.")
        self.fill_qc_list(var_name, my_list, 1)

    def compute_qc_stationary(self, var_name, time_taken_into_account, discrepancy, flag):
        """
        Performs a stationary test as specified in SOCIB QC document. (Use of only good measurements not yet applied
            --> see TODO #2)
        Care! I assume, that the sampling is constant!
        Only gets it, when there are no outliers, or small deviations! (only max(x) / min(x) used) --> see TODO #1

        reads / creates / calls:
            converts hours to seconds (time_taken_into_account): X * 60 * 60 = X * minutes * seconds
                as  time_taken_into_account
            finds first index in time, where time_taken_into_account applies (first index of time[0]+interval) and uses
                it as interval for further calculations     as time_taken_interval
            sets first end index at the previous calculated interval        as end_time_idx
            sets start time index at time position 0        as start_time_idx
            reads _netCDF4.Variable (self.get_variable(var_name))       as  var
            accesses __array__().data to acquire numpy ndarray          as  var
            creates temporary list to store QC flags (ones actually)    as  my_list
            iterates through self.time in time_taken_interval steps
                calculates np.min & max between iterated start & end time       as  local_min & local_max
                takes abs() value of difference between max and min     as  difference_min_max
                checks if the difference is smaller or same than the discrepancy parameter
                    yes: flag all values in my_list at the interval with the specified flag
            calls self.fill_qc_list to set flags in self.qc_dict[0]: self.fill_qc_list(var_name, my_list, 2). The last
                parameter defines the flag to be set in the self.qc_dict[1] (flag which QC was applied) 2--> stationary

        #1 TODO: consider variance or use median or something similar for large time scales.
        #2 TODO: use only good measurements
        #3 TODO: check that interval is correct (that there is no ranging sampling; otherwise the steps are incorrect)

        :param var_name: str
        :param time_taken_into_account: number (HOURS! As in QC document described.)
        :param discrepancy: number
        (:param variance: number) --> not yet implemented
        :param flag: number
        :return: void
        example: self.compute_qc_stationary('AIR_TEM', 1, 0, 4)
        """
        time_taken_into_account = time_taken_into_account * 60 * 60
        # noinspection PyTypeChecker
        time_taken_interval = np.argmax(self.time > self.time[0] + time_taken_into_account)
        end_time_idx = time_taken_interval
        start_time_idx = 0

        var = self.get_variable(var_name)
        var = get_data_array(var)[self.corrected_starting_idx:self.end_idx]

        # testing purpose

        # get a workaround for too many NaNs
        # if this threshold is reached, the test is skipped
        nan_threshold_percent = 50

        my_list = np.ones(len(self.time))
        nan_idx = np.where(np.isnan(var))[0]

        while end_time_idx <= len(self.time):
            nan_percent = len(np.where(np.isnan(var[start_time_idx:end_time_idx]))[0]) * 100.0 / len(var[start_time_idx:end_time_idx])
            if nan_percent >= nan_threshold_percent:
                start_time_idx += 1
                end_time_idx += 1
                continue
            local_min = np.nanmin(var[start_time_idx:end_time_idx])
            local_max = np.nanmax(var[start_time_idx:end_time_idx])
            difference_min_max = abs(local_max - local_min)
            if difference_min_max <= discrepancy:
                logger.debug(
                    'Stationary values detected between ' + str(get_converted_time(self.time[start_time_idx])) +
                    ' and ' + str(get_converted_time(self.time[end_time_idx-1])))
                logger.debug('min: ' + str(local_min))
                logger.debug('max: ' + str(local_max))
                my_list[start_time_idx:end_time_idx] = flag
            start_time_idx += 1
            end_time_idx += 1
        my_list[nan_idx] = 9
        self.fill_qc_list(var_name, my_list, 2)
        logger.info(str(len(np.argwhere(my_list == flag))) + " entries marked.")

    def compute_qc_stationary_std(self, var_name, time_taken_into_account, std_threshold, flag):
        """
        Performs a stationary test as specified in SOCIB QC document. (Use of only good measurements not yet applied
            --> see TODO #2)
        Care! I assume, that the sampling is constant!
        Only gets it, when there are no outliers, or small deviations! (only max(x) / min(x) used) --> see TODO #1

        reads / creates / calls:
            converts hours to seconds (time_taken_into_account): X * 60 * 60 = X * minutes * seconds
                as  time_taken_into_account
            finds first index in time, where time_taken_into_account applies (first index of time[0]+interval) and uses
                it as interval for further calculations     as time_taken_interval
            sets first end index at the previous calculated interval        as end_time_idx
            sets start time index at time position 0        as start_time_idx
            reads _netCDF4.Variable (self.get_variable(var_name))       as  var
            accesses __array__().data to acquire numpy ndarray          as  var
            creates temporary list to store QC flags (ones actually)    as  my_list
            iterates through self.time in time_taken_interval steps
                calculates np.min & max between iterated start & end time       as  local_min & local_max
                takes abs() value of difference between max and min     as  difference_min_max
                checks if the difference is smaller or same than the std_threshold parameter
                    yes: flag all values in my_list at the interval with the specified flag
            calls self.fill_qc_list to set flags in self.qc_dict[0]: self.fill_qc_list(var_name, my_list, 2). The last
                parameter defines the flag to be set in the self.qc_dict[1] (flag which QC was applied) 2--> stationary

        #1 TODO: consider variance or use median or something similar for large time scales.
        #2 TODO: use only good measurements
        #3 TODO: check that interval is correct (that there is no ranging sampling; otherwise the steps are incorrect)

        :param var_name: str
        :param time_taken_into_account: number (HOURS! As in QC document described.)
        :param std_threshold: number
        (:param variance: number) --> not yet implemented
        :param flag: number
        :return: void
        example: self.compute_qc_stationary('AIR_TEM', 1, 0, 4)
        """
        time_taken_into_account = time_taken_into_account * 60 * 60
        # noinspection PyTypeChecker
        time_taken_interval = np.argmax(self.time > self.time[0] + time_taken_into_account)
        end_time_idx = time_taken_interval
        start_time_idx = 0

        var = self.get_variable(var_name)
        var = get_data_array(var)[self.corrected_starting_idx:self.end_idx]

        # var[1:731] = np.array(np.random.rand(1,730)/10) + 1019

        my_list = np.ones(len(self.time))
        nan_idx = np.where(np.isnan(var))[0]

        while end_time_idx < len(self.time):
            window_std = np.nanstd(var[start_time_idx:end_time_idx])
            if window_std <= std_threshold:
                logger.debug('Stationary values between ' + str(get_converted_time(self.time[start_time_idx])) +
                             ' and ' + str(get_converted_time(self.time[end_time_idx])))
                logger.debug('computed window std: ' + str(window_std))
                my_list[start_time_idx:end_time_idx] = flag
            start_time_idx += 1
            end_time_idx += 1
        my_list[nan_idx] = 9
        self.fill_qc_list(var_name, my_list, 2)
        logger.info(str(len(np.argwhere(my_list == flag))) + " entries marked.")

    def compute_qc_spike(self, var_name, limit, flag):
        """
        Performs a spike test as specified in SOCIB QC document. (takes 1 measurement before and after into
            consideration)

        reads / creates / calls:
            creates temporary list to store QC flags (ones actually)    as  my_list
            reads _netCDF4.Variable (self.get_variable(var_name))       as  var
            accesses __array__().data to acquire numpy ndarray          as  var
            starting at index 1 (not 0), for each element until end-1, calculate the spike      as spike
                if the spike value exceeds the limit parameter, the element at my_list will be flagged as with the flag
            calls self.fill_qc_list to set flags in self.qc_dict[0]: self.fill_qc_list(var_name, my_list, 3). The last
                parameter defines the flag to be set in the self.qc_dict[1] (flag which QC was applied) 3--> spike

        :param var_name: str
        :param limit: number
        :param flag: int
        :return: void
        example: self.compute_qc_spike('AIR_TEM', 3, 6)
        """
        # TODO: implement weights to spikes
        my_list = np.ones(len(self.time))
        var = self.get_variable(var_name)
        var = get_data_array(var)[self.corrected_starting_idx:self.end_idx]
        # 21.10.2015 AK change to use only good measurements
        run_through_interval = np.array(range(1, len(self.time) - 1, 1))
        nan_list = np.where(np.isnan(var))[0]
        del_index_list = []
        for n in nan_list:
            if 0 < n < len(self.time):
                # del_index_list.append(n-2)
                del_index_list.append(n - 1)
                # del_index_list.append(n)

        del_index_list = np.unique(del_index_list)
        run_through_interval = np.delete(run_through_interval, del_index_list)
        del_index_list = np.where(self.qc_dict[var_name][0] == 4)
        for n in del_index_list[0]:
            run_through_interval = np.delete(run_through_interval, np.where(run_through_interval == n))
        # run_through_interval = np.delete(run_through_interval, del_index_list)

        # intermediate case
        for i in run_through_interval[1:]:
            # for i in range(1, len(var)-1):
            good_measurements_time_before, good_measurements_values_before = \
                self.get_good_measurement_before(var_name, my_list, [i - 1], 0)
            good_measurements_time_after, good_measurements_values_after = \
                self.get_good_measurement_before(var_name, my_list, [i + 1], 1)
            # spike = abs(var[i]-(var[i+1]+var[i-1])/2)-abs((var[i+1]-var[i-1])/2)
            # TODO: insert into documentation
            # if there are no more good measurements after, break the for loop
            if len(good_measurements_values_after) == 0:
                break
            spike = abs(var[i] - (good_measurements_values_after[0] + good_measurements_values_before[0]) / 2) - \
                    abs((good_measurements_values_after[0] - good_measurements_values_before[0]) / 2)
            if spike > limit:
                logger.info("spike detected at " + get_converted_time(self.time[i]))
                logger.debug("spike value: " + str(spike))
                logger.debug("limit value: " + str(limit))
                my_list[i] = flag
        self.fill_qc_list(var_name, my_list, 3)
        logger.info(str(len(np.argwhere(my_list == flag))) + " entries marked.")

    def fill_qc_list(self, var_name, my_list, qc_marker):
        """
        Fills the central qcDict at the position var_name with the qc values from a QC test. the qc_marker will be set
        at qcDict[var_name][1] for each changed value. Only old_flags < new_flags will be changes. This means, no
        previously bad value (4), can be flagged as good (1) now.

        reads / creates / calls:
            finds the bool table (indexes) of the flags to be changed in qc_dict by comparing the entries of
                qc_dict[var_name][0] < entries of my_list. This is possible, since these two have the same dimension
                (time).     as  idx
            replaces the qc flags, as specified in the idx table.       as  self.qc_dict[var_name][0][idx]
            sets the qc_marker      as self.qc_dict[var_name][1][idx]

        :param var_name: str
        :param my_list: 1D list
        :param qc_marker: number
        :return: void
        example: self.fill_qc_list('AIR_TEM', [4, 4, 1, 1, ...], 3)
            --> qcDict before: [[1, 6, 1, 4],[0, 4, 0, 2]]
            --> qcDict after:  [[4, 6, 1, 4],[3, 4, 0, 2]]
        """
        idx = np.where(self.qc_dict[var_name][0] < my_list)
        self.qc_dict[var_name][0][idx] = my_list[idx]
        self.qc_dict[var_name][1][idx] = qc_marker
        # AK 15.10.2015
        self.qc_dict[var_name][2][idx] = False
        logger.info(str(len(idx[0])) + " entries changed in " + var_name + " qcDict.")

    def compare_qc_lists(self, own_qc_name, imported_qc_name):
        """
        compares the own qc results with the NetCDF imported QC values and plots a graph for this purpose
        reads / creates / calls:
            sets a new base axis for this plot (expected to plot different plots into one figure)
                as  self.base_axis
            detects indexes of values != 1 in the imported QC NetCDF variable (e.g. 'QC_AIR_TEM').
                as  imported_qc_name_idx
            detects indexes of values != 1 in the own QC dict (e.g. qc_dict['AIR_TEM'][0]).
                as  own_qc_name_idx
            detects, which values in the own QC dict not correspond with the imported values from NetCDF
                as  differences_idx
            plots the original data from NetCDF data (e.g. 'AIR_TEM'). Care! Actually this is only possible, when
                the original data name ('AIR_TEM') is also the name of the entry in the own qc dict.
                e.g. qc_dict['AIR_TEM']
                    uses NetCDF data as y axis, black as color and labels it as 'data'. (used for legend later)
            defines a new y axis for the qc values (visually the values on the right side)      as  new_axe
            plots the original QC data (e.g. 'QC_AIR_TEM')
                uses original QC data from the NetCDF file as y axis, blue as color, labels it as 'original' and uses
                    the new_axe to plot the values
            plots own QC results for this variable using qc_dict
                uses qc_dict[own_qc_name][0] as y axis data, green as color, labels as 'own' and uses the new_axe to
                    plot the values to
            plot the highlights / differences between own QC applied and imported from NetCDF file (If there are
                differences. Otherwise this part will be skipped.)
                Uses the input data ('AIR_TEM')[differences_idx] as y axis data, time[differences_idx] as x axis data,
                    sets the marker to *, colors it red, labels it as 'Diffs' (for the legend), sets the marker to plot
                    vertical lines at the data points to 1 and uses new_axe as y axis to plot the values to
            plot has to be made visible (via self.show_plot()) separately

        :param own_qc_name: str
        :param imported_qc_name: str
        :return: void
        example: self.compare_qc_lists('AIR_TEM','QC_AIR_TEM')

        #1 TODO: plot original data with different name than this of qc_dict[name]
        #2 TODO: define a variable interval for this comparison as input parameter
        """
        self.base_axis = plt.gca()
        get_start_idx = self.start_idx + self.corrected_starting_idx
        imported_qc_name_idx = np.where(self.get_variable(imported_qc_name)[get_start_idx:self.end_idx] != 1)
        own_qc_name_idx = np.where(self.qc_dict[own_qc_name][0][self.start_idx:] != 1)
        differences_idx = np.where(self.qc_dict[own_qc_name][0][self.start_idx:] != self.get_variable(imported_qc_name)[get_start_idx:self.end_idx])
        # imported_data_nan_list = np.where(np.isnan(self.get_variable(imported_qc_name)[0]))
        imported_data_nan_list = np.where(get_data_array(self.get_variable(imported_qc_name)[get_start_idx:self.end_idx]) == 9)
        # own_data_nan_list = np.where(np.isnan(self.qc_dict[own_qc_name][0]))
        own_data_nan_list = np.where(self.qc_dict[own_qc_name][0][self.start_idx:] == 9)
        applied_methods = np.unique(self.qc_dict[own_qc_name][1][self.start_idx:])

        for m in applied_methods:
            amount = np.where(self.qc_dict[own_qc_name][1][self.start_idx:] == m)
            if m == 0:
                logger.info(str(len(amount[0])) + " entries untouched")
            elif m == 1:
                logger.info(str(len(amount[0])) + " entries range test marked")
                self.differences_log['own']['bads']['validRange'] = len(amount[0])
            elif m == 2:
                logger.info(str(len(amount[0])) + " entries stationary test marked")
                self.differences_log['own']['bads']['stationary'] = len(amount[0])
            elif m == 3:
                logger.info(str(len(amount[0])) + " entries spike test marked")
                self.differences_log['own']['bads']['spike'] = len(amount[0])
            elif m == 4:
                logger.info(str(len(amount[0])) + " entries gradient test marked")
                self.differences_log['own']['bads']['gradient'] = len(amount[0])
            elif m == 9:
                logger.info(str(len(amount[0])) + " entries NaN marked")
                self.differences_log['own']['bads']['nans'] = len(amount[0])
            else:
                logger.warning(str(len(amount)) + " entries by undefined test marked")

        imported_nan_idx = np.where((self.get_variable(imported_qc_name)[get_start_idx:self.end_idx] == 9))
        self.differences_log['netcdf']['bads'] = len(imported_qc_name_idx[0]) - len(imported_nan_idx[0])
        self.differences_log['netcdf']['nans'] = len(imported_nan_idx[0])

        if len(differences_idx[0] > 0):
            bool_own_nans = self.qc_dict[own_qc_name][0][self.start_idx:] == 9
            bool_imported_nans = self.get_variable(imported_qc_name)[get_start_idx:self.end_idx] == 9

            nan_diffs = np.where(bool_own_nans != bool_imported_nans)[0]

            bool_own_bads = self.qc_dict[own_qc_name][0][self.start_idx:] != 1
            bool_own_bads = np.logical_xor(bool_own_bads, bool_own_nans)
            bool_imported_bads = self.get_variable(imported_qc_name)[get_start_idx:self.end_idx] != 1
            bool_imported_bads = np.logical_xor(bool_imported_bads, bool_imported_nans)

            bad_diffs = np.where(bool_own_bads != bool_imported_bads)[0]
            self.differences_log['diffs']['nans'] = len(nan_diffs)
            self.differences_log['diffs']['bads'] = len(bad_diffs)

        # insert extraction
        # check for differences
        # if len(differences_idx[0]) > 0:
        #     # find index of timestamps
        #     idx_last_timestamp = np.where(self.time >= self.start_time)[0]
        #     if idx_last_timestamp.any():
        #         idx_last_timestamp = np.where(self.time >= self.start_time)[0][0]
        #     idx_this_timestamp = np.where(self.time <= self.end_time)[0][-1]
        #     # self.start_idx = idx_last_timestamp
        #     # self.end_idx = idx_this_timestamp
        #     if idx_last_timestamp and idx_this_timestamp:
        #         is_in_range = True
        #
        #         logger.debug("idx last timestamp :" + str(idx_last_timestamp))
        #         logger.debug("idx this timestamp :" + str(idx_this_timestamp))
        #         # get full data
        #         netcdf_bad_list = np.where(get_data_array(self.get_variable(imported_qc_name))[get_start_idx:self.end_idx] == 4) or \
        #                           np.where(get_data_array(self.get_variable(imported_qc_name))[get_start_idx:self.end_idx] == 6)
        #         netcdf_nan_list = np.where(get_data_array(self.get_variable(imported_qc_name))[get_start_idx:self.end_idx] == 9)
        #
        #         own_bad_list = np.where(self.qc_dict[own_qc_name][0][self.start_idx:] == 4) or \
        #                        np.where(self.qc_dict[own_qc_name][0][self.start_idx:] == 6)
        #
        #         # get segment data
        #         # returns the index of the netcdf...xxx indices! --> e.g. [0 1 2 3 4] and not [7134 7135 7268 7269 7900]
        #         netcdf_bad_segment = np.where(netcdf_bad_list[0] <= idx_this_timestamp) and \
        #                              np.where(netcdf_bad_list[0] >= idx_last_timestamp)
        #         netcdf_nan_segment = np.where(netcdf_nan_list[0] <= idx_this_timestamp) and \
        #                              np.where(netcdf_nan_list[0] >= idx_last_timestamp)
        #         self.differences_log['netcdf']['bads'] = len(netcdf_bad_segment[0])
        #         self.differences_log['netcdf']['nans'] = len(netcdf_nan_segment[0])
        #
        #         own_nan_segment = np.where(own_data_nan_list[0] <= idx_this_timestamp) and \
        #                           np.where(own_data_nan_list[0] >= idx_last_timestamp)
        #         self.differences_log['own']['bads']['nans'] = len(own_nan_segment[0])
        #         own_bad_segment = np.where(own_bad_list[0] <= idx_this_timestamp) and \
        #                           np.where(own_bad_list[0] >= idx_last_timestamp)
        #         self.differences_log['own']['bads']['sum_excluding_nans'] = len(own_bad_segment[0])
        #
        #         # differences
        #         differences_including_nans_segment = np.where(differences_idx[0] <= idx_this_timestamp) and \
        #                                              np.where(differences_idx[0] >= idx_last_timestamp)
        #         differences_nans = len(netcdf_nan_segment[0]) - len(own_nan_segment[0])
        #         self.differences_log['diffs']['bads'] = len(differences_including_nans_segment[0]) - differences_nans
        #         self.differences_log['diffs']['nans'] = differences_nans
        #     else:
        #         is_in_range = False
        #         logger.debug("no new data for your station")
        # else:
        #     is_in_range = False
        #     logger.debug("no differences found")
        #     # check for differences between last timestamp and current timestamp
        # # end extraction
        #
        # for m in applied_methods:
        #     amount = np.where(self.qc_dict[own_qc_name][1][self.start_idx:] == m)
        #     if m == 0:
        #         logger.info(str(len(amount[0])) + " entries untouched")
        #     elif m == 1:
        #         logger.info(str(len(amount[0])) + " entries range test marked")
        #         if is_in_range:
        #             segment = np.where(amount[0] <= idx_this_timestamp) and np.where(amount[0] >= idx_last_timestamp)
        #             self.differences_log['own']['bads']['validRange'] = len(segment[0])
        #     elif m == 2:
        #         logger.info(str(len(amount[0])) + " entries stationary test marked")
        #         if is_in_range:
        #             segment = np.where(amount[0] <= idx_this_timestamp) and np.where(amount[0] >= idx_last_timestamp)
        #             self.differences_log['own']['bads']['stationary'] = len(segment[0])
        #     elif m == 3:
        #         logger.info(str(len(amount[0])) + " entries spike test marked")
        #         if is_in_range:
        #             segment = np.where(amount[0] <= idx_this_timestamp) and np.where(amount[0] >= idx_last_timestamp)
        #             self.differences_log['own']['bads']['spike'] = len(segment[0])
        #     elif m == 4:
        #         logger.info(str(len(amount[0])) + " entries gradient test marked")
        #         if is_in_range:
        #             segment = np.where(amount[0] <= idx_this_timestamp) and np.where(amount[0] >= idx_last_timestamp)
        #             self.differences_log['own']['bads']['gradient'] = len(segment[0])
        #     elif m == 9:
        #         logger.info(str(len(amount[0])) + " entries NaN marked")
        #     else:
        #         logger.warning(str(len(amount)) + " entries by undefined test marked")
        logger.info(str(len(own_qc_name_idx[0])) + " entries in " + own_qc_name)
        logger.info(str(len(imported_qc_name_idx[0])) + " entries in " + imported_qc_name)
        logger.info(str(len(imported_data_nan_list[0])) + " NaN values in " + imported_qc_name)
        logger.info(str(len(own_data_nan_list[0])) + " NaN values in " + own_qc_name)
        logger.info(str(len(differences_idx[0])) + " differences found")
        # no_qc_mean, imported_qc_mean, own_qc_mean = self.get_different_means(own_qc_name)
        # logger.debug("no qc mean: " + str(no_qc_mean))
        # logger.info("imported qc mean: " + str(imported_qc_mean))
        # logger.info("own qc mean: " + str(own_qc_mean))
        logger.info("----")

        # save difference plot
        self.plot_basic(get_data_array(self.get_variable(own_qc_name)[get_start_idx:self.end_idx]), x_axis=self.time[self.start_idx:], color_name='k', label_name='data', y_label=own_qc_name)

        new_axe = self.base_axis.twinx()
        self.plot_basic(get_data_array(self.get_variable(imported_qc_name)[get_start_idx:self.end_idx]), x_axis=self.time[self.start_idx:], color_name='b', label_name='original', axis=new_axe)

        self.plot_basic(get_data_array(self.qc_dict[own_qc_name][0][self.start_idx:]), x_axis=self.time[self.start_idx:], label_name='own', color_name='g', axis=new_axe)

        if len(differences_idx[0]) > 0:
            # print_differences_idx = differences_idx[0][differences_including_nans_segment[0]]
            self.plot_basic(get_data_array(self.get_variable(own_qc_name)[get_start_idx + differences_idx[0][:]]), self.time[self.start_idx + differences_idx[0]], '*', 'None',
                            'red', 'Diffs', 1, axis=new_axe)
        # self.show_plot((str(self.root.platform_code) + "-" + own_qc_name))

    def get_update_interval(self):
        """
        applies a lambda function upon the string stored in the attribute self.root.update_interval
            (e.g. 'every 600 seconds') to get the defined update interval (no calculation, just reading from dataset)

        :return: str (unicode)
            --> '600'
        example: self.get_update_interval()

        #1 TODO: maybe switch directly to int, float or whatever as return value, than string
        """
        return filter(lambda x: x.isdigit(), self.root.update_interval)

    def get_last_sample_interval(self):
        """
        computes the last sample interval from the last two measurements. No prior differences identified.
        :return: np.float64

        example: self.get_last_sample_interval()
            --> e.g. 60.0
        """
        if len(self.time) <= 1:
            logger.warning("Length of time <= 1. Sample Interval set to 0!")
            return 0
        return self.time[len(self.time) - 1] - self.time[len(self.time) - 2]

    def check_update(self):
        """
        checks, if the latest data received (in the NetCDF file) is received in the last
            self.update_interval*2+self.last_sample_interval.
        checks basically just the up-to-dateness of the netCDF file

        reads / creates / calls:
            computes the difference between actual time and the latest time value in the NetCDF file
            defines a threshold (self.update_interval*2+self.last_sample_interval)
            checks, if the difference > threshold, then log a warning, else log an info that everything seems fine

        :return: void

        example: self.check_update()
        """
        difference = time.time() - self.time[len(self.time) - 1]
        logger.info('Seconds since last update: ' + str(difference))
        threshold = int(self.update_interval * 2) + int(self.last_sample_interval)
        if difference > threshold:
            logger.warning(self.url + " threshold exceeded. Probably dead. Difference: " + str(difference) +
                           " seconds.")
        else:
            logger.info(self.url + " threshold NOT exceeded. Seems to be alive.")

    def get_bool_list(self, var_name, my_list):
        """

        :param var_name:
        :param my_list:
        :return:
        """
        bool_list_qc_dict = self.qc_dict[var_name][2]
        bool_list_my_list = my_list == 1
        merged_booleans = bool_list_qc_dict & bool_list_my_list
        return merged_booleans

    def get_good_measurement_before(self, var_name, my_list, index_array, flag_before_after):
        self.temp_boolean_list = []
        self.temp_boolean_list = np.array(self.get_bool_list(var_name, my_list))
        temp_value_output_list = []
        temp_time_output_list = []
        for idx in index_array:
            if self.temp_boolean_list[idx]:
                self.temp_boolean_list[idx] = False
                temp_value_output_list.append(self.temp_var_data[idx])
                temp_time_output_list.append(self.time[idx])
                continue
            elif not self.temp_boolean_list[idx]:
                if flag_before_after == 0:
                    logger.debug("Bad measurement (before) found at " + str(idx))
                    measurements_available = idx - 1
                elif flag_before_after == 1:
                    logger.debug("Bad measurement (after) found at " + str(idx))
                    measurements_available = len(self.time) - 1 - idx
                else:
                    logger.error("Wrong flag_before_after")
                    return [0], [0]
                if measurements_available < 0:
                    logger.debug("No good measurement available for " + str(idx) + " and flag " +
                                 str(flag_before_after))
                    temp_value_output_list.append(np.nan)
                    temp_time_output_list.append(np.nan)
                else:
                    counter = 1
                    while measurements_available > 0:
                        if flag_before_after == 0:
                            check_for_good = self.temp_boolean_list[idx - counter]
                        elif flag_before_after == 1:
                            check_for_good = self.temp_boolean_list[idx + counter]
                        if check_for_good:
                            if flag_before_after == 0:
                                logger.debug("Good measurement for index " + str(idx) + " found at index " +
                                             str(idx - counter))
                                self.temp_boolean_list[idx - counter] = False
                                temp_value_output_list.append(self.temp_var_data[idx - counter])
                                temp_time_output_list.append(self.time[idx - counter])
                            elif flag_before_after == 1:
                                logger.debug("Good measurement for index " + str(idx) + " found at index " +
                                             str(idx + counter))
                                self.temp_boolean_list[idx + counter] = False
                                temp_value_output_list.append(self.temp_var_data[idx + counter])
                                temp_time_output_list.append(self.time[idx + counter])
                            break
                        elif not check_for_good:
                            counter += 1
                            measurements_available -= 1
                        else:
                            logger.warning("Something unexpected happened here...")
            else:
                logger.error("boolean list error")
        return temp_time_output_list, temp_value_output_list

    def get_good_measurement_after(self):
        pass

    def get_different_means(self, var_name):
        data = get_data_array(self.get_variable(var_name))
        no_qc = get_mean(data)
        temp = get_data_array(self.get_variable("QC_" + var_name))
        idx = np.where(temp == 1)
        data_imported_qc = data[idx]
        imported_qc = get_mean(data_imported_qc)
        temp = self.qc_dict[var_name][0]
        idx = np.where(temp == 1)
        data_own_qc = data[idx]
        own_qc = get_mean(data_own_qc)
        return no_qc, imported_qc, own_qc

    # noinspection PyUnresolvedReferences
    def plot_basic(self, y_axis, x_axis=None, marker_name=None, linestyle_name=None, color_name=None, label_name=None,
                   vert_line_marker=None, axis=None, y_label=None):
        """
        plots data specified in the parameters. Standard x-axis is full time from NetCDF file
        reads / creates / calls:
            sets the color cycle for the figure (e.g. black, red, green, blue, yellow)
            sets correct values for "None" input parameters
            sets plots with the specified parameters
            sets X-ticks (from self.x_ticks --> actually three steps (every third of full time) )
            and the x ticks / x axis label(s)
            if the y axis has long_name and units attributes, the y axis will be labelled accordingly
                else, it gets a standard label
            for the legend, the plot object is appended to self.legend_names array
            function returns the actual axis plot object (actually not used for any purpose)

        :param y_axis: 1D list []
        ( :param x_axis: 1D list [] )
        ( :param marker_name: str (e.g. '*') )
        ( :param linestyle_name: str (e.g. '--') )
        ( :param color_name: str (e.g. 'k') )
        ( :param label_name: str (e.g. 'graph 1') )
        ( :param vert_line_marker: int (e.g. 1) )
        ( :param axis: matplotlib.axes._subplots.AxesSubplot (basically the 2nd y-Axis; y-Axis for QC values)
        :return: int (0 or 1) ... not used actually

        example: self.plot_basic(self.get_variable('AIR_TEM'), color_name='k', label_name='air temperature')
        #1 TODO: calculate x ticks according to x axis (not unitary)
        """
        # plt.rc('axes', color_cycle=['k', 'r', 'g', 'b', 'y'])
        xfmt = md.DateFormatter('%Y-%m-%d %H:%M')
        if x_axis is None:
            # x_axis = self.time
            dates = [datetime.fromtimestamp(ts) for ts in self.time]
            x_axis = md.date2num(dates)
        else:
            dates = [datetime.fromtimestamp(ts) for ts in x_axis]
            x_axis = md.date2num(dates)
        if marker_name is None:
            marker_name = ''
        if linestyle_name is None:
            linestyle_name = '-'
        if y_label is None:
            y_label = 'Y Axis'
        if vert_line_marker is not None:
            axis.vlines(x_axis, 0, 10, color=color_name)
            axis.set_ylabel('QC Flag')
            axis = self.base_axis
        if axis is None:
            axis = self.base_axis
        else:
            lns = axis.plot(x_axis, y_axis, marker=marker_name, linestyle=linestyle_name, color=color_name,
                            label=label_name)
            if label_name is not None:
                self.legend_names += lns
            return 1
        axis.xaxis.set_major_formatter(xfmt)

        lns = axis.plot(x_axis, y_axis, marker=marker_name, linestyle=linestyle_name, color=color_name,
                        label=label_name)
        # axis.set_xticks(self.x_ticks)
        # axis.set_xticklabels(self.x_ticks_label)

        # axis.set_xlabel(self.get_variable('time').standard_name + ' ' + self.get_variable('time').units)
        axis.set_xlabel('Date')
        if hasattr(y_axis, 'long_name') & hasattr(y_axis, 'units'):
            axis.set_ylabel(y_axis.long_name + ' ' + y_axis.units)
        else:
            axis.set_ylabel(y_label)
        if label_name is not None:
            self.legend_names += lns
        return axis

    # noinspection PyUnresolvedReferences
    def show_plot(self, title=None, directory=None):
        """
        shows the figure (halts the program), shows the legends and empties the self.base_axis, as well as the
        self.legend_names for new plots.

        reads / creates / calls:
            if a legend in self.legend_names is set, each entry is added to the self.base_axis.legend attribute.
            shows the figure (plot)
            empties base_axis and legend_names
        :return: void

        example: self.show_plot()
        """
        if len(self.legend_names) > 0:
            labs = [l.get_label() for l in self.legend_names]
            self.base_axis.legend(self.legend_names, labs, loc=0)
        if directory is None:
            directory = '.'
        if title is None:
            title = str(self.root.title)
        plt.title(title)
        fig_handle = plt.gcf()
        plt.xlim(self.date_converted_backward[self.start_idx], self.date_converted_backward[-1])
        self.base_axis.grid(b=False, which='major', color='k', linestyle='--', linewidth=0.25)
        fig_handle.autofmt_xdate()
        # plt.show()
        # plt.draw()
        plt.savefig(directory + "/" + title + '.png', bbox_inches='tight')
        pickle.dump(plt.gcf(), file(directory + "/" + title + '.pickle', 'w'), protocol=pickle.HIGHEST_PROTOCOL)
        self.base_axis = []
        self.legend_names = []
        plt.clf()

    def plot_variable(self, var_name):
        """
        var_name takes the name of the variable that should be plotted (from NetCDF file). Plots 2D graph along time
        axis. Can handle lists or single entries. For up to 4 variables to be printed, a single figure (with 4
        subplots) will be plotted. For any more, additional figures will be added.

        reads / creates / calls:
            if var_name is a list and if more than 4 entries (str) are stored
                split into arrays by up to 4 and call yourself (plot_variable) with these smaller arrays (v.tolist())
                e.g. 8 entries result in [array([1, 2, 3, 4],), array([5, 6, 7, 8])]
            else if var_name is a list, but contains less than 4 entries
                create a new figure
                check, if the list contains more than one entry and set the internally used length to 1
                if more than 2 entries are used, set the sublen to 2 (for clarification: sublen is used to determine
                    the number of subplots. If only two graphs are to plot, the figure will be split in 2 parts. If
                    there are more than 2 (3 or 4), the subplot will be split in 4 parts.
                    --> fig.add_subplot(2, 1, counter) or fig.add_subplot(2, 2, counter), where counter is used to
                    set, which plot is being added to the figure actually).
                    create a placeholder list l and set the counter to 1
                    for each variable name stored in the var_name list
                        get the variable data (self.get_variable(v))
                        and append it to the placeholder list with the specified subplot
                        self.plot_basic(y_axis) (data from variable)
                        increment the counter by one
                else (var_name is list but contains only a single entry)
                    simply plot the data from the self.get_variable function
            else (var_name is str and thus only a single entry)
                 simply plot the data from the self.get_variable function
            draw the plot


        :param var_name:str or list<str>
        :return:void
        example usage:
            1)single string input:plot_variable('AIR_TEM')
            2)list of string input:plot_variable(['AIR_TEM', 'AIR_PRE', 'REL_HUM', 'WIN_SPE'])
            3)single entry list input:plot_variable(['AIR_TEM', 'AIR_PRE'])
        """
        if isinstance(var_name, list):
            if len(var_name) > 4:
                var_name = np.array(var_name)
                amount_parts = len(var_name) / 4.0
                amount_parts = math.floor(amount_parts)
                amount_parts = int(amount_parts)
                temp_array = [4]
                for x in range(1, amount_parts):
                    temp_array.append((x + 1) * 4)
                sub_lists = np.split(var_name, temp_array)
                for v in sub_lists:
                    self.plot_variable(v.tolist())
            else:
                fig = plt.figure()
                if len(var_name) > 1:
                    sublen = 1
                    if len(var_name) > 2:
                        sublen = 2
                    l = [None] * 10
                    counter = 1
                    for v in var_name:
                        y_axis = self.get_variable(v)
                        l.append(fig.add_subplot(2, sublen, counter))
                        self.plot_basic(y_axis)
                        counter += 1
                else:
                    self.plot_basic(self.get_variable(var_name[0]))
        else:
            self.plot_basic(self.get_variable(var_name))
        plt.draw()


class Mooring(NCDFDataRoot):
    def __init__(self, url, name, start_time, end_time):
        """
        actually only calls the super constructor and sets the station name

        :param url: str
        :return: void
        """
        NCDFDataRoot.__init__(self, url, start_time, end_time)
        self.station_name = name

    # noinspection PyTypeChecker
    def compute_qc_gradient(self, var_name, time_taken_into_account, limit, flag):
        """
        computes a gradient test as specified in socib qc document.
        the general note: np.mean values are taken, if the step interval is bigger than one (if more than one data
            value is considered to obtain the specified time_taken_into_account)

        reads / creates / calls:
            creates a list of ones (size of length of time variable) for the temporary storage of the qc output
            gets the numpy array data from the variable (calls self.get_variable(var_name))
            gets the last sample interval
            and calculates the steps by dividing the the time_taken_into_account by this last_sample value
            the steps are now used for getting data from the time series
            -- since the initial case of the gradient test is handled different than the other cases, the initial case
                is calculated separately.
            -- again, the general note: np.mean values are taken, if the step interval is bigger than one (if more than
                one data value is considered to obtain the specified time_taken_into_account)
            the intermediate case does not differ much from the initial case. only weights are calculated and
            NOTE THAT c2-c3 and c1-c2 are casted from seconds to minutes! (by dividing by 60) --> otherwise the
            formula used in socib qc document does not work properly
            to check if the limit is exceeded, rounding to the np.float check_var is applied to the 10th decimal number,
                since rounding errors occurs by casting a float to numpy float


        :param var_name: str
        :param time_taken_into_account: number (in seconds)
        :param limit: number
        :param flag: int
        :return: void

        example: self.compute_qc_gradient('AIR_TEM', 60, 0.9, 4)

        #1 Note: if time_taken_into_account > lastSampleInterval, mean will be used. Perhaps change this
        #1 TODO: implement using only good measurements
        #2 TODO: implement ending case
        """

        my_list = np.ones(len(self.time))
        var = self.get_variable(var_name)
        var = get_data_array(var)[self.corrected_starting_idx:self.end_idx]

        logger.debug(str(self.get_last_sample_interval()))
        last_sample = self.get_last_sample_interval()
        divide = time_taken_into_account / last_sample
        if divide <= 1:
            steps = 1
        else:
            steps = int(math.ceil(divide))
        logger.debug("Regarding " + str(steps) + " as steps.")

        # initial case start
        v1 = var[0]
        c1 = self.time[0]
        v2 = np.mean(var[1:1 + steps])
        c2 = np.mean(self.time[1:1 + steps])
        check_value = abs((v1 - v2) / (c1 - c2))
        if check_value >= limit:
            logger.info("gradient detected at " + get_converted_time(self.time[0]))
            logger.debug("check val: " + str(check_value))
            logger.debug("limit: " + str(limit))
            my_list[0] = flag
            logger.debug("v2: " + get_converted_time(self.time[1]) + " value: " + str(v1))
            logger.debug("v3: " + get_converted_time(self.time[1 + steps]) + " value (perhaps mean): " + str(v2))
        # initial case end

        nan_list = np.where(np.isnan(var))[0]

        run_through_interval = np.array(range(steps, len(self.time) - steps, steps))

        del_index_list = []
        for n in nan_list:
            if 0 < n < len(self.time):
                # del_index_list.append(n-2)
                del_index_list.append(n - 1)
                # del_index_list.append(n)

        del_index_list = np.unique(del_index_list)
        run_through_interval = np.delete(run_through_interval, del_index_list)

        # intermediate case
        for i in run_through_interval[1:]:
            # change AK XXX 19.10.2015
            # if range(i-steps, i) in nan_list or range(i+1, i+steps+1) in nan_list or i in nan_list:
            #     logger.debug("nan value skip " + str(i))
            #     continue
            # ignore my_list entries (22.10.2015)
            temp_ignore_list = np.full(len(self.time), 1, dtype=np.int)
            good_measurements_time_before, good_measurements_values_before = \
                self.get_good_measurement_before(var_name, temp_ignore_list, range(i - steps, i), 0)
            good_measurements_time_after, good_measurements_values_after = \
                self.get_good_measurement_before(var_name, temp_ignore_list, range(i + 1, i + steps + 1), 1)

            # TODO: insert into documentation
            # if there are no more good measurements after, break the for loop
            if len(good_measurements_values_after) == 0:
                # my_list[i] = flag
                break

            # c1 = np.mean(float(var[i-steps:i]))
            c1 = np.mean(float(np.array(good_measurements_time_before)))
            c2 = float(self.time[i])
            # c3 = np.mean(float(self.time[i+1:i+steps+1]))
            c3 = np.mean(float(np.array(good_measurements_time_after)))

            w1 = (c1 - c2) / (c1 - c3)

            # w1 = c3 / c2
            w2 = (c2 - c3) / (c1 - c3)
            # w2 = c1 / c2

            # v1 = np.mean(var[i-steps:i])
            v1 = np.mean(np.array(good_measurements_values_before))
            v2 = var[i]
            # v3 = np.mean(var[i+1:i+steps+1])
            v3 = np.mean(np.array(good_measurements_values_after))

            check_value = abs(w1 * (v2 - v3) / ((c2 - c3) / 60) + w2 * (v1 - v2) / ((c1 - c2) / 60))
            # change AK XXX 15.10.2015
            # check_value = abs(w1 * (v2 - v3)/c1 + w2 * (v1 - v2)/c3)
            if round(check_value, 10) >= limit:
                logger.info("gradient detected at " + get_converted_time(self.time[i]))
                logger.debug("check val: " + str(check_value))
                logger.debug("limit: " + str(limit))
                my_list[i] = flag
                logger.debug("v1: " + get_converted_time(self.time[i - steps]) + " value (perhaps mean): " + str(v1))
                logger.debug("v2: " + get_converted_time(self.time[i]) + " value: " + str(v2))
                logger.debug("v3: " + get_converted_time(self.time[i + steps]) + " value (perhaps mean): " + str(v3))
        self.fill_qc_list(var_name, my_list, 4)
        logger.info(str(len(np.argwhere(my_list == flag))) + " entries marked.")

    def perform_qc(self, var_name, methods):
        """
        Calls the QC methods with the specified methods.

        reads / creates / calls:
            checks if there are enough input parameters
            creates a new entry in the qc_dict[var_name]
            for each method in first array, check if it is one of the specified functions (actually "range",
                "stationary", "spike" and "gradient")
                check, if enough input parameters are set
                call the methods

        :param var_name: str
        :param methods: list ([[], [], [] ]) see example - methods
        :return: void

        example:
        methods = ['range', 'range', 'stationary', 'spike', 'gradient', 'stationary'],\
                  [[-5, 35], [-30, 60], [6, 0], [3], [60, 0.9], [12, 0.2]],\
                  [2, 4, 4, 6, 4, 4]
        first array in methods describes the test that will be carried out. e.g. range calls the compute_qc_valid_range
        second array specifies the first set of arguments given to the qc functions (e.g. min / max temperature)
        third array describes the flags which will be set for the tests

        self.perform_qc('AIR_TEM', methods)
        """
        if len(methods[2]) != len(methods[0]):
            logger.error("Incorrect amount of flags with respect to the QC methods set.")
            return 4
        logger.info("create new QC List on " + var_name)
        self.append_qc_list(var_name)
        self.temp_var_data = get_data_array(self.get_variable(var_name))[self.corrected_starting_idx:self.end_idx]
        counter = 0
        for n in methods[0]:
            if n == "range":
                if len(methods[1][counter]) != 2:
                    logger.error("Not enough input parameters in range test specification (2 required)")
                else:
                    logger.info("Apply range test")
                    self.compute_qc_valid_range(var_name, methods[1][counter][0], methods[1][counter][1],
                                                methods[2][counter])
                    logger.info("Finished range test")
            elif n == "stationary":
                if len(methods[1][counter]) != 2:
                    logger.error("Not enough input parameters in stationary test specification (3 required)")
                else:
                    logger.info("Apply stationary test")
                    self.compute_qc_stationary(var_name, methods[1][counter][0], methods[1][counter][1],
                                               methods[2][counter])
                    logger.info("Finished stationary test")
            elif n == "spike":
                if len(methods[1][counter]) != 1:
                    logger.error("Not enough input parameters in spike test specification (1 required)")
                else:
                    logger.info("Apply spike test")
                    self.compute_qc_spike(var_name, methods[1][counter][0], methods[2][counter])
                    logger.info("Finished spike test")
            elif n == "gradient":
                if len(methods[1][counter]) != 2:
                    logger.error("Not enough input parameters in gradient test specification (2 required)")
                else:
                    logger.info("Apply gradient test")
                    self.compute_qc_gradient(var_name, methods[1][counter][0], methods[1][counter][1],
                                             methods[2][counter])
                    logger.info("Finished gradient test")
            elif n == "stationary_std":
                if len(methods[1][counter]) != 2:
                    logger.error("Not enough input parameters in gradient test specification (2 required)")
                else:
                    logger.info("Apply stationary std test")
                    self.compute_qc_stationary_std(var_name, methods[1][counter][0], methods[1][counter][1],
                                                   methods[2][counter])
                    logger.info("Finished stationary std test")
            else:
                logger.error("qc method called " + n + " not defined! " + n + " not applied.")
            counter += 1


class SurfaceDrifter(NCDFDataRoot):
    def __init__(self, url, name):
        NCDFDataRoot.__init__(self, url)
        self.station_name = name

    def compute_run_aground(self):
        pass


class Process:
    def __init__(self, title):
        self.title = title
        self.method_container = dict()
        # CARE! HardCoded lookup table here.
        # TODO- stuff that stuff into config and pack it into the corresponding functions
        self.method_lookup_table = {'validRange': 1, 'stationary': 2, 'spike': 3, 'gradient': 4, 'NaN': 9, 'nothing': 0}

    def add_method(self, name):
        self.method_container[name] = Method(name)

    def get_method(self, name):
        return self.method_container[name]


class Method:
    def __init__(self, title):
        self.title = title
        # self.method_dictionary = dict()
        self.method_names = []
        self.method_data = []
        self.flag_array = []

    def get_method_arrays(self):
        return self.method_names, self.method_data, self.flag_array

    def fill_dict(self, name, data, flag):
        self.method_names.append(name)
        self.method_data.append(data)
        self.flag_array.append(flag)

    def range(self, range_min, range_max, flag):
        name = 'range'
        data = [range_min, range_max]
        self.fill_dict(name, data, flag)

    def spike(self, threshold, flag):
        name = 'spike'
        data = [threshold]
        self.fill_dict(name, data, flag)

    def gradient(self, interval, threshold, flag):
        name = 'gradient'
        data = [interval, threshold]
        self.fill_dict(name, data, flag)

    def stationary(self, interval, threshold, flag):
        name = 'stationary'
        data = [interval, threshold]
        self.fill_dict(name, data, flag)

    def stationary_std(self, interval, threshold, flag):
        name = 'stationary_std'
        data = [interval, threshold]
        self.fill_dict(name, data, flag)
