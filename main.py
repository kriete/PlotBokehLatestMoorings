from __future__ import division
import logging.config

from logging_configuration import *
from Data_Utils import Mooring, Method, Process
from urllib2 import Request, urlopen, URLError
from lxml import html
import copy
from datetime import datetime, timedelta
import numpy as np
import threading
import csv
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models import LinearAxis, Range1d, CustomJS
from bokeh.models.widgets import Panel, Tabs
from bokeh.io import output_notebook, show, output_file, gridplot, vplot, hplot
import bokeh
import pandas as pd
import os.path
import shutil
from datetime import datetime
from collections import OrderedDict

__author__ = 'akrietemeyer'

logging.config.dictConfig(LOGGING_CONFIGURATION_NETCDF_LAYER)
logger = logging.getLogger("root")
logging.getLogger("root").setLevel(logging.WARNING)

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


def get_pandas_timestamp_series(datetime_array):
    out = pd.Series(np.zeros(len(datetime_array)))
    counter = 0
    for i in datetime_array:
        out[counter] = pd.tslib.Timestamp(i)
        counter += 1
    return out


def find_name_in_dict(d, search_key):
    for k in d:
        if d[k] == search_key:
            return k


def get_str_time(x): return str(x)


def get_converted_time(stamp):
    return str(datetime.utcfromtimestamp(stamp))


def mkdirnotex(filename):
    folder=os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)


def totimestamp(dt, epoch=datetime(1970,1,1)):
    td = dt - epoch
    # return td.total_seconds()
    return (td.microseconds + (td.seconds + td.days * 86400) * 10**6) / 10**6


class PerformQC:
    def __init__(self, url, start_time_of_interest, end_time_of_interest, output_csv, figure_directory,
                 run_through=None, is_bokeh_plot=None):
        if run_through is None:
            run_through = False
        if is_bokeh_plot is None:
            is_bokeh_plot = False
        self.url = url
        self.name_list = []
        self.URLBuilder = []

        self.is_bokeh_plot = is_bokeh_plot

        self.start_time_of_interest = start_time_of_interest
        self.converted_start_time = get_converted_time(start_time_of_interest)
        self.start_time_year = self.converted_start_time[0:4]
        self.start_time_month = self.converted_start_time[5:7]
        self.end_time_of_interest = end_time_of_interest
        self.output_csv = output_csv
        self.figure_dir = figure_directory

        self.summary_log = dict()
        self.netcdf_bad_counter = 0
        self.own_bad_counter = 0
        self.differences_counter = 0

        self.output_data = dict()

        self.processes = dict()
        self.process_name0 = 'MeteoStation_Vaisala_Airp_Mbar'
        self.process_name1 = 'MeteoStation_Vaisala'
        self.process_name2 = 'MeteoStation_Aanderaa'
        self.process_name3 = 'Axys_WatchMate_Meteo'
        self.get_mooring_stations()
        self.define_processes()
        if run_through:
            self.run_through_stations()
        self.draw_bokeh()

    def get_mooring_stations(self):
        req = Request(self.url)
        try:
            response = urlopen(req)
        except URLError as e:
            if hasattr(e, 'reason'):
                print 'We failed to reach a server.'
                print 'Reason: ', e.reason
            elif hasattr(e, 'code'):
                print 'The server couldn\'t fulfill the request.'
                print 'Error code: ', e.code
        else:
            URLBuilder = []
            tree = html.fromstring(response.read())
            link_path = tree.xpath('//a')
            for x in range(1, len(link_path)):
                URLBuilder.append(link_path[x].values())
            URLLister = []
            for n in range(0, len(URLBuilder) - 4):
                string = str(URLBuilder[n])
                idx = string.find("/")
                # url = "http://thredds.socib.es/thredds/catalog/mooring/weather_station/" + URLBuilder[n][0][0:idx-1] + "/L1/catalog.html"
                url = "http://thredds.socib.es/thredds/catalog/mooring/weather_station/" + URLBuilder[n][0][
                                                                                           0:idx - 1] + "/L1/catalog.html"
                name = URLBuilder[n][0][0:idx - 2]
                req = Request(url)
                try:
                    response = urlopen(req)
                except URLError as e:
                    if hasattr(e, 'reason'):
                        print 'We failed to reach a server.'
                        print 'Reason: ', e.reason
                    elif hasattr(e, 'code'):
                        print 'The server couldn\'t fulfill the request.'
                        print 'Error code: ', e.code
                else:
                    URLLister.append(url)
                    self.name_list.append(name)

            for m in URLLister:
                req = Request(m)
                try:
                    response = urlopen(req)
                except URLError as e:
                    if hasattr(e, 'reason'):
                        print 'We failed to reach a server.'
                        print 'Reason: ', e.reason
                    elif hasattr(e, 'code'):
                        print 'The server couldn\'t fulfill the request.'
                        print 'Error code: ', e.code
                else:
                    tree = html.fromstring(response.read())
                    link_path = tree.xpath('//a')
                    for x in range(1, len(link_path)):
                        string = str(link_path[x].values())
                        idx = string.find("=")
                        self.URLBuilder.append("http://thredds.socib.es/thredds/dodsC/" + str(
                            link_path[x].values()[0][idx - 1:len(string)]))
                        break

    def define_processes(self):

        self.processes[self.process_name0] = Process(self.process_name0)
        self.processes[self.process_name0].add_method('AIR_PRE')
        self.processes[self.process_name0].method_container['AIR_PRE'].title = 'AIR_PRE'
        self.processes[self.process_name0].get_method('AIR_PRE').range(960, 1050, 2)
        self.processes[self.process_name0].get_method('AIR_PRE').range(920, 1080, 4)
        self.processes[self.process_name0].get_method('AIR_PRE').spike(10, 6)
        self.processes[self.process_name0].get_method('AIR_PRE').gradient(60, 0.3, 4)
        self.processes[self.process_name0].get_method('AIR_PRE').stationary(6, 0, 4)
        self.processes[self.process_name0].get_method('AIR_PRE').stationary(12, 0.6, 4)

        # Non-documented method input!!!
        # self.processes[self.process_name0].get_method('AIR_PRE').stationary_std(12, 0.05, 4)
        # End non-documented method input

        self.processes[self.process_name0].add_method('AIR_TEM')
        self.processes[self.process_name0].method_container['AIR_TEM'].title = 'AIR_TEM'
        self.processes[self.process_name0].get_method('AIR_TEM').range(-5, 40, 2)
        self.processes[self.process_name0].get_method('AIR_TEM').range(-30, 60, 4)
        self.processes[self.process_name0].get_method('AIR_TEM').spike(3, 6)
        self.processes[self.process_name0].get_method('AIR_TEM').gradient(60, 0.9, 4)
        self.processes[self.process_name0].get_method('AIR_TEM').stationary(6, 0, 4)
        self.processes[self.process_name0].get_method('AIR_TEM').stationary(12, 0.2, 4)

        self.processes[self.process_name0].add_method('REL_HUM')
        self.processes[self.process_name0].method_container['REL_HUM'].title = 'REL_HUM'
        self.processes[self.process_name0].get_method('REL_HUM').range(0, 100, 4)
        self.processes[self.process_name0].get_method('REL_HUM').spike(4, 6)
        self.processes[self.process_name0].get_method('REL_HUM').gradient(60, 3.6, 4)
        self.processes[self.process_name0].get_method('REL_HUM').stationary(6, 0, 4)
        self.processes[self.process_name0].get_method('REL_HUM').stationary(12, 1, 4)

        self.processes[self.process_name0].add_method('WIN_SPE')
        self.processes[self.process_name0].method_container['WIN_SPE'].title = 'WIN_SPE'
        self.processes[self.process_name0].get_method('WIN_SPE').range(0, 30, 2)
        self.processes[self.process_name0].get_method('WIN_SPE').range(0, 79, 4)
        self.processes[self.process_name0].get_method('WIN_SPE').spike(10, 6)
        self.processes[self.process_name0].get_method('WIN_SPE').gradient(60, 7.2, 4)
        self.processes[self.process_name0].get_method('WIN_SPE').stationary(6, 0, 4)
        self.processes[self.process_name0].get_method('WIN_SPE').stationary(12, 0.3, 4)

        self.processes[self.process_name1] = copy.deepcopy(self.processes[self.process_name0])
        self.processes[self.process_name1].title = self.process_name1
        self.processes[self.process_name1].method_container['AIR_PRE'].title = 'AIRP'
        self.processes[self.process_name1].method_container['AIR_TEM'].title = 'AIRT'
        self.processes[self.process_name1].method_container['REL_HUM'].title = 'RHUM'
        self.processes[self.process_name1].method_container['WIN_SPE'].title = 'WSPE_AVG'

        self.processes[self.process_name1].method_container['REL_HUM'].method_data[1] = [10]

        self.processes[self.process_name2] = copy.deepcopy(self.processes[self.process_name0])
        self.processes[self.process_name2].title = self.process_name2
        self.processes[self.process_name2].method_container['AIR_PRE'].title = 'APRE'
        self.processes[self.process_name2].method_container['AIR_TEM'].title = 'AIRT'
        self.processes[self.process_name2].method_container['REL_HUM'].title = 'RHUM'
        self.processes[self.process_name2].method_container['WIN_SPE'].title = 'WSPE'

        self.processes[self.process_name2].method_container['AIR_TEM'].method_data[3] = [300, 0.42]
        self.processes[self.process_name2].method_container['AIR_PRE'].method_data[4] = [12, 0]
        self.processes[self.process_name2].method_container['AIR_PRE'].method_data[5] = [24, 1]

        self.processes[self.process_name2].method_container['REL_HUM'].method_data[1] = [10]
        self.processes[self.process_name2].method_container['REL_HUM'].method_data[2] = [300, 2.4]
        self.processes[self.process_name2].method_container['REL_HUM'].method_data[4] = [24, 1]

        self.processes[self.process_name2].method_container['WIN_SPE'].method_data[2] = [7]
        self.processes[self.process_name2].method_container['WIN_SPE'].method_data[3] = [300, 0.54]
        self.processes[self.process_name2].method_container['WIN_SPE'].method_data[5] = [24, 0.3]

        self.processes[self.process_name3] = copy.deepcopy(self.processes[self.process_name0])
        self.processes[self.process_name3].title = self.process_name3
        self.processes[self.process_name3].method_container['AIR_TEM'].method_data[3] = [60, 0.42]
        self.processes[self.process_name3].method_container['AIR_TEM'].method_data[5] = [24, 0.2]
        self.processes[self.process_name3].method_container['AIR_PRE'].method_data[4] = [12, 0]
        self.processes[self.process_name3].method_container['AIR_PRE'].method_data[5] = [24, 0.4]
        self.processes[self.process_name3].method_container['REL_HUM'].method_data[1] = [10]
        self.processes[self.process_name3].method_container['REL_HUM'].method_data[2] = [60, 18]
        self.processes[self.process_name3].method_container['REL_HUM'].method_data[4] = [24, 1]
        self.processes[self.process_name3].method_container['WIN_SPE'].method_data[2] = [7]
        self.processes[self.process_name3].method_container['WIN_SPE'].method_data[3] = [60, 0.54]
        self.processes[self.process_name3].method_container['WIN_SPE'].method_data[5] = [24, 0.3]

    def execute_qc(self, process_name, station_name, temp_station):
        if not temp_station.return_flag:
            self.output_data[station_name] = dict()
        for qc_name in self.processes[process_name].method_container:
            methods = self.processes[process_name].method_container[qc_name].get_method_arrays()
            variable_name = self.processes[process_name].method_container[qc_name].title
            if not temp_station.return_flag:
                temp_station.perform_qc(variable_name, methods)
                temp_station.compare_qc_lists(variable_name, 'QC_' + str(variable_name))
                temp_station.show_plot(station_name + "-" + qc_name, self.figure_dir)

                selection_start_idx = temp_station.start_idx + temp_station.corrected_starting_idx
                self.output_data[station_name][qc_name] = dict()
                self.output_data[station_name][qc_name]['time_selection'] = temp_station.time[temp_station.start_idx:]
                self.output_data[station_name][qc_name]['data_selection'] = get_data_array(
                    temp_station.get_variable(variable_name))[selection_start_idx:temp_station.end_idx]
                self.output_data[station_name][qc_name]['own_qc'] = temp_station.qc_dict[variable_name][0][
                                                                    temp_station.start_idx:]
                self.output_data[station_name][qc_name]['applied_qc'] = temp_station.qc_dict[variable_name][1][
                                                                        temp_station.start_idx:]
                self.output_data[station_name][qc_name]['applied_qc_strings'] = [find_name_in_dict(
                    self.processes[process_name].method_lookup_table, x) for x in self.output_data[station_name]
                    [qc_name]['applied_qc']]
                self.output_data[station_name][qc_name]['imported_qc'] = get_data_array(
                    temp_station.get_variable('QC_' + str(variable_name)))[selection_start_idx:temp_station.end_idx]
                self.output_data[station_name][qc_name]['diffs'] = np.where(
                    self.output_data[station_name][qc_name]['own_qc'] != self.output_data[station_name][qc_name][
                        'imported_qc'])[0]
                self.output_data[station_name][qc_name]['converted_time'] = get_pandas_timestamp_series(
                    temp_station.date_converted[temp_station.start_idx:])

            self.print_difference_log(temp_station.differences_log, station_name, variable_name)
            self.summary_log[station_name] = dict()
            temp_station.set_up_differences_log()

    def draw_bokeh(self):
        output_stations = []
        output_stations_name = []
        html_file = '/home/akrietemeyer/workspace/NotifyBokeh/' + str(int(self.start_time_of_interest)) + '.html'
        filenames = os.listdir('/home/akrietemeyer/workspace/NotifyBokeh/')
        found = False
        for filename in filenames:
            if os.path.isfile('/home/akrietemeyer/workspace/NotifyBokeh/' + filename) and filename.endswith('.html'):
                found = True
                temp_filename = filename
            if found:
                archive_html_file_path = '/home/akrietemeyer/workspace/NotifyBokeh/notify_output/archive/' + self.start_time_year + '/' + self.start_time_month + '/' + \
                                         temp_filename
                mkdirnotex(archive_html_file_path)
                shutil.move('/home/akrietemeyer/workspace/NotifyBokeh/' + temp_filename, archive_html_file_path)
                found = False
        output_file(html_file)
        for station_name in self.output_data:
            output_stations_name.append(station_name)
            subplot = []
            tab_variables = []
            counter = 0
            for variable_name in self.output_data[station_name]:
                time = self.output_data[station_name][variable_name]['time_selection']
                data = self.output_data[station_name][variable_name]['data_selection']
                own_qc = self.output_data[station_name][variable_name]['own_qc']
                applied_qcs = self.output_data[station_name][variable_name]['applied_qc_strings']
                imported_qc = self.output_data[station_name][variable_name]['imported_qc']
                diffs = self.output_data[station_name][variable_name]['diffs']
                converted_time = self.output_data[station_name][variable_name]['converted_time']
                subplot.append(self.get_bokeh_grid_figure(time, data, own_qc, imported_qc, diffs, converted_time,
                                                          station_name, applied_qcs))
                tab_variables.append(variable_name)
                counter += 1
            # shared_x_range = subplot[0].x_range
            # subplot[1].x_range = shared_x_range
            # subplot[2].x_range = shared_x_range
            # subplot[3].x_range = shared_x_range
            tab1 = Panel(child=subplot[0], title=tab_variables[0])
            tab2 = Panel(child=subplot[1], title=tab_variables[1])
            tab3 = Panel(child=subplot[2], title=tab_variables[2])
            tab4 = Panel(child=subplot[3], title=tab_variables[3])
            # p = gridplot([[subplot[0], subplot[1]], [subplot[2], subplot[3]]])
            p = Tabs(tabs=[tab1, tab2, tab3, tab4])
            output_stations.append(p)
        amount_stations = len(output_stations)
        rest = amount_stations % 2
        verticals = []
        if amount_stations >= 2:
            verticals.append(hplot(output_stations[0], output_stations[1]))
        elif amount_stations == 1:
            verticals.append(hplot(output_stations[0]))
        else:
            logger.warning("No stations to plot (PerformQC.draw_bokeh()).")
            return 1
        for i in range(1, int(amount_stations/2)):
            verticals.append(hplot(output_stations[i*2], output_stations[i*2+1]))
        if rest > 0:
            verticals.append(output_stations[-1])
        show(vplot(*verticals))







    def get_bokeh_grid_figure(self, time, data, own_qc, imported_qc, diffs, converted_time, variable_name, applied_qcs):
        time_strings = map(get_str_time, converted_time)
        hover = HoverTool(names=["data"])
        fig = figure(width=800, plot_height=300, title=variable_name, tools=["pan, box_zoom, xwheel_zoom, save, reset, resize", hover], x_axis_type="datetime")

        source = ColumnDataSource(
            data=dict(
                time=time_strings,
                data=data,
                python_qc=own_qc,
                applied_qc=applied_qcs,
                imported_qc=imported_qc,
            )
        )
        # data line
        fig.line(converted_time, data, color="navy", alpha=0.5, name="data", source=source)
        # data points
        fig.square(converted_time, data, color="navy", alpha=0.5)
        fig.extra_y_ranges = {"foo": Range1d(start=0, end=10)}
        fig.add_layout(LinearAxis(y_range_name="foo"), 'right')
        fig.line(converted_time, own_qc, color="firebrick", alpha=0.5, y_range_name="foo")
        fig.line(converted_time, imported_qc, color="green", alpha=0.5, y_range_name="foo")
        zeros = np.zeros(len(diffs))
        tens = zeros[:] + 10
        fig.segment(converted_time[diffs], zeros, converted_time[diffs], tens, line_width=0.5, color="red",
                    y_range_name="foo")
        jscode = """
                range.set('start', parseInt(%s));
                range.set('end', parseInt(%s));
                """
        fig.extra_y_ranges['foo'].callback = CustomJS(
            args=dict(range=fig.extra_y_ranges['foo']),
            code=jscode % (fig.extra_y_ranges['foo'].start,
                           fig.extra_y_ranges['foo'].end)
        )
        pan_tool = fig.select(dict(type=bokeh.models.PanTool))
        pan_tool.dimensions = ["width"]

        hover = fig.select(dict(type=HoverTool))
        hover.tooltips = OrderedDict([
            ('time', '@time'),
            ('value', '@data{0.0}'),
            ('python qc', '@python_qc'),
            ('py-method', '@applied_qc'),
            ('imported qc', '@imported_qc'),

        ])

        # check for ranges, if they are nan
        if (np.isnan(np.nanmin(data)) & np.isnan(np.nanmax(data))) or (np.nanmin(data) == np.nanmax(data)):
            bottom_y_range = 0
            top_y_range = 10
        else:
            # add a 10% buffer to the max ranges
            temp_min = np.nanmin(data)
            temp_max = np.nanmax(data)
            temp_diff = abs(temp_max-temp_min)
            temp_thresh = round(temp_diff*0.1, 3)

            bottom_y_range = temp_min - temp_thresh
            top_y_range = temp_max + temp_thresh

        fig.y_range = Range1d(bottom_y_range, top_y_range)
        translate_time = converted_time.apply(lambda x: x.to_pydatetime())
        converted_time_backward = map(totimestamp, translate_time)
        source = ColumnDataSource({'x': converted_time_backward, 'y': data})

        jscode = """
        function isNumeric(n) {
          return !isNaN(parseFloat(n)) && isFinite(n);
        }
        var data = source.get('data');
        var start = yrange.get('start');
        var end = yrange.get('end');

        var time_start = xrange.get('start')/1000;
        var time_end = xrange.get('end')/1000;

        var pre_max_old = end;
        var pre_min_old = start;

        var time = data['x'];
        var pre = data['y'];
        t_idx_start = time.filter(function(st){return st>=time_start})[0];
        t_idx_start = time.indexOf(t_idx_start);

        t_idx_end = time.filter(function(st){return st>=time_end})[0];
        t_idx_end = time.indexOf(t_idx_end);

        var pre_interval = pre.slice(t_idx_start, t_idx_end);
        pre_interval = pre_interval.filter(function(st){return !isNaN(st)});
        var pre_max = Math.max.apply(null, pre_interval);
        var pre_min = Math.min.apply(null, pre_interval);
        var ten_percent = (pre_max-pre_min)*0.1;

        pre_max = pre_max + ten_percent;
        pre_min = pre_min - ten_percent;

        if((!isNumeric(pre_max)) || (!isNumeric(pre_min))) {
            pre_max = pre_max_old;
            pre_min = pre_min_old;
        }

        yrange.set('start', pre_min);
        yrange.set('end', pre_max);
        console.log(yrange.get('end'))

        source.trigger('change');
        """

        fig.y_range.callback = CustomJS(
            args=dict(source=source, yrange=fig.y_range, xrange=fig.x_range), code=jscode)
        fig.x_range.callback = CustomJS(
            args=dict(source=source, yrange=fig.y_range, xrange=fig.x_range), code=jscode)
        return fig

    def print_difference_log(self, difference_log, station_name, variable_name):
        with open(self.output_csv, 'ab+') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([str(datetime.utcfromtimestamp(self.start_time_of_interest)).replace(' ', '_'),
                             '-', str(datetime.utcfromtimestamp(self.end_time_of_interest)).replace(' ', '_'), '', '',
                             '', '', '', '', '', ''])
            writer.writerow([station_name, variable_name, 'NetCDF', '', 'Python', '', '', '', '', 'Diffs', ''])
            writer.writerow(['', '', 'Bads', 'NaNs', 'Bads', '', '', '', 'NaNs', 'Bads', 'NaNs'])
            writer.writerow(['', '', '', '', 'Range', 'Stationary', 'Spike', 'Gradient', '', '', ''])
            writer.writerow(['', '', difference_log['netcdf']['bads'], difference_log['netcdf']['nans'],
                             difference_log['own']['bads']['validRange'], difference_log['own']['bads']['stationary'],
                             difference_log['own']['bads']['spike'], difference_log['own']['bads']['gradient'],
                             difference_log['own']['bads']['nans'], difference_log['diffs']['bads'],
                             difference_log['diffs']['nans']])
            self.netcdf_bad_counter = self.netcdf_bad_counter + difference_log['netcdf']['bads']
            self.own_bad_counter = self.own_bad_counter + (difference_log['own']['bads']['validRange'] +
                                                           difference_log['own']['bads']['stationary'] +
                                                           difference_log['own']['bads']['spike'] +
                                                           difference_log['own']['bads']['gradient'])
            self.differences_counter = self.differences_counter + (difference_log['diffs']['bads'] +
                                                                   difference_log['diffs']['nans'])

    def print_summary_log(self, station_name):
        with open(self.output_csv, 'ab+') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Summary:', str(self.netcdf_bad_counter), 'new_Bad_data_in_NetCDF_files', '', '', '', '',
                             '', '', '', ''])
            writer.writerow(['', str(self.own_bad_counter), 'new_Bad_data_in_Python_implementation', '', '', '', '',
                             '', '', '', ''])
            writer.writerow(['', str(self.differences_counter), 'Differences_detected', '', '', '', '',
                             '', '', '', ''])

        summary_file_name = "/home/akrietemeyer/workspace/NotifyBokeh/summary.txt"
        # print summary_file_name
        with open(summary_file_name, 'a') as text_file:
            text_file.write(station_name + "\n")
            text_file.write("Summary:\n" + str(self.netcdf_bad_counter) + " new Bad Data since " +
                            str(datetime.utcfromtimestamp(self.start_time_of_interest)) + " in NetCDF document\n")
            text_file.write(str(self.own_bad_counter) + " new Bad Data since " +
                            str(datetime.utcfromtimestamp(self.start_time_of_interest)) + " in python implementation\n")
            text_file.write(str(self.differences_counter) +
                            " Differences between QC in NetCDF document and own python implementation since " +
                            str(datetime.utcfromtimestamp(self.start_time_of_interest)) + "\n\n")
        self.own_bad_counter = 0
        self.netcdf_bad_counter = 0
        self.differences_counter = 0

    def run_through_stations(self):
        counter = 0
        for station in self.URLBuilder:
            temp = Mooring(station, self.name_list[counter], self.start_time_of_interest, self.end_time_of_interest)
            if self.name_list[counter] == "station_salines-ime_met002":
                # 'MeteoStation_Aanderaa'
                self.execute_qc(self.process_name2, self.name_list[counter], temp)
                self.print_summary_log(self.name_list[counter])

            elif self.name_list[counter] == "station_parcbit-scb_met004" or self.name_list[counter] == \
                    "station_galfi-scb_met005" or self.name_list[counter] == \
                    "station_esporles-scb_met003" or self.name_list[counter] == \
                    "mobims_playadepalma-scb_met006" or self.name_list[counter] == "mobims_playadepalma-scb_met006":
                # 'MeteoStation_Vaisala_Airp_Mbar'
                self.execute_qc(self.process_name0, self.name_list[counter], temp)
                self.print_summary_log(self.name_list[counter])

            elif self.name_list[counter] == "mobims_playadepalma-scb_met003" or self.name_list[counter] == \
                    "mobims_sonbou-scb_met002" or self.name_list[counter] == "mobims_calamillor-scb_met001":
                # 'MeteoStation_Vaisala'
                self.execute_qc(self.process_name1, self.name_list[counter], temp)
                self.print_summary_log(self.name_list[counter])

            elif self.name_list[counter] == "buoy_canaldeibiza-scb_met010" or self.name_list[counter] == \
                    "buoy_bahiadepalma-scb_met010" or self.name_list[counter] == "buoy_bahiadepalma-scb_met008":
                # 'Axys_WatchMate_Meteo'
                self.execute_qc(self.process_name3, self.name_list[counter], temp)
                self.print_summary_log(self.name_list[counter])
            else:
                # 'MeteoStation_Vaisala'
                logger.warning('Undefined station: ' + self.name_list[counter] + '. Default MeteoStation_Vaisala used.')
                self.execute_qc(self.process_name0, self.name_list[counter], temp)
                self.print_summary_log(self.name_list[counter])
            counter += 1


def main():
    asd = PerformQC("http://thredds.socib.es/thredds/catalog/mooring/weather_station/catalog.html")
    asd.get_mooring_stations()
    asd.define_processes()
    asd.test_case_example()
    # asd.run_through_stations()

    url = "http://thredds.socib.es/thredds/dodsC/mooring/weather_station/station_parcbit-scb_met004/L1/dep0002_station-parcbit_scb-met004_L1_latest.nc"

    url_list = []
    url_list.append(
        "http://thredds.socib.es/thredds/dodsC/mooring/weather_station/station_parcbit-scb_met004/L1/dep0002_station-parcbit_scb-met004_L1_latest.nc")

    moorings = []

    # for n in url_list:
    #     moorings.append(Mooring(n))

    name_list = []

    req = Request("http://thredds.socib.es/thredds/catalog/mooring/weather_station/catalog.html")
    try:
        response = urlopen(req)
    except URLError as e:
        if hasattr(e, 'reason'):
            print 'We failed to reach a server.'
            print 'Reason: ', e.reason
        elif hasattr(e, 'code'):
            print 'The server couldn\'t fulfill the request.'
            print 'Error code: ', e.code
    else:
        URLBuilder = []
        tree = html.fromstring(response.read())
        link_path = tree.xpath('//a')
        for x in range(1, len(link_path)):
            URLBuilder.append(link_path[x].values())
        URLLister = []
        for n in range(0, len(URLBuilder) - 4):
            string = str(URLBuilder[n])
            idx = string.find("/")
            url = "http://thredds.socib.es/thredds/catalog/mooring/weather_station/" + URLBuilder[n][0][
                                                                                       0:idx - 1] + "/L1/catalog.html"
            name = URLBuilder[n][0][0:idx - 2]
            req = Request(url)
            try:
                response = urlopen(req)
            except URLError as e:
                if hasattr(e, 'reason'):
                    print 'We failed to reach a server.'
                    print 'Reason: ', e.reason
                elif hasattr(e, 'code'):
                    print 'The server couldn\'t fulfill the request.'
                    print 'Error code: ', e.code
            else:
                URLLister.append(url)
                name_list.append(name)
        URLBuilder = []
        for m in URLLister:
            req = Request(m)
            try:
                response = urlopen(req)
            except URLError as e:
                if hasattr(e, 'reason'):
                    print 'We failed to reach a server.'
                    print 'Reason: ', e.reason
                elif hasattr(e, 'code'):
                    print 'The server couldn\'t fulfill the request.'
                    print 'Error code: ', e.code
            else:
                tree = html.fromstring(response.read())
                link_path = tree.xpath('//a')
                for x in range(1, len(link_path)):
                    string = str(link_path[x].values())
                    idx = string.find("=")
                    URLBuilder.append(
                        "http://thredds.socib.es/thredds/dodsC/" + str(link_path[x].values()[0][idx - 1:len(string)]))
                    break
    mooring_list = []
    counter = 0

    for station in URLBuilder:
        temp = Mooring(station, name_list[counter])
        if name_list[counter] == "station_salines-ime_met002":
            # methods = ['range', 'range', 'stationary', 'spike', 'gradient', 'stationary'],\
            #       [[-5, 40], [-30, 60], [6, 0], [3], [300, 0.42], [12, 0.2]],\
            #       [2, 4, 4, 6, 4, 4]

            # TODO: automate that stuff, by running through the process_2 method container. Catch the var_names with
            # TODO: the titles
            process = process_2
            methods = process.get_method('AIR_TEM').get_method_arrays()
            var_name = process.method_container['AIR_TEM'].title
            temp.perform_qc(var_name, methods)
            temp.compare_qc_lists(var_name, 'QC_' + str(var_name))
            temp.show_plot(name_list[counter])

            # methods = ['range', 'range', 'stationary', 'spike', 'gradient', 'stationary'],\
            #           [[960, 1050], [920, 1080], [12, 0], [10], [300, 0.3], [24, 1]],\
            #           [2, 4, 4, 6, 4, 4]
            methods = process.get_method('AIR_PRE').get_method_arrays()
            var_name = process.method_container['AIR_PRE'].title
            temp.perform_qc(var_name, methods)
            temp.compare_qc_lists(var_name, 'QC_' + str(var_name))
            temp.show_plot(name_list[counter])

        elif name_list[counter] == "mobims_sonbou-scb_met002":
            methods = ['range', 'range', 'stationary', 'spike', 'gradient', 'stationary'], \
                      [[-5, 40], [-30, 60], [6, 0], [3], [60, 0.9], [12, 0.2]], \
                      [2, 4, 4, 6, 4, 4]

            temp.perform_qc('AIRT', methods)
            temp.compare_qc_lists('AIRT', 'QC_AIRT')
            temp.show_plot(name_list[counter])

            methods = ['range', 'range', 'stationary', 'spike', 'gradient', 'stationary'], \
                      [[960, 1050], [920, 1080], [6, 0], [10], [60, 0.3], [12, 0.6]], \
                      [2, 4, 4, 6, 4, 4]
            temp.perform_qc('AIRP', methods)
            temp.compare_qc_lists('AIRP', 'QC_AIRP')
            temp.show_plot(name_list[counter])

        elif name_list[counter] == "mobims_playadepalma-scb_met003" or name_list[
            counter] == 'mobims_calamillor-scb_met001':
            methods = ['range', 'range', 'stationary', 'spike', 'gradient', 'stationary'], \
                      [[-5, 35], [-30, 60], [6, 0], [3], [60, 0.9], [12, 0.2]], \
                      [2, 4, 4, 6, 4, 4]

            temp.perform_qc('AIRT', methods)
            temp.compare_qc_lists('AIRT', 'QC_AIRT')
            temp.show_plot(name_list[counter])
        elif name_list[counter] == "buoy_canaldeibiza-scb_met010" or name_list[
            counter] == "buoy_bahiadepalma-scb_met010" or name_list[counter] == "buoy_bahiadepalma-scb_met008":
            methods = ['range', 'range', 'stationary', 'spike', 'gradient', 'stationary'], \
                      [[-5, 40], [-30, 60], [6, 0], [3], [60, 0.9], [24, 0.2]], \
                      [2, 4, 4, 6, 4, 4]
            temp.perform_qc('AIR_TEM', methods)
            temp.compare_qc_lists('AIR_TEM', 'QC_AIR_TEM')
            temp.show_plot(name_list[counter])
        else:
            methods = ['range', 'range', 'stationary', 'spike', 'gradient', 'stationary'], \
                      [[-5, 35], [-30, 60], [6, 0], [3], [60, 0.9], [12, 0.2]], \
                      [2, 4, 4, 6, 4, 4]
            temp.perform_qc('AIR_TEM', methods)
            temp.compare_qc_lists('AIR_TEM', 'QC_AIR_TEM')
            temp.show_plot(name_list[counter])

            methods = ['range', 'range', 'stationary', 'spike', 'gradient', 'stationary'], \
                      [[960, 1050], [920, 1080], [6, 0], [10], [60, 0.3], [12, 0.6]], \
                      [2, 4, 4, 6, 4, 4]

        # A.plot_basic(A.qcDict['AIR_TEM'][0])
        # A.printActualQC('QC_AIR_TEM')


        counter += 1

    pass
    url = "http://thredds.socib.es/thredds/dodsC/mooring/weather_station/station_parcbit-scb_met004/L1/dep0002_station-parcbit_scb-met004_L1_latest.nc"
    A = Mooring(url, "parcbit")

    # B = A.getVariable('AIR_TEM')


    # A.plot_variable('AIR_TEM')
    # A.show_plot()

    # A.compute_qc_gradient('AIR_TEM', 60, 0.25)
    # methods = ['range', 'stationary', 'spike', 'gradient'], [[-30, 60], [6, 0, 0], [3], [60, 0.25]]
    methods = ['range', 'range', 'stationary', 'spike', 'gradient', 'stationary'], \
              [[-5, 35], [-30, 60], [6, 0], [3], [60, 0.9], [12, 0.2]], \
              [2, 4, 4, 6, 4, 4]
    A.perform_qc('AIR_TEM', methods)
    # A.plot_basic(A.qcDict['AIR_TEM'][0])
    # A.printActualQC('QC_AIR_TEM')
    A.compare_qc_lists('AIR_TEM', 'QC_AIR_TEM')
    A.show_plot()

    methods = ['range', 'range', 'stationary', 'spike', 'gradient', 'stationary'], \
              [[960, 1050], [920, 1080], [6, 0], [10], [60, 3], [12, 0.6]], \
              [2, 4, 4, 6, 4, 4]
    A.perform_qc('AIR_PRE', methods)
    A.compare_qc_lists('AIR_PRE', 'QC_AIR_PRE')
    A.show_plot()

    url = "http://thredds.socib.es/thredds/dodsC/mooring/weather_station/station_galfi-scb_met005/L1/dep0001_station-galfi_scb-met005_L1_latest.nc"
    B = Mooring(url, "galfi")
    methods = ['range', 'range', 'stationary', 'spike', 'gradient', 'stationary'], \
              [[-5, 35], [-30, 60], [6, 0], [3], [60, 0.9], [12, 0.2]], \
              [2, 4, 4, 6, 4, 4]
    B.perform_qc('AIR_TEM', methods)
    B.compare_qc_lists('AIR_TEM', 'QC_AIR_TEM')
    B.show_plot()



    # while True:
    #     temp = Mooring(url)
    #     temp.checkUpdate()
    #     time.sleep(5)
    # threading.Timer(10, i.checkUpdate).start()


if __name__ == "__main__":
    main()
