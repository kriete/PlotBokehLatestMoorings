import os
import errno
import time
import datetime
import logging.config
from logging_configuration import *
from main import PerformQC

__author__ = 'akrietemeyer'

logging.config.dictConfig(LOGGING_CONFIGURATION_NETCDF_LAYER)
logger = logging.getLogger("root")


def global_config():
    config = dict()
    config['dir_path'] = '/home/akrietemeyer/workspace/NotifyBokeh/notify_output'
    config['figure_path'] = 'figures'
    config['last_start_path'] = 'last_performed.txt'
    config['csv_output'] = 'stats.csv'
    config['date_format'] = '%Y%m%d%H%M%S'
    config['summary_output_file'] = 'summary.txt'
    config['thredds_catalog_url'] = 'http://thredds.socib.es/thredds/catalog/mooring/weather_station/catalog.html'
    return config


def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def read_last_starting_date(file_path):
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        file_handle = os.open(file_path, flags)
    except OSError as e:
        if e.errno == errno.EEXIST:  # Failed as the file already exists.
            with open(file_path, 'r') as file_obj:
                for line in file_obj:
                    pass
                return line
        else:  # Something unexpected went wrong so reraise the exception.
            raise
    else:  # No exception, so the file must have been created successfully.
        with os.fdopen(file_handle, 'w') as file_obj:
            logger.info("No previous file - file created now.")
            file_obj.write(str(time.mktime(datetime.datetime.now().timetuple())))
            return time.mktime(datetime.datetime.now().timetuple())


def overwrite_last_starting_date(file_path, time_str):
    with open(file_path, 'w') as this_file:
        this_file.write(time_str)


def main():
    cfg = global_config()
    dir_path = cfg['dir_path']
    create_directory(dir_path)
    file_name = cfg['last_start_path']
    date_format = cfg['date_format']
    last_starting_date = read_last_starting_date((dir_path + "/" + file_name))
    current_starting_date = time.mktime(datetime.datetime.now().timetuple())
    # overwrite_last_starting_date((dir_path + "/" + file_name), str(float(current_starting_date)))
    figure_directory = dir_path + "/figures/" + str(datetime.datetime.utcfromtimestamp(current_starting_date))
    create_directory(figure_directory)
    thredds_url = cfg['thredds_catalog_url']
    csv_path = (cfg['csv_output'])
    initialize_qc_perform = PerformQC(thredds_url, float(last_starting_date)-86000, float(current_starting_date), dir_path +
                                      "/" + csv_path, figure_directory, run_through=True)
    overwrite_last_starting_date((dir_path + "/" + file_name), str(float(current_starting_date)))


if __name__ == "__main__":
    main()
