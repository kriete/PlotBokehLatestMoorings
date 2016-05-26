import os


class bcolors:

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Get the root path of the socib_python_tools package. It will be the base path of the logging files
# /path/organize_directory/logs
#
# The logging configuration file must be in /path/socib_python_tools/configuration/
logs_base_path = os.path.split(os.path.dirname(__file__))[0]

LOGGING_CONFIGURATION_NETCDF_LAYER = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s - %(filename)s %(lineno)s - %(levelname)s - %(message)s"
        # },
        # "simpleERR": {
        #     "format": bcolors.WARNING + "%(asctime)s - %(filename)s %(lineno)s - %(levelname)s \n %(message)s" +
        }
    },

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        },

        "info_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "simple",
            "filename": "info.log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        },

        "error_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "simple",
            "filename": "errors.log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        },

        "debug_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filename": "debug.log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        }
    },

    "loggers": {
        "my_module": {
            "level": "ERROR",
            "handlers": ["console"],
            "propagate": "no"
        }
    },
    # "root": {
    #    "level": "INFO",
    #     "handlers": ["console", "info_file_handler", "error_file_handler"]
    # },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "info_file_handler", "error_file_handler", "debug_file_handler"]
    }
}