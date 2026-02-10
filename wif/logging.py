import sys
import re
from datetime import datetime
import logging

stdout_orig = None
stderr_orig = None

# from https://stackoverflow.com/a/39215961
class _StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


class _StreamToFiles(object):
    """
    Fake file-like stream object that redirects writes to a file
    """
    def __init__(self, *files):
       self.files = files

    def write(self, buf):
        for single_file in self.files:
            single_file.write(buf)

    def flush(self):
        for single_file in self.files:
            single_file.flush()


def reset_logging():
    # back to python default
    logging.basicConfig(level=logging.WARNING)
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.WARNING)

    # clean up existing handlers
    for _ in range(len(rootLogger.handlers)):
        rootLogger.removeHandler(rootLogger.handlers[0])

    if stdout_orig is not None:
        sys.stdout = stdout_orig
    if stderr_orig is not None:
        sys.stderr = stderr_orig


def setup_logging(output_dir, output_prefix, wif_exec, prelim_warnings=[], *, level=logging.INFO):
    """Set up logging to stdout and a file, and print current time to log

    Parameters
    ----------
    output_dir: Path
        directory for logging file
    output_prefix: str
        prefix for logging file name
    wif_exec: str
        wif executable name to embed in filename
    level: loglevel, default logging.INFO
        level of logging
    """
    logging.basicConfig(level=level)

    rootLogger = logging.getLogger()
    rootLogger.setLevel(level)

    # clean up existing handlers
    for _ in range(len(rootLogger.handlers)):
        rootLogger.removeHandler(rootLogger.handlers[0])

    # find number of last log
    last_old_log_i = -1
    for old_log in output_dir.glob(output_prefix + f"wif_{wif_exec}.*.log"):
        m = re.search(rf".*wif_{wif_exec}.([0-9]+).log$", str(old_log))
        if m:
            old_log_i = int(m.group(1))
            last_old_log_i = max(last_old_log_i, old_log_i)
    log_i = last_old_log_i + 1

    myFormatter = logging.Formatter('%(asctime)s - %(message)s')

    # one copy to console
    global stdout_orig
    if stdout_orig is None:
        stdout_orig = sys.stdout
    consoleHandler = logging.StreamHandler(stdout_orig)
    consoleHandler.setFormatter(myFormatter)
    rootLogger.addHandler(consoleHandler)

    # one copy to file
    fileHandler = logging.FileHandler(output_dir / (output_prefix + f"wif_{wif_exec}.{log_i:02d}.log"))
    fileHandler.setFormatter(myFormatter)
    rootLogger.addHandler(fileHandler)

    # capture stdout and re-output using logging so it goes everywhere
    sys.stdout = _StreamToLogger(rootLogger, logging.INFO)

    ####################################################################################################
    # STDERR
    global stderr_orig
    if stderr_orig is None:
        stderr_orig = sys.stderr

    sys.stderr = _StreamToFiles(stderr_orig, open(output_dir / (output_prefix + f"wif_{wif_exec}.{log_i:02d}.err.log"), "w"))

    ####################################################################################################

    for warning in prelim_warnings:
        logging.warning(warning)

    cur_time = datetime.now()
    time_str = cur_time.strftime('%Y-%m-%d_%H:%M:%S')
    tz = cur_time.strftime('%z')
    if len(tz) > 0:
        time_str += "_" + tz
    logging.info(f"Start time: {time_str}")
