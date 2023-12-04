import sys,os
import logging, traceback
from typing import Optional

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error(''.join(["Uncaught exception: ",
                         *traceback.format_exception(exc_type, exc_value, exc_traceback)
                         ])
                 )

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

def logging_setup(logfile: Optional[str]) -> None:
    # for interactive debugging
    if logfile is None:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s [%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            )
    # for running as a script
    else:
        logging.basicConfig(filename=logfile,
                            level=logging.INFO,
                            format='%(asctime)s [%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            )
        sys.excepthook = handle_exception
        sys.stdout = StreamToLogger(logging.getLogger(),logging.INFO)
        sys.stderr = StreamToLogger(logging.getLogger(),logging.ERROR)

def log_snakemake_info(snakemake) -> None:
    logging.info("Start")
    logging.info(f"snakemake config:")
    logging.info(f"\tthreads: {snakemake.threads}")
    for key in snakemake.params.keys():
        logging.info(f"\tparams: {key}: {snakemake.params[key]}")
    for key in snakemake.input.keys():
        logging.info(f"\tinput: {key}: {snakemake.input[key]}")
    for key in snakemake.output.keys():
        logging.info(f"\toutput: {key}: {snakemake.output[key]}")

