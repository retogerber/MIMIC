import sys,os
import logging, traceback
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


def setNThreads(n):
    if "cv2" in sys.modules:
        import cv2
        cv2.setNumThreads(n)
    if "SimpleITK" in sys.modules:
        import SimpleITK as sitk
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(n)
    if "segment_anything" in sys.modules:
        import torch
        torch.set_num_threads(n)

