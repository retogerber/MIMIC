import sys

def setNThreads(n: int) -> None:
    """
    Set the number of threads for certain modules.

    This function sets the number of threads for the cv2, SimpleITK, and segment_anything modules if they have been imported.
    It does this by calling the appropriate function for each module with the number of threads as an argument.

    Parameters:
    n (int): The number of threads to set.

    Returns:
    None
    """
    if "cv2" in sys.modules:
        import cv2
        cv2.setNumThreads(n)
    if "SimpleITK" in sys.modules or "wsireg" in sys.modules:
        import SimpleITK as sitk
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(n)
    if "segment_anything" in sys.modules or "torch" in sys.modules or "romatch" in sys.modules:
        import torch
        torch.set_num_threads(n)

class snakeMakeMock():
    def __init__(self):
        self.threads = 4
        self.params = dict()
        self.input = dict()
        self.output = dict()
        self.log = dict()
        self.log['stdout'] = None
