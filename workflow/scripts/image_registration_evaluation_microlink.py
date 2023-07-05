import json
import numpy as np
import sys,os
import logging, traceback
logging.basicConfig(filename=snakemake.log["stdout"],
                    level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    )
from logging_utils import handle_exception, StreamToLogger
sys.excepthook = handle_exception
sys.stdout = StreamToLogger(logging.getLogger(),logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger(),logging.ERROR)

logging.info("Start")
# transform_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMS/postIMS_to_IMS_test_combined-IMSML-meta.json"
# transform_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMS/postIMS_to_IMS_test_combined-IMSML-meta.json"
transform_file = snakemake.input["imsmicrolink_meta"]
logging.info("Read transform json file")
j0 = json.load(open(transform_file, "r"))
IMS = np.asarray(j0["IMS pixel map points (xy, microns)"])
postIMS = np.asarray(j0["PAQ microscopy points (xy, microns)"])

t1 = np.asarray(j0["Affine transformation matrix (xy,microns)"])
t2 = np.asarray(j0["Inverse Affine transformation matrix (xy,microns)"])

def mean_error(A, B, T):
    A_trans = np.matmul(A, T[:2,:2]) + T[:2,2]
    dists = [ np.linalg.norm(A_trans[i,:]-B[i,:]) for i in range(A.shape[0])]
    return np.mean(dists)

logging.info("Apply transformations and calculate error")
e1 = mean_error(IMS, postIMS, t1)
e2 = mean_error(postIMS, IMS, t1)
e3 = mean_error(IMS, postIMS, t2)
e4 = mean_error(postIMS, IMS, t2)

ee = [e1, e2, e3, e4]

logging.info("Save json")
json.dump({"IMS_to_postIMS_error": np.min(ee)}, open(snakemake.output["IMS_to_postIMS_error"],"w"))

logging.info("Finished")


