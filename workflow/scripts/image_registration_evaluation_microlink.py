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
transform_file = "/home/retger/Downloads/Lipid_TMA_3781_042-IMSML-meta.json"
transform_file = snakemake.input["imsmicrolink_meta"]
logging.info("Read transform json file")
j0 = json.load(open(transform_file, "r"))
IMS = np.asarray(j0["IMS pixel map points (xy, microns)"])
postIMS = np.asarray(j0["PAQ microscopy points (xy, microns)"])

t1 = np.asarray(j0["Affine transformation matrix (xy,microns)"])
t2 = np.asarray(j0["Inverse Affine transformation matrix (xy,microns)"])
t3 = t1.copy()
t3[:2,:2] = np.array([[1-(t3[0,0]-1),-t3[1,0]],[-t3[0,1],1-(t3[1,1]-1)]])
t3[:2,2]=-t1[:2,2]
t4 = t2.copy()
t4[:2,:2] = np.array([[1-(t4[0,0]-1),-t4[1,0]],[-t4[0,1],1-(t4[1,1]-1)]])
t4[:2,2]=-t2[:2,2]


def mean_error(A, B, T):
    A_trans = np.matmul(A, T[:2,:2]) + T[:2,2]
    return np.mean(np.sqrt(np.sum((A_trans-B)**2,axis=1)))

logging.info("Apply transformations and calculate error")
e1 = mean_error(IMS, postIMS, t1)
e2 = mean_error(postIMS, IMS, t1)
e3 = mean_error(IMS, postIMS, t2)
e4 = mean_error(postIMS, IMS, t2)
e5 = mean_error(IMS, postIMS, t3)
e6 = mean_error(postIMS, IMS, t3)
e7 = mean_error(IMS, postIMS, t4)
e8 = mean_error(postIMS, IMS, t4)

ee = [e1, e2, e3, e4, e5, e6, e7, e8]

logging.info("Save json")
json.dump({"IMS_to_postIMS_error": np.min(ee)}, open(snakemake.output["IMS_to_postIMS_error"],"w"))

logging.info("Finished")


