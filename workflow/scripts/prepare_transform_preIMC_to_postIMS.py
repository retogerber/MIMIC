import json
from utils import setNThreads, snakeMakeMock
import sys,os
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.input["preIMC_to_preIMS_transform"] = ""
    snakemake.input["preIMS_orig_size_transform"] = ""
    snakemake.input["preIMS_to_postIMS_transform"] = ""
    snakemake.output["preIMC_to_postIMS_transform"] = ""
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# inputs
transform_file_preIMC_to_preIMS = snakemake.input["preIMC_to_preIMS_transform"]
orig_size_tform_preIMC_to_preIMS=snakemake.input["preIMS_orig_size_transform"]
transform_file_preIMS_to_postIMS=snakemake.input["preIMS_to_postIMS_transform"]
# outputs
preIMC_to_postIMS_transform = snakemake.output["preIMC_to_postIMS_transform"]

logging.info("Read transformations")
# load all transforms
j1 = json.load(open(transform_file_preIMC_to_preIMS, "r"))
j2 = json.load(open(orig_size_tform_preIMC_to_preIMS, "r"))
# set correct spacing because of used downsampling=2 in registration
j2['Spacing'] = [ str(float(e)/2) for e in j2['Spacing']]
# j2 is a single transform, add a name to it
j2_new = {"orig_size_tform_preIMC_to_preIMS":j2}

j3 = json.load(open(transform_file_preIMS_to_postIMS, "r"))

logging.info("Combine transformations")
# combine all transforms
j1.update(j2_new)
j1.update(j3)

logging.info("Save transformations")
json.dump(j1, open(preIMC_to_postIMS_transform,"w"))

logging.info("Finished")