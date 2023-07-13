import json
import sys,os
import logging, traceback
logging.basicConfig(filename=snakemake.log["stdout"],
                    level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    )
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

#transform_file_preIMC_to_preIMS="/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/registrations/preIMC_to_preIMS/1/NASH_HCC_TMA_1-preIMC_to_preIMS_transformations.json"
transform_file_preIMC_to_preIMS = snakemake.input["preIMC_to_preIMS_transform"]
#orig_size_tform_preIMC_to_preIMS="/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/registrations/preIMC_to_preIMS/1/.imcache_NASH_HCC_TMA_1/preIMS_orig_size_tform.json"
orig_size_tform_preIMC_to_preIMS=snakemake.input["preIMS_orig_size_transform"]
#transform_file_preIMS_to_postIMS="/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/registrations/postIMC_to_postIMS/NASH_HCC_TMA-preIMS_to_postIMS_transformations.json"
transform_file_preIMS_to_postIMS=snakemake.input["preIMS_to_postIMS_transform"]

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