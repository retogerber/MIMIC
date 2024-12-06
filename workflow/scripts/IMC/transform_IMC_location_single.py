import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
import json
from utils import setNThreads, snakeMakeMock
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.input['IMC_location_transformed'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_location/test_split_pre_IMC_mask_on_preIMS_combined.geojson"
    snakemake.output['IMC_location_transformed_single'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_location/test_split_pre_IMC_mask_on_preIMS_A1.geojson"
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# inputs
IMC_location_transformed = snakemake.input['IMC_location_transformed']
if isinstance(IMC_location_transformed, list):
    IMC_location_transformed = IMC_location_transformed[0]

# outputs
IMC_location_transformed_single = snakemake.output['IMC_location_transformed_single']
if isinstance(IMC_location_transformed_single, list):
    IMC_location_transformed_single = IMC_location_transformed_single[0]

with open(IMC_location_transformed, "r") as f:
    geojson_out_dict = json.load(f)

core_name = IMC_location_transformed_single.split("_")[-1].split(".")[0]
logging.info(f"Writing transformed IMC location for core {core_name}")
json.dump(
    geojson_out_dict[core_name],
    open(
        IMC_location_transformed_single,
        "w",
    ),
    indent=1,
)
logging.info(f"Finished")