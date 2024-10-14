import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
import pandas as pd
import re
import json
from utils import setNThreads, snakeMakeMock
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.input['registration_metrics'] = ['/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_001_reg_metrics.csv','/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_002_reg_metrics.csv']
    snakemake.input['IMS_to_postIMS_error'] = ['/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_001_IMS_to_postIMS_reg_metrics.json','/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_002_IMS_to_postIMS_reg_metrics.json']
    snakemake.input['postIMC_to_preIMC_error'] = ['/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_001_error_metrics_postIMC_on_preIMC.json','/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_002_error_metrics_postIMC_on_preIMC.json']
    snakemake.input['preIMC_to_preIMS_error'] = ['/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_001_error_metrics_preIMC_on_preIMS.json','/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_002_error_metrics_preIMC_on_preIMS.json']
    snakemake.input['preIMS_to_postIMS_error'] = ['/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_001_error_metrics_preIMS_on_postIMS.json','/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_002_error_metrics_preIMS_on_postIMS.json']
    snakemake.input['preIMS_to_postIMS_sitk_error'] = ""
    snakemake.input['postIMC_to_preIMC_region_error'] = ""
    snakemake.input['preIMC_to_preIMS_region_error'] = ""
    snakemake.input['preIMS_to_postIMS_region_error'] = ""
    snakemake.input['postIMC_to_postIMS_region_error'] = ""
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

generic_input = snakemake.params['generic_input']

input_csvs = snakemake.input['registration_metrics']
if input_csvs != generic_input:
    logging.info("Read postIMC to postIMS csv")
    dfls = []
    for f in input_csvs:
        dfls.append(pd.read_csv(f, index_col="sample"))
    dfout = pd.concat(dfls)
    dfout = dfout.reset_index()
    dfout['sample'] = [re.sub("_transformed_on_postIMS.ome.tiff$","",s) for s in dfout['sample']]
    dfout = dfout.set_index("sample")
    dfout_exists = True
else:
    dfout_exists = False

input_jsons = snakemake.input['IMS_to_postIMS_error']
input_jsons2 = snakemake.input['postIMC_to_preIMC_error']
input_jsons3 = snakemake.input['preIMC_to_preIMS_error']
input_jsons4 = snakemake.input['preIMS_to_postIMS_error']
input_jsons41 = snakemake.input['postIMC_to_postIMS_error']
input_jsons5 = snakemake.input['preIMS_to_postIMS_sitk_error']
input_jsons6 = snakemake.input['postIMC_to_preIMC_region_error']
input_jsons7 = snakemake.input['preIMC_to_preIMS_region_error']
input_jsons8 = snakemake.input['preIMS_to_postIMS_region_error']
input_jsons9 = snakemake.input['postIMC_to_postIMS_region_error']
input_jsons10 = snakemake.input['postIMC_to_postIMS_global_metrics']
input_jsons11 = snakemake.input['postIMC_to_preIMC_global_metrics']
input_jsons12 = snakemake.input['preIMC_to_preIMS_global_metrics']
input_jsons13 = snakemake.input['preIMS_to_postIMS_global_metrics']
output_csv = snakemake.output['registration_metrics_combined']


def to_pd(input_jsons, reg):
    jl = [json.load(open(f, "r")) for f in input_jsons]
    samplenames = [re.sub(reg,"",os.path.basename(s)) for s in input_jsons]

    logging.info("to dataframe")
    dfls = []
    for i in range(len(samplenames)):
        jl[i]["sample"] = samplenames[i]
        dfls.append(pd.DataFrame(jl[i], index=["sample"]))
    dfout1 = pd.concat(dfls).set_index("sample")

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           ):
        logging.info(dfout1)
    return dfout1

if input_jsons != generic_input:
    logging.info("Read IMS_to_postIMS json")
    temp_dfout = to_pd(input_jsons, "_IMS_to_postIMS_reg_metrics(_auto){0,1}.json$")
    if dfout_exists:
        dfout = dfout.join(temp_dfout)
    else:
        dfout = temp_dfout
        dfout_exists = True
    logging.info(dfout.columns)

if input_jsons2 != generic_input:
    logging.info("Read postIMC_to_preIMC json")
    temp_dfout = to_pd(input_jsons2, "_error_metrics_postIMC_on_preIMC.json$")
    if dfout_exists:
        dfout = dfout.join(temp_dfout)
    else:
        dfout = temp_dfout
        dfout_exists = True
    logging.info(dfout.columns)

if input_jsons3 != generic_input:
    logging.info("Read preIMC_to_preIMS json")
    temp_dfout = to_pd(input_jsons3, "_error_metrics_preIMC_on_preIMS.json$")
    if dfout_exists:
        dfout = dfout.join(temp_dfout)
    else:
        dfout = temp_dfout
        dfout_exists = True
    logging.info(dfout.columns)

if input_jsons4 != generic_input:
    logging.info("Read preIMS_to_postIMS json")
    temp_dfout = to_pd(input_jsons4, "_error_metrics_preIMS_on_postIMS.json$")
    if dfout_exists:
        dfout = dfout.join(temp_dfout)
    else:
        dfout = temp_dfout
        dfout_exists = True
    logging.info(dfout.columns)

if input_jsons41 != generic_input:
    logging.info("Read postIMC_to_postIMS json")
    temp_dfout = to_pd(input_jsons41, "_error_metrics_postIMC_on_postIMS.json$")
    if dfout_exists:
        dfout = dfout.join(temp_dfout)
    else:
        dfout = temp_dfout
        dfout_exists = True
    logging.info(dfout.columns)

if input_jsons5 != generic_input:
    logging.info("Read preIMS_to_postIMS sitk json")
    temp_dfout = to_pd(input_jsons5, "_error_metrics_preIMS_on_postIMS_sitk.json$")
    if dfout_exists:
        dfout = dfout.join(temp_dfout)
    else:
        dfout = temp_dfout
        dfout_exists = True
    logging.info(dfout.columns)

if input_jsons6 != generic_input:
    logging.info("Read postIMC_to_preIMC region json")
    temp_dfout = to_pd(input_jsons6, "_error_metrics_regions_postIMC_on_preIMC.json$")
    if dfout_exists:
        dfout = dfout.join(temp_dfout)
    else:
        dfout = temp_dfout
        dfout_exists = True
    logging.info(dfout.columns)

if input_jsons7 != generic_input:
    logging.info("Read preIMC_to_preIMS region json")
    temp_dfout = to_pd(input_jsons7, "_error_metrics_regions_preIMC_on_preIMS.json$")
    if dfout_exists:
        dfout = dfout.join(temp_dfout)
    else:
        dfout = temp_dfout
        dfout_exists = True
    logging.info(dfout.columns)

if input_jsons8 != generic_input:
    logging.info("Read preIMS_to_postIMS region json")
    temp_dfout = to_pd(input_jsons8, "_error_metrics_regions_preIMS_on_postIMS.json$")
    if dfout_exists:
        dfout = dfout.join(temp_dfout)
    else:
        dfout = temp_dfout
        dfout_exists = True
    logging.info(dfout.columns)

if input_jsons9 != generic_input:
    logging.info("Read postIMC_to_postIMS region json")
    temp_dfout = to_pd(input_jsons9, "_error_metrics_regions_postIMC_on_postIMS.json$")
    if dfout_exists:
        dfout = dfout.join(temp_dfout)
    else:
        dfout = temp_dfout
        dfout_exists = True
    logging.info(dfout.columns)

if input_jsons10 != generic_input:
    logging.info("Read postIMC_to_postIMS global metrics json")
    temp_dfout = to_pd(input_jsons10, "_global_error_metrics_postIMC_on_postIMS.json$")
    if dfout_exists:
        dfout = dfout.join(temp_dfout)
    else:
        dfout = temp_dfout
        dfout_exists = True
    logging.info(dfout.columns)

if input_jsons11 != generic_input:
    logging.info("Read postIMC_to_preIMC global metrics json")
    temp_dfout = to_pd(input_jsons11, "_global_error_metrics_postIMC_on_preIMC.json$")
    if dfout_exists:
        dfout = dfout.join(temp_dfout)
    else:
        dfout = temp_dfout
        dfout_exists = True
    logging.info(dfout.columns)

if input_jsons12 != generic_input:
    logging.info("Read preIMC_to_preIMS global metrics json")
    temp_dfout = to_pd(input_jsons12, "_global_error_metrics_preIMC_on_preIMS.json$")
    if dfout_exists:
        dfout = dfout.join(temp_dfout)
    else:
        dfout = temp_dfout
        dfout_exists = True
    logging.info(dfout.columns)

if input_jsons13 != generic_input:
    logging.info("Read preIMS_to_postIMS global metrics json")
    temp_dfout = to_pd(input_jsons13, "_global_error_metrics_preIMS_on_postIMS.json$")
    if dfout_exists:
        dfout = dfout.join(temp_dfout)
    else:
        dfout = temp_dfout
        dfout_exists = True
    logging.info(dfout.columns)


with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    logging.info(dfout)



logging.info("Save csv")
dfout.to_csv(output_csv)

logging.info("Finished")