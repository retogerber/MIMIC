import pandas as pd
import re
import json
from utils import setNThreads, snakeMakeMock
import sys,os
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

input_csvs = snakemake.input['registration_metrics']
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
output_csv = snakemake.output['registration_metrics_combined']

logging.info("Read IMS_to_postIMS json")
jl = [json.load(open(f, "r")) for f in input_jsons]
samplenames = [re.sub("_IMS_to_postIMS_reg_metrics(_auto){0,1}.json$","",os.path.basename(s)) for s in input_jsons]

logging.info("IMS_to_postIMS to dataframe")
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

logging.info("Read postIMC_to_preIMC json")
jl = [json.load(open(f, "r")) for f in input_jsons2]
samplenames = [re.sub("_error_metrics_postIMC_on_preIMC.json$","",os.path.basename(s)) for s in input_jsons2]

logging.info("postIMC_to_preIMC to dataframe")
dfls = []
for i in range(len(samplenames)):
    jl[i]["sample"] = samplenames[i]
    dfls.append(pd.DataFrame(jl[i], index=["sample"]))
dfout11 = pd.concat(dfls).set_index("sample")

logging.info("Read preIMC_to_preIMS json")
jl = [json.load(open(f, "r")) for f in input_jsons3]
samplenames = [re.sub("_error_metrics_preIMC_on_preIMS.json$","",os.path.basename(s)) for s in input_jsons3]

logging.info("preIMC_to_preIMS to dataframe")
dfls = []
for i in range(len(samplenames)):
    jl[i]["sample"] = samplenames[i]
    dfls.append(pd.DataFrame(jl[i], index=["sample"]))
dfout12 = pd.concat(dfls).set_index("sample")

logging.info("Read preIMS_to_postIMS json")
jl = [json.load(open(f, "r")) for f in input_jsons4]
samplenames = [re.sub("_error_metrics_preIMS_on_postIMS.json$","",os.path.basename(s)) for s in input_jsons4]

logging.info("preIMS_to_postIMS to dataframe")
dfls = []
for i in range(len(samplenames)):
    jl[i]["sample"] = samplenames[i]
    dfls.append(pd.DataFrame(jl[i], index=["sample"]))
dfout13 = pd.concat(dfls).set_index("sample")

logging.info("Read postIMC_to_postIMS json")
jl = [json.load(open(f, "r")) for f in input_jsons41]
samplenames = [re.sub("_error_metrics_postIMC_on_postIMS.json$","",os.path.basename(s)) for s in input_jsons41]

logging.info("postIMC_to_postIMS to dataframe")
dfls = []
for i in range(len(samplenames)):
    jl[i]["sample"] = samplenames[i]
    dfls.append(pd.DataFrame(jl[i], index=["sample"]))
dfout131 = pd.concat(dfls).set_index("sample")



logging.info("Read preIMS_to_postIMS sitk json")
jl = [json.load(open(f, "r")) for f in input_jsons5]
samplenames = [re.sub("_error_metrics_preIMS_on_postIMS_sitk.json$","",os.path.basename(s)) for s in input_jsons5]

logging.info("preIMS_to_postIMS to dataframe")
dfls = []
for i in range(len(samplenames)):
    jl[i]["sample"] = samplenames[i]
    dfls.append(pd.DataFrame(jl[i], index=["sample"]))
dfout14 = pd.concat(dfls).set_index("sample")


logging.info("Read postIMC_to_preIMC region json")
jl = [json.load(open(f, "r")) for f in input_jsons6]
samplenames = [re.sub("_error_metrics_regions_postIMC_on_preIMC.json$","",os.path.basename(s)) for s in input_jsons6]

logging.info("postIMC_to_preIMC region to dataframe")
dfls = []
for i in range(len(samplenames)):
    jl[i]["sample"] = samplenames[i]
    dfls.append(pd.DataFrame(jl[i], index=["sample"]))
dfout15 = pd.concat(dfls).set_index("sample")

logging.info("Read preIMC_to_preIMS region json")
jl = [json.load(open(f, "r")) for f in input_jsons7]
samplenames = [re.sub("_error_metrics_regions_preIMC_on_preIMS.json$","",os.path.basename(s)) for s in input_jsons7]

logging.info("preIMC_to_preIMS region to dataframe")
dfls = []
for i in range(len(samplenames)):
    jl[i]["sample"] = samplenames[i]
    dfls.append(pd.DataFrame(jl[i], index=["sample"]))
dfout16 = pd.concat(dfls).set_index("sample")

logging.info("Read preIMS_to_postIMS region json")
jl = [json.load(open(f, "r")) for f in input_jsons8]
samplenames = [re.sub("_error_metrics_regions_preIMS_on_postIMS.json$","",os.path.basename(s)) for s in input_jsons8]

logging.info("preIMS_to_postIMS region to dataframe")
dfls = []
for i in range(len(samplenames)):
    jl[i]["sample"] = samplenames[i]
    dfls.append(pd.DataFrame(jl[i], index=["sample"]))
dfout17 = pd.concat(dfls).set_index("sample")


logging.info("Read postIMC_to_postIMS region json")
jl = [json.load(open(f, "r")) for f in input_jsons9]
samplenames = [re.sub("_error_metrics_regions_postIMC_on_postIMS.json$","",os.path.basename(s)) for s in input_jsons9]

logging.info("postIMC_to_postIMS region to dataframe")
dfls = []
for i in range(len(samplenames)):
    jl[i]["sample"] = samplenames[i]
    dfls.append(pd.DataFrame(jl[i], index=["sample"]))
dfout18 = pd.concat(dfls).set_index("sample")




logging.info("Read postIMC to postIMS csv")
dfls = []
for f in input_csvs:
    dfls.append(pd.read_csv(f, index_col="sample"))
dfout2 = pd.concat(dfls)
dfout2 = dfout2.reset_index()
dfout2['sample'] = [re.sub("_transformed_on_postIMS.ome.tiff$","",s) for s in dfout2['sample']]
dfout2 = dfout2.set_index("sample")

logging.info("Combine to dataframe")
dfout3 = dfout2.join(dfout1)
dfout3 = dfout3.join(dfout11)
dfout3 = dfout3.join(dfout12)
dfout3 = dfout3.join(dfout13)
dfout3 = dfout3.join(dfout131)
dfout3 = dfout3.join(dfout14)
dfout3 = dfout3.join(dfout15)
dfout3 = dfout3.join(dfout16)
dfout3 = dfout3.join(dfout17)
dfout3 = dfout3.join(dfout18)

with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    logging.info(dfout3)



logging.info("Save csv")
dfout3.to_csv(output_csv)

logging.info("Finished")