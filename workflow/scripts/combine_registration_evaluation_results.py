import pandas as pd
import re
import json
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

# input_csvs = ['/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_001_reg_metrics.csv','/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_002_reg_metrics.csv']
# input_jsons = ['/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_001_IMS_to_postIMS_reg_metrics.json','/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_002_IMS_to_postIMS_reg_metrics.json']
# input_jsons2 = ['/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_001_error_metrics_postIMC_on_preIMC.json','/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_002_error_metrics_postIMC_on_preIMC.json']
# input_jsons3 = ['/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_001_error_metrics_preIMC_on_preIMS.json','/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_002_error_metrics_preIMC_on_preIMS.json']
# input_jsons4 = ['/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_001_error_metrics_preIMS_on_postIMS.json','/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_002_error_metrics_preIMS_on_postIMS.json']
input_csvs = snakemake.input['registration_metrics']
input_jsons = snakemake.input['IMS_to_postIMS_error']
input_jsons2 = snakemake.input['postIMC_to_preIMC_error']
input_jsons3 = snakemake.input['preIMC_to_preIMS_error']
input_jsons4 = snakemake.input['preIMS_to_postIMS_error']
input_jsons5 = snakemake.input['preIMS_to_postIMS_sitk_error']
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


logging.info("Read preIMS_to_postIMS sitk json")
jl = [json.load(open(f, "r")) for f in input_jsons5]
samplenames = [re.sub("_error_metrics_preIMS_on_postIMS_sitk.json$","",os.path.basename(s)) for s in input_jsons5]

logging.info("preIMS_to_postIMS to dataframe")
dfls = []
for i in range(len(samplenames)):
    jl[i]["sample"] = samplenames[i]
    dfls.append(pd.DataFrame(jl[i], index=["sample"]))
dfout14 = pd.concat(dfls).set_index("sample")



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
dfout3 = dfout3.join(dfout14)

with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    logging.info(dfout3)



logging.info("Save csv")
dfout3.to_csv(output_csv)

logging.info("Finished")