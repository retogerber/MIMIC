import pandas as pd
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
input_csvs = snakemake.input['registration_metrics']
input_jsons = snakemake.input['IMS_to_postIMS_error']
output_csv = snakemake.output['registration_metrics_combined']

logging.info("Read IMS_to_postIMS json")
jl = [json.load(open(f, "r")) for f in input_jsons]
samplenames = [os.path.basename(s).replace("_IMS_to_postIMS_reg_metrics.json","") for s in input_jsons]

logging.info("IMS_to_postIMS to dataframe")
dfls = []
for i in range(len(samplenames)):
    jl[i]["sample"] = samplenames[i]
    dfls.append(pd.DataFrame(jl[i], index=["sample"]))
dfout1 = pd.concat(dfls).set_index("sample")

logging.info("Read postIMC to postIMS csv")
dfls = []
for f in input_csvs:
    dfls.append(pd.read_csv(f, index_col="sample"))
dfout2 = pd.concat(dfls)

logging.info("Combine to dataframe")
dfout2 = dfout2.join(dfout1)


logging.info("Save csv")
dfout2.to_csv(output_csv)

logging.info("Finished")