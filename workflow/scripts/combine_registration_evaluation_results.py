import pandas as pd

# input_csvs = ['/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_001_reg_metrics.csv','/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_002_reg_metrics.csv']
input_csvs = snakemake.input['registration_metrics']
output_csv = snakemake.output['registration_metrics_combined']


dfls = []
for f in input_csvs:
    dfls.append(pd.read_csv(f))

dfout = pd.concat(dfls)

dfout.to_csv(output_csv)