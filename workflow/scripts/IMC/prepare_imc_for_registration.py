import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
from tifffile import imread
from tifffile import imwrite
from pandas import read_csv
import numpy as np
from utils import setNThreads, snakeMakeMock
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["IMC_channels_for_aggr"] = []
    snakemake.input["IMC"] = ""
    snakemake.input["IMC_summary_panel"] = ""
    snakemake.output["IMC_aggr"] = ""
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)


# params
channels_to_use = snakemake.params["IMC_channels_for_aggr"]
# inputs
imname = snakemake.input["IMC"]
dfname = snakemake.input["IMC_summary_panel"]
# outputs
imaggrname = snakemake.output["IMC_aggr"]

logging.info("Read csv")
df = read_csv(dfname)
logging.info("Read Image")
im = imread(imname).astype(np.float64)

if channels_to_use == []:
    channels_to_use = df['name'].values
    logging.info("No channels specified, using all channels")
logging.info(f"Channels to use: {channels_to_use}")

logging.info(f"Image Shape: {im.shape}")
logging.info("Subset Image")
chind = [np.where(df["name"] == ch)[0][0] for ch in channels_to_use]
im = im[chind,:,:]
logging.info(f"Image Shape: {im.shape}")

logging.info("Save Image")
imwrite(imaggrname, im)

logging.info("Finished")