from tifffile import imread
from tifffile import imwrite
from pandas import read_csv
import numpy as np
from scipy.signal import medfilt2d
from utils import setNThreads, snakeMakeMock
import sys,os
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


logging.info("Process Image")
chind = [np.where(df["name"] == ch)[0][0] for ch in channels_to_use]

imsub = im[chind,:,:]
imsplit=np.split(imsub,len(chind),0)

for i in range(len(chind)):
    imsplit[i]=np.squeeze(imsplit[i])
    imsplit[i] = medfilt2d(imsplit[i], kernel_size=3)
    if np.sum(imsplit[i]>0) == 0:
        continue
    imagg_tmp_nonzero=imsplit[i][imsplit[i]>0]
    maxval = np.quantile(imagg_tmp_nonzero,0.99)
    minval = np.quantile(imagg_tmp_nonzero,0.01)
    imsplit[i][imsplit[i]>maxval] = maxval
    imsplit[i][imsplit[i]<minval] = 0
    imsplit[i]/=maxval
imsub=np.stack(imsplit)
imagg_filt = np.max(imsub,axis=0)

imagg_filt_nonzero=imagg_filt[imagg_filt>0]
maxval = np.quantile(imagg_filt_nonzero,0.99)
minval = np.quantile(imagg_filt_nonzero,0.01)
imagg_filt[imagg_filt>maxval] = maxval
imagg_filt[imagg_filt<minval] = minval
imagg_filt-=minval
imagg_filt/=(maxval-minval)

imagg_filt*=(2**8-1)
imagg_filt=imagg_filt.astype(np.uint8)

logging.info("Save Image")
imwrite(imaggrname, imagg_filt)

logging.info("Finished")