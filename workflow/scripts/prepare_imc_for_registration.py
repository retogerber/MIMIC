from tifffile import imread
from tifffile import imwrite
from pandas import read_csv
import numpy as np
import sys
from scipy.signal import medfilt2d

#imname = sys.argv[1] 
imname = snakemake.input["IMC"]
#dfname = sys.argv[2]
dfname = snakemake.input["IMC_summary_panel"]
#imaggrname = sys.argv[3]
imaggrname = snakemake.output["IMC_aggr"]

#channels_to_use = ["aSMA","E-cadherin","ST6GAL1","Collagen-1","Seg1","Seg2","Seg3"]
#channels_to_use = ["E-cadherin","ST6GAL1","Collagen-1","Seg1","Seg2","Seg3"]
#channels_to_use = ["E-cadherin","ST6GAL1","Collagen-1","HepPar1","Seg1","Seg2","Seg3"]
channels_to_use = snakemake.params["IMC_channels_for_aggr"]

df = read_csv(dfname)
im = imread(imname)


chind = [np.where(df["name"] == ch)[0][0] for ch in channels_to_use]

imsub = im[chind,:,:]
imagg = np.max(imsub,axis=0)
imagg_filt = medfilt2d(imagg, kernel_size=3)

#maxval = np.quantile(imagg_filt,0.99)
#imagg_filt[imagg_filt>maxval] = maxval
#imagg_filt/=maxval

imagg_filt_nonzero=imagg_filt[imagg_filt>0]
maxval = np.quantile(imagg_filt_nonzero,0.9)
minval = np.quantile(imagg_filt_nonzero,0.1)
imagg_filt[imagg_filt>maxval] = maxval
imagg_filt[imagg_filt<minval] = minval
imagg_filt-=minval
imagg_filt/=(maxval-minval)

imagg_filt*=(2**8-1)
imagg_filt=imagg_filt.astype(np.uint8)

imwrite(imaggrname, imagg_filt)
