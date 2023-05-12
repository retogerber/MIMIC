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

imwrite(imaggrname, imagg_filt)
