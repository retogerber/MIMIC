import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
import h5py
import numpy as np
import pandas as pd
import json
import cv2
import skimage
from utils import setNThreads, snakeMakeMock
from image_utils import get_image_shape
import zarr
import tifffile
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["IMS_pixelsize"] = 10
    snakemake.params["microscopy_pixelsize"] = 0.252
    snakemake.input["imzml_peaks"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow_artificial_tissue/results/LysineCoatedAT/data/IMS/LysineCoatedAT_ioncount_peaks.h5"
    snakemake.input["imzml_coords"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow_artificial_tissue/results/LysineCoatedAT/data/IMS/LysineCoatedAT_huh7-IMSML-coords.h5"
    snakemake.input["IMC_transformed"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow_artificial_tissue/results/LysineCoatedAT/data/IMC/LysineCoatedAT_huh7_aggr_transformed.ome.tiff"
    snakemake.input["IMC_location"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow_artificial_tissue/results/LysineCoatedAT/data/IMC_location/LysineCoatedAT_IMC_mask_on_postIMS_A1.geojson"
    snakemake.output["IMC_mean_on_IMS"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow_artificial_tissue/results/LysineCoatedAT/data/cell_overlap/LysineCoatedAT_huh7_mean_intensity_on_IMS.csv"
    # snakemake.output["IMC_mask_transformed"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMC_mask/Cirrhosis_TMA_5_01262022_004_transformed_on_postIMS_cropped.ome.tiff"
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
ims_spacing = snakemake.params["IMS_pixelsize"]
microscopy_spacing = snakemake.params["microscopy_pixelsize"]
channelnames = snakemake.params["IMC_channels_for_aggr"]
# inputs
imsml_peaks = snakemake.input["imzml_peaks"]
imsml_coords = snakemake.input["imzml_coords"]
IMC_on_postIMS = snakemake.input['IMC_transformed']
IMC_location = snakemake.input['IMC_location']
if isinstance(IMC_location, list):
    IMC_location = IMC_location[0]
dfname = snakemake.input["IMC_summary_panel"]


logging.info("Read csv")
df = pd.read_csv(dfname)
if channelnames == []:
    channelnames = df['name'].values
    logging.info("No channels specified, using all channels")
logging.info(f"Channels to use: {channelnames}")

# outputs
outfile = snakemake.output["IMC_mean_on_IMS"]


logging.info("core bounding box extraction")
IMC_geojson = json.load(open(IMC_location, "r"))
if isinstance(IMC_geojson,list):
    IMC_geojson=IMC_geojson[0]
boundary_points = np.array(IMC_geojson['geometry']['coordinates'])[0,:,:]
xmin=np.min(boundary_points[:,1])
xmax=np.max(boundary_points[:,1])
ymin=np.min(boundary_points[:,0])
ymax=np.max(boundary_points[:,0])

bb1 = [int(xmin),int(ymin),int(xmax),int(ymax)]

m2full_shape = get_image_shape(IMC_on_postIMS)
bb1[0] = bb1[0] if bb1[0]>=0 else 0
bb1[1] = bb1[1] if bb1[1]>=0 else 0
bb1[2] = bb1[2] if bb1[2]<=m2full_shape[1] else m2full_shape[1]
bb1[3] = bb1[3] if bb1[3]<=m2full_shape[2] else m2full_shape[2]
logging.info(f"bounding box mask whole image 1: {bb1}")


logging.info("Read h5 coords file")
with h5py.File(imsml_coords, "r") as f:
    # if in imsmicrolink IMS was the target
    if "xy_micro_physical" in [key for key, val in f.items()]:
        xy_micro_physical = f["xy_micro_physical"][:]

        coords_micro = xy_micro_physical
    # if the microscopy image was the target
    else:
        padded = f["xy_padded"][:]
        
        coords_micro = (padded * ims_spacing)
    
    coords_original = f["xy_original"][:]
    ids = np.arange(coords_original.shape[0])

    data = {
        "coord_0": coords_original[:,0].tolist(),
        "coord_1": coords_original[:,1].tolist(),
        "coord_micro_0": coords_micro[:,0].tolist(),
        "coord_micro_1": coords_micro[:,1].tolist(),
        "id": ids.tolist()
    }

    merged_df = pd.DataFrame(data)

# transform microscopy coords to pixels
coords = np.array(merged_df[['coord_micro_0','coord_micro_1']])
coords=coords/microscopy_spacing
coords = coords.astype(int)
# get peaks as array
ids = np.array(merged_df['id'])

logging.info("Create image")
image_shape = [bb1[3]-bb1[1],bb1[2]-bb1[0]]
coords = coords-bb1[:2][::-1]
# Calculate ranges
stepsize_px =  int(ims_spacing / microscopy_spacing)
stepsize_px_half = int(stepsize_px/2)
x_ranges = np.clip([coords[:,1] - stepsize_px_half, coords[:,1] + stepsize_px_half], 0, image_shape[1]-1)
y_ranges = np.clip([coords[:,0] - stepsize_px_half, coords[:,0] + stepsize_px_half], 0, image_shape[0]-1)
# create empty image
image = np.zeros((image_shape[0],image_shape[1]))
# Fill the image 
for i, (xr, yr) in enumerate(zip(x_ranges.T, y_ranges.T)):
    image[xr[0]:(xr[1]+1), yr[0]:(yr[1]+1)] = ids[i]


logging.info("Downscale image")
wn = int(image.shape[0]*microscopy_spacing)
hn = int(image.shape[1]*microscopy_spacing)
image = cv2.resize(image, (hn,wn), interpolation=cv2.INTER_NEAREST)

logging.info("Read IMC mask image")
bbox = [int(b) for b in bb1]
store = tifffile.imread(IMC_on_postIMS, aszarr=True)
z = zarr.open(store, mode='r')
imcimg = z[0][:,bbox[0]:bbox[2],bbox[1]:bbox[3]]
imcimg_small = np.zeros((imcimg.shape[0],wn,hn))
for i in range(imcimg.shape[0]):
    imcimg_small[i,:,:] = cv2.resize(imcimg[i,:,:], (hn,wn), interpolation=cv2.INTER_NEAREST)

logging.info("Calculate mean intensities")
df = pd.DataFrame({'label': ids})
for i in range(imcimg_small.shape[0]):
    rps = skimage.measure.regionprops_table(image.astype(int), intensity_image=imcimg_small[i,:,:], properties=('label','mean_intensity'))
    tdf = pd.DataFrame({
        'label': rps['label'],
        f'{channelnames[i]}': rps['mean_intensity']
    })
    df = df.merge(tdf, on=["label"], how="left")

logging.info("Write output")
df.to_csv(outfile, index=False)

logging.info("Finished")


