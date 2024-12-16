import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
import numpy as np
import pandas as pd
import json
import skimage
from utils import setNThreads, snakeMakeMock
from image_utils import get_image_shape, readimage_crop
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["IMS_pixelsize"] = 30
    snakemake.params["microscopy_pixelsize"] = 0.22537
    snakemake.params["IMC_pixelsize"] = 1
    snakemake.params["IMC_channels_for_aggr"] = ['ST6GAL1', 'HepPar1']
    snakemake.input["IMS_transformed"] = "results/test_split_pre/data/IMS/test_split_pre_IMS_on_postIMS.ome.tiff"
    snakemake.input["IMC_transformed"] = "results/test_split_pre/data/IMC/test_split_pre_IMC_aggr_transformed.ome.tiff"
    snakemake.input["IMC_summary_panel"] = "results/test_split_pre/data/IMC_summary_panel/Cirrhosis-TMA-5_New_Detector_002_summary.csv"
    snakemake.input["IMC_location"] = "results/test_split_pre/data/IMC_location/test_split_pre_IMC_mask_on_postIMS_B1.geojson"
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
imc_spacing = snakemake.params["IMC_pixelsize"]
channelnames = snakemake.params["IMC_channels_for_aggr"]
# inputs
IMS_on_postIMS=snakemake.input["IMS_transformed"]
IMC_on_postIMS = snakemake.input['IMC_transformed']
IMC_location = snakemake.input['IMC_location']
if isinstance(IMC_location, list):
    IMC_location = IMC_location[0]
dfname = snakemake.input["IMC_summary_panel"]
# outputs
outfile = snakemake.output["IMC_mean_on_IMS"]

logging.info("Read csv")
df = pd.read_csv(dfname)
if channelnames == []:
    channelnames = df['name'].values
    logging.info("No channels specified, using all channels")
logging.info(f"Channels to use: {channelnames}")


logging.info("core bounding box extraction")
IMC_geojson = json.load(open(IMC_location, "r"))
if isinstance(IMC_geojson,list):
    IMC_geojson=IMC_geojson[0]
boundary_points = np.array(IMC_geojson['geometry']['coordinates'])[0,:,:]/(imc_spacing/microscopy_spacing)
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

IMS_on_postIMS_file = f"IMS_on_postIMS/mz_indices.ome.tiff"
logging.info(f"Read IMS image: {IMS_on_postIMS_file}")
ims_image = readimage_crop(IMS_on_postIMS_file, bb1).astype(np.uint32)
logging.info(f"IMS image shape: {ims_image.shape}")
imc_image = readimage_crop(IMC_on_postIMS, bb1)
logging.info(f"IMC image shape: {imc_image.shape}")

ids = np.unique(ims_image)
logging.info(f"Number of IMS pixels: {len(ids)}")

logging.info("Calculate mean intensities")
df = pd.DataFrame({'label': ids})
for i in range(imc_image.shape[0]):
    rps = skimage.measure.regionprops_table(ims_image.astype(int), intensity_image=imc_image[i,:,:], properties=('label','mean_intensity'))
    tdf = pd.DataFrame({
        'label': rps['label'],
        f'{channelnames[i]}': rps['mean_intensity']
    })
    df = df.merge(tdf, on=["label"], how="left")

logging.info(f"df shape: {df.shape}")
logging.info(f"df columns: {df.columns}")
logging.info(f"df head: {df.head()}")
logging.info(f"max mean intensity: {df.max()}")

logging.info("Write output")
df.to_csv(outfile, index=False)

logging.info("Finished")