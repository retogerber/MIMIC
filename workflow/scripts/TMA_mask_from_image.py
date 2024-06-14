from rembg import new_session
import SimpleITK as sitk
import cv2
import numpy as np
from image_utils import get_image_shape, extract_mask
from utils import setNThreads, snakeMakeMock
import sys,os
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["microscopy_pixelsize"] = 0.22537
    snakemake.input["postIMS"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/postIMS/test_combined_postIMS.ome.tiff"
    snakemake.output["postIMSmask"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/postIMS/test_combined_postIMS_mask_for_reg.ome.tiff"

    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# parameters
resolution = float(snakemake.params["microscopy_pixelsize"])

# inputs
postIMS_file = snakemake.input["postIMS"]

model_name = "isnet-general-use"
os.environ["OMP_NUM_THREADS"] = str(snakemake.threads)
rembg_session = new_session(model_name)

# outputs
postIMSr_file = snakemake.output["postIMSmask"]

logging.info("Remove background")
postIMS_shape = np.array(get_image_shape(postIMS_file)[:2]).astype(int)
mask = extract_mask(postIMS_file,np.array([0,0,postIMS_shape[0],postIMS_shape[1]]), rembg_session, resolution)[0,:,:]

logging.info("Postprocess mask")
kernel_size = int(np.ceil(1/resolution*5))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
_, labels = cv2.connectedComponents(mask)
counts = np.bincount(labels.flatten())
ind = counts>(1000/2)**2*np.pi
lind = np.arange(len(ind))[ind][1:]
mask = np.zeros(mask.shape, dtype=np.uint8)
for i in lind:
    mask[labels==i] = 255

kernel_size = int(np.ceil(1/resolution*20))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

logging.info("Resize mask")
mask = cv2.resize(mask, postIMS_shape[::-1], interpolation=cv2.INTER_NEAREST)

assert np.all(mask.shape == postIMS_shape)

logging.info("Save mask")
sitk.WriteImage(sitk.GetImageFromArray(mask), postIMSr_file, useCompression=True, imageIO="TIFFImageIO", compressor="LZW")
