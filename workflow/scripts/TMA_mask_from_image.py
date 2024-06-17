from rembg import new_session
import SimpleITK as sitk
import cv2
import numpy as np
from image_utils import get_image_shape, extract_mask, preprocess_mask
from utils import setNThreads, snakeMakeMock
import sys,os
import json
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["microscopy_pixelsize"] = 0.22537
    snakemake.params["mask_type"] = "bbox"
    snakemake.input["postIMS"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow_2/results/HCC_TMA_Panel2/data/postIMS/HCC_TMA_Panel2_postIMS.ome.tiff"
    snakemake.input["TMA_location"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow_2/results/HCC_TMA_Panel2/data/TMA_location/HCC_TMA_Panel2_TMA_mask_on_postIMS.geojson"
    snakemake.output["postIMSmask"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/postIMS/test_combined_postIMS_mask_for_reg.ome.tiff"

    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# parameters
resolution = float(snakemake.params["microscopy_pixelsize"])
mask_type = snakemake.params["mask_type"]
assert mask_type in ["bbox", "segment"]

# inputs
postIMS_file = snakemake.input["postIMS"]
try:
    TMA_location_file = snakemake.input["TMA_location"]
    if isinstance(TMA_location_file,list):
        TMA_location_file = TMA_location_file[0]
    geojson_exists = postIMS_file != TMA_location_file
except:
    TMA_location_file = None
    geojson_exists = False

model_name = "isnet-general-use"
os.environ["OMP_NUM_THREADS"] = str(snakemake.threads)
rembg_session = new_session(model_name)

# outputs
postIMSr_file = snakemake.output["postIMSmask"]


postIMS_shape = np.array(get_image_shape(postIMS_file)[:2]).astype(int)
logging.info(f"Input shape: {postIMS_shape}")

if geojson_exists:
    logging.info("Get bounding box from geojson")
    TMA_geojson = json.load(open(TMA_location_file, "r"))
    bboxls = list()
    for i in range(len(TMA_geojson)):
        boundary_points = np.array(TMA_geojson[i]['geometry']['coordinates'])[0,:,:]
        xmin=int(np.min(boundary_points[:,1]))
        xmax=int(np.max(boundary_points[:,1]))
        ymin=int(np.min(boundary_points[:,0]))
        ymax=int(np.max(boundary_points[:,0]))
        bbox = np.array([xmin,ymin,xmax,ymax])
        bboxls.append(bbox)

    if mask_type == "bbox":
        logging.info("Create mask from bounding box")
        mask = np.zeros(get_image_shape(postIMS_file)[:2], dtype=np.uint8)
        for i in range(len(bboxls)):
            bbn = bboxls[i]
            mask[bbn[0]:bbn[2],bbn[1]:bbn[3]] = 255
    elif mask_type == "segment":
        logging.info("Extract mask from individual bounding boxes")
        mask = np.zeros(get_image_shape(postIMS_file)[:2], dtype=np.uint8)
        for i in range(len(bboxls)):
            bbn = bboxls[i]
            img_mask = extract_mask(postIMS_file, bbn, session=rembg_session, rescale=resolution/4, is_postIMS = False)[0,:,:]
            img_mask = preprocess_mask(img_mask,1)
            wn=bbn[2]-bbn[0]
            hn=bbn[3]-bbn[1]
            img_mask = cv2.resize(img_mask.astype(np.uint8), (hn,wn), interpolation=cv2.INTER_NEAREST)
            mask[bbn[0]:bbn[2],bbn[1]:bbn[3]] = np.max(np.stack([mask[bbn[0]:bbn[2],bbn[1]:bbn[3]],img_mask], axis=0),axis=0)
        mask[mask>0] = 255
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")

elif not geojson_exists and mask_type == "segment":
    logging.info("Remove background")
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
else:
    logging.error("No mask created")
    mask = np.ones(postIMS_shape, dtype=np.uint8)*255
    mask[:,0] = 0
    mask[:,-1] = 0
    mask[0,:] = 0
    mask[-1,:] = 0


assert np.all(mask.shape == postIMS_shape)

logging.info("Save mask")
sitk.WriteImage(sitk.GetImageFromArray(mask), postIMSr_file, useCompression=True, imageIO="TIFFImageIO", compressor="LZW")
