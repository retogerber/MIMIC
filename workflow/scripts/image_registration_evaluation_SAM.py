import pandas as pd
import json
import cv2
import matplotlib.pyplot as plt
from rembg import new_session
from segment_anything import sam_model_registry
import skimage
import numpy as np
import tifffile
from image_utils import extract_mask
from registration_utils import get_max_dice_score, dist_centroids
from utils import setNThreads, snakeMakeMock
import sys,os
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["input_spacing_1"] = 1
    snakemake.params["input_spacing_2"] = 0.22537
    snakemake.params["output_spacing"] = 1 
    # snakemake.input["postIMC_on_preIMC"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/Cirrhosis-TMA-5_New_Detector_001_transformed_on_preIMC.ome.tiff"
    snakemake.input["postIMC_on_preIMC"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/Lipid_TMA_3781/data/postIMC/Lipid_TMA_37819_033_transformed_on_preIMC.ome.tiff"
    # snakemake.input['preIMC'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMC/test_split_pre_preIMC.ome.tiff"
    snakemake.input['preIMC'] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/Lipid_TMA_3781/data/preIMC/Lipid_TMA_3781_preIMC.ome.tiff"
    snakemake.input["IMC_on_preIMC"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/Lipid_TMA_3781/data/IMC_location/Lipid_TMA_3781_IMC_mask_on_preIMC_B6.geojson"
    # snakemake.input['preIMC_on_preIMS'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMC/Cirrhosis-TMA-5_New_Detector_001_transformed_on_preIMS.ome.tiff"
    snakemake.input['preIMC_on_preIMS'] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/Lipid_TMA_3781/data/preIMC/Lipid_TMA_37819_033_transformed_on_preIMS.ome.tiff"
    # snakemake.input['preIMS'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMS/test_split_pre_preIMS.ome.tiff"
    snakemake.input['preIMS'] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/Lipid_TMA_3781/data/preIMS/Lipid_TMA_3781_preIMS.ome.tiff"
    snakemake.input["IMC_on_preIMS"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/Lipid_TMA_3781/data/IMC_location/Lipid_TMA_3781_IMC_mask_on_preIMS_B6.geojson"
    # snakemake.input['preIMS_on_postIMS'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMS/Cirrhosis-TMA-5_New_Detector_002_transformed_on_postIMS.ome.tiff"
    snakemake.input['preIMS_on_postIMS'] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/Lipid_TMA_3781/data/preIMS/Lipid_TMA_37819_033_transformed_on_postIMS.ome.tiff"
    # snakemake.input['postIMS'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMS/test_split_pre_postIMS.ome.tiff"
    snakemake.input['postIMS'] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/Lipid_TMA_3781/data/postIMS/Lipid_TMA_3781_postIMS.ome.tiff"
    snakemake.input['postIMC_on_postIMS'] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/Lipid_TMA_3781/data/postIMC/Lipid_TMA_37819_033_transformed_on_postIMS.ome.tiff"
    snakemake.input["IMC_on_postIMS"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/Lipid_TMA_3781/data/IMC_location/Lipid_TMA_3781_IMC_mask_on_postIMS_B6.geojson"
    snakemake.input['sam_weights'] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/Misc/sam_vit_h_4b8939.pth"
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
input_spacing_1 = snakemake.params["input_spacing_1"]
input_spacing_2 = snakemake.params["input_spacing_2"]
output_spacing = snakemake.params["output_spacing"]
# inputs
postIMC_on_preIMC_file = snakemake.input["postIMC_on_preIMC"]
preIMC_file = snakemake.input["preIMC"]
IMC_on_preIMC_location = snakemake.input["IMC_on_preIMC"]
if isinstance(IMC_on_preIMC_location, list):
    IMC_on_preIMC_location = IMC_on_preIMC_location[0]
logging.info(f"IMC_on_preIMC_location: {IMC_on_preIMC_location}")
preIMC_on_preIMS_file = snakemake.input["preIMC_on_preIMS"]
preIMS_file = snakemake.input["preIMS"]
IMC_on_preIMS_location = snakemake.input["IMC_on_preIMS"]
if isinstance(IMC_on_preIMS_location, list):
    IMC_on_preIMS_location = IMC_on_preIMS_location[0]
logging.info(f"IMC_on_preIMS_location: {IMC_on_preIMS_location}")
preIMS_on_postIMS_file = snakemake.input["preIMS_on_postIMS"]
postIMS_file = snakemake.input["postIMS"]
postIMC_on_postIMS_file = snakemake.input["postIMC_on_postIMS"]
IMC_on_postIMS_location = snakemake.input["IMC_on_postIMS"]
if isinstance(IMC_on_postIMS_location, list):
    IMC_on_postIMS_location = IMC_on_postIMS_location[0]
logging.info(f"IMC_on_postIMS_location: {IMC_on_postIMS_location}")
CHECKPOINT_PATH = snakemake.input["sam_weights"]
# outputs
output_df = snakemake.output["registration_metrics"]

logging.info("Setup rembg model")
# prepare model for rembg
model_name = "isnet-general-use"
os.environ["OMP_NUM_THREADS"] = str(snakemake.threads)
rembg_session = new_session(model_name)

logging.info("Setup segment anything model")
DEVICE = 'cpu'
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)






logging.info("postIMC on preIMC bounding box extraction")
postIMC_on_preIMC = skimage.io.imread(postIMC_on_preIMC_file)
rp = skimage.measure.regionprops((postIMC_on_preIMC[:,:,0]>0).astype(np.uint8))
bb1 = rp[0].bbox
bb2 = [int(bb1[0]/input_spacing_2),int(bb1[1]/input_spacing_2),int(bb1[2]/input_spacing_2),int(bb1[3]/input_spacing_2)] 
logging.info(f"\tBB1: {bb1}")
logging.info(f"\tBB2: {bb2}")

logging.info("mask extraction")
postIMC_on_preIMCmasks = extract_mask(postIMC_on_preIMC_file, bb1, rembg_session, input_spacing_1/output_spacing, multiple_rembgth=True)
preIMCmasks = extract_mask(preIMC_file, bb2, rembg_session, input_spacing_2/output_spacing, multiple_rembgth=True)
s1 = postIMC_on_preIMCmasks.shape[1] if postIMC_on_preIMCmasks.shape[1] <= preIMCmasks.shape[1] else preIMCmasks.shape[1]
s2 = postIMC_on_preIMCmasks.shape[2] if postIMC_on_preIMCmasks.shape[2] <= preIMCmasks.shape[2] else preIMCmasks.shape[2]
logging.info(f"\tMask crop: {(s1,s2)}")
postIMC_on_preIMCmasks = postIMC_on_preIMCmasks[:,:s1,:s2]
preIMCmasks = preIMCmasks[:,:s1,:s2]
    
logging.info("IMC area calculation")
IMC_geojson = json.load(open(IMC_on_preIMC_location, "r"))
if isinstance(IMC_geojson,list):
    IMC_geojson=IMC_geojson[0]
boundary_points = (np.flip(np.array(IMC_geojson['geometry']['coordinates'])[0,:,:])-bb2[:2])*input_spacing_2/output_spacing
IMC_area = cv2.contourArea(boundary_points.reshape(-1,1,2).astype(np.float32))

postIMC_on_preIMCmasks = np.array([msk for msk in postIMC_on_preIMCmasks if np.sum(msk)<np.prod(postIMC_on_preIMCmasks.shape[1:])*0.9])
postIMC_on_preIMCmasks = np.array([msk for msk in postIMC_on_preIMCmasks if np.sum(msk)/IMC_area > 1.05])
preIMCmasks = np.array([msk for msk in preIMCmasks if np.sum(msk)<np.prod(preIMCmasks.shape[1:])*0.9])
preIMCmasks = np.array([msk for msk in preIMCmasks if np.sum(msk)/IMC_area > 1.05])

logging.info("score calculation")
if len(postIMC_on_preIMCmasks) == 0 or len(preIMCmasks) == 0:
    logging.info("No masks found!")
    dice_score_postIMC_on_preIMC_to_preIMC = 0
    overlap_area=0
else:
    postIMC_on_preIMC_to_preIMC, preIMC_to_postIMC_on_preIMC, dice_score_postIMC_on_preIMC_to_preIMC = get_max_dice_score(postIMC_on_preIMCmasks, preIMCmasks)
    overlap_area = np.sum(np.logical_and(postIMC_on_preIMC_to_preIMC, preIMC_to_postIMC_on_preIMC))

logging.info(f"Dice score: {dice_score_postIMC_on_preIMC_to_preIMC}")
logging.info(f"Overlap area: {overlap_area}")
logging.info(f"IMC area: {IMC_area}")
logging.info(f"Overlap area / IMC area: {overlap_area/IMC_area}")
if dice_score_postIMC_on_preIMC_to_preIMC < 0.95 or overlap_area/IMC_area < 0.98:
    logging.info("use SAM for mask extraction")
    logging.info("mask extraction")
    sample_pts = boundary_points[:-1].astype(int)
    for q1 in [0.25,0.5,0.75]:
        for q2 in [0.25,0.5,0.75]:
            sample_pts = np.vstack([sample_pts,np.array([
                np.min(boundary_points[:,0]) + q1*(np.max(boundary_points[:,0])-np.min(boundary_points[:,0])),
                np.min(boundary_points[:,1]) + q2*(np.max(boundary_points[:,1])-np.min(boundary_points[:,1]))
            ]).reshape(1,2).astype(int)])
    logging.info(f"\tSample points: {sample_pts}")

    postIMC_on_preIMCmasks_tmp = extract_mask(postIMC_on_preIMC_file, bb1, rembg_session, input_spacing_1/output_spacing, sam=sam, pts=sample_pts)
    postIMC_on_preIMCmasks = np.array([msk for msk in postIMC_on_preIMCmasks_tmp if np.sum(msk)/IMC_area > 1.05])
    if len(postIMC_on_preIMCmasks) > 0:
        postIMC_on_preIMCmasks = np.array([msk for msk in postIMC_on_preIMCmasks if np.sum(msk)<np.prod(postIMC_on_preIMCmasks_tmp.shape[1:])*0.9])
    if len(postIMC_on_preIMCmasks) > 0:
        tb1 = np.array([np.sum(np.array([ postIMC_on_preIMCmasks[i,x,y] for x,y in sample_pts ])) for i in range(postIMC_on_preIMCmasks.shape[0])])
        postIMC_on_preIMCmasks = postIMC_on_preIMCmasks[tb1==np.max(tb1)]

    preIMCmasks_tmp = extract_mask(preIMC_file, bb2, rembg_session, input_spacing_2/output_spacing, sam=sam, pts=sample_pts)
    preIMCmasks = np.array([msk for msk in preIMCmasks_tmp if np.sum(msk)/IMC_area > 1.05])
    if len(preIMCmasks) > 0:
        preIMCmasks = np.array([msk for msk in preIMCmasks if np.sum(msk)<np.prod(preIMCmasks_tmp.shape[1:])*0.9])
    if len(preIMCmasks) > 0:
        tb1 = np.array([np.sum(np.array([ preIMCmasks[i,x,y] for x,y in sample_pts ])) for i in range(preIMCmasks.shape[0])])
        preIMCmasks = preIMCmasks[tb1==np.max(tb1)]

    if len(postIMC_on_preIMCmasks) == 0 or len(preIMCmasks) == 0:
        logging.info("No masks found!")
        postIMC_on_preIMC_to_preIMC = np.zeros(postIMC_on_preIMCmasks_tmp[0].shape+tuple([1]))
        preIMC_to_postIMC_on_preIMC = np.zeros(preIMCmasks_tmp[0].shape+tuple([1]))
        dice_score_postIMC_on_preIMC_to_preIMC = 0
    else:
        s1 = postIMC_on_preIMCmasks.shape[1] if postIMC_on_preIMCmasks.shape[1] <= preIMCmasks.shape[1] else preIMCmasks.shape[1]
        s2 = postIMC_on_preIMCmasks.shape[2] if postIMC_on_preIMCmasks.shape[2] <= preIMCmasks.shape[2] else preIMCmasks.shape[2]
        logging.info(f"\tMask crop: {(s1,s2)}")
        postIMC_on_preIMCmasks = postIMC_on_preIMCmasks[:,:s1,:s2]
        preIMCmasks = preIMCmasks[:,:s1,:s2]

        logging.info("score calculation")
        postIMC_on_preIMC_to_preIMC, preIMC_to_postIMC_on_preIMC, dice_score_postIMC_on_preIMC_to_preIMC = get_max_dice_score(postIMC_on_preIMCmasks, preIMCmasks)
        logging.info(f"Dice score: {dice_score_postIMC_on_preIMC_to_preIMC}")


postIMC_on_preIMC_area = np.sum(postIMC_on_preIMC_to_preIMC)/(output_spacing**2)
preIMC_area = np.sum(preIMC_to_postIMC_on_preIMC)/(output_spacing**2)

reg1 = skimage.measure.regionprops(postIMC_on_preIMC_to_preIMC.astype(np.uint8)[:,:,0])
if len(reg1) == 0:
    postIMC_on_preIMC_centroid = [np.nan,np.nan]
else:
    postIMC_on_preIMC_centroid = reg1[0].centroid
reg2 = skimage.measure.regionprops(preIMC_to_postIMC_on_preIMC.astype(np.uint8)[:,:,0])
if len(reg2) == 0:
    preIMC_centroid = [np.nan,np.nan]
else:
    preIMC_centroid = reg2[0].centroid

postIMC_on_preIMC_to_preIMC_dist = dist_centroids(postIMC_on_preIMC_centroid, preIMC_centroid, 1/output_spacing)








logging.info("preIMC on preIMS bounding box extraction")
preIMC_on_preIMS = skimage.io.imread(preIMC_on_preIMS_file)
rp = skimage.measure.regionprops((preIMC_on_preIMS[:,:,0]>0).astype(np.uint8))
bb1 = rp[0].bbox
bb2 = [int(bb1[0]/input_spacing_2),int(bb1[1]/input_spacing_2),int(bb1[2]/input_spacing_2),int(bb1[3]/input_spacing_2)] 
logging.info(f"\tBB1: {bb1}")
logging.info(f"\tBB2: {bb2}")

logging.info("mask extraction")
preIMC_on_preIMSmasks = extract_mask(preIMC_on_preIMS_file, bb1, rembg_session, input_spacing_1/output_spacing, multiple_rembgth=True)
preIMSmasks = extract_mask(preIMS_file, bb2, rembg_session, input_spacing_2/output_spacing, multiple_rembgth=True)
s1 = preIMC_on_preIMSmasks.shape[1] if preIMC_on_preIMSmasks.shape[1] <= preIMSmasks.shape[1] else preIMSmasks.shape[1]
s2 = preIMC_on_preIMSmasks.shape[2] if preIMC_on_preIMSmasks.shape[2] <= preIMSmasks.shape[2] else preIMSmasks.shape[2]
logging.info(f"\tMask crop: {(s1,s2)}")
preIMC_on_preIMSmasks = preIMC_on_preIMSmasks[:,:s1,:s2]
preIMSmasks = preIMSmasks[:,:s1,:s2]

logging.info("IMC area calculation")
IMC_geojson = json.load(open(IMC_on_preIMS_location, "r"))
if isinstance(IMC_geojson,list):
    IMC_geojson=IMC_geojson[0]
boundary_points = (np.flip(np.array(IMC_geojson['geometry']['coordinates'])[0,:,:])-bb2[:2])*input_spacing_2/output_spacing
IMC_area = cv2.contourArea(boundary_points.reshape(-1,1,2).astype(np.float32))

preIMC_on_preIMSmasks = np.array([msk for msk in preIMC_on_preIMSmasks if np.sum(msk)<np.prod(preIMC_on_preIMSmasks.shape[1:])*0.9])
preIMC_on_preIMSmasks = np.array([msk for msk in preIMC_on_preIMSmasks if np.sum(msk)/IMC_area > 1.05])
preIMSmasks = np.array([msk for msk in preIMSmasks if np.sum(msk)<np.prod(preIMSmasks.shape[1:])*0.9])
preIMSmasks = np.array([msk for msk in preIMSmasks if np.sum(msk)/IMC_area > 1.05])

logging.info("score calculation")
if len(preIMC_on_preIMSmasks) == 0 or len(preIMSmasks) == 0:
    logging.info("No masks found!")
    dice_score_preIMC_on_preIMS_to_preIMS = 0
    overlap_area=0
else:
    preIMC_on_preIMS_to_preIMS, preIMS_to_preIMC_on_preIMS, dice_score_preIMC_on_preIMS_to_preIMS = get_max_dice_score(preIMC_on_preIMSmasks, preIMSmasks)
    overlap_area = np.sum(np.logical_and(preIMC_on_preIMS_to_preIMS, preIMS_to_preIMC_on_preIMS))

logging.info(f"Dice score: {dice_score_preIMC_on_preIMS_to_preIMS}")
logging.info(f"Overlap area: {overlap_area}")
logging.info(f"IMC area: {IMC_area}")
logging.info(f"Overlap area / IMC area: {overlap_area/IMC_area}")
if dice_score_preIMC_on_preIMS_to_preIMS < 0.95 or overlap_area/IMC_area < 0.98:
    logging.info("use SAM for mask extraction")
    logging.info("mask extraction")
    sample_pts = boundary_points[:-1].astype(int)
    for q1 in [0.25,0.5,0.75]:
        for q2 in [0.25,0.5,0.75]:
            sample_pts = np.vstack([sample_pts,np.array([
                np.min(boundary_points[:,0]) + q1*(np.max(boundary_points[:,0])-np.min(boundary_points[:,0])),
                np.min(boundary_points[:,1]) + q2*(np.max(boundary_points[:,1])-np.min(boundary_points[:,1]))
            ]).reshape(1,2).astype(int)])
    logging.info(f"\tSample points: {sample_pts}")

    preIMC_on_preIMSmasks_tmp = extract_mask(preIMC_on_preIMS_file, bb1, rembg_session, input_spacing_1/output_spacing, sam=sam, pts=boundary_points)
    preIMC_on_preIMSmasks = np.array([msk for msk in preIMC_on_preIMSmasks_tmp if np.sum(msk)/IMC_area > 1.05])
    if len(preIMC_on_preIMSmasks) > 0:
        preIMC_on_preIMSmasks = np.array([msk for msk in preIMC_on_preIMSmasks if np.sum(msk)<np.prod(preIMC_on_preIMSmasks_tmp.shape[1:])*0.9])
    if len(preIMC_on_preIMSmasks) > 0:
        tb1 = np.array([np.sum(np.array([ preIMC_on_preIMSmasks[i,x,y] for x,y in sample_pts ])) for i in range(preIMC_on_preIMSmasks.shape[0])])
        preIMC_on_preIMSmasks = preIMC_on_preIMSmasks[tb1==np.max(tb1)]

    preIMSmasks_tmp = extract_mask(preIMS_file, bb2, rembg_session, input_spacing_2/output_spacing, sam=sam, pts=boundary_points)
    preIMSmasks = np.array([msk for msk in preIMSmasks_tmp if np.sum(msk)/IMC_area > 1.05])
    if len(preIMSmasks) > 0:
        preIMSmasks = np.array([msk for msk in preIMSmasks if np.sum(msk)<np.prod(preIMSmasks_tmp.shape[1:])*0.9])
    if len(preIMSmasks) > 0:
        tb1 = np.array([np.sum(np.array([ preIMSmasks[i,x,y] for x,y in sample_pts ])) for i in range(preIMSmasks.shape[0])])
        preIMSmasks = preIMSmasks[tb1==np.max(tb1)]

    if len(preIMC_on_preIMSmasks) == 0 or len(preIMSmasks) == 0:
        logging.info("No masks found!")
        preIMC_on_preIMS_to_preIMS = np.zeros(preIMC_on_preIMSmasks_tmp[0].shape+tuple([1]))
        preIMS_to_preIMC_on_preIMS = np.zeros(preIMSmasks_tmp[0].shape+tuple([1]))
        dice_score_preIMC_on_preIMS_to_preIMS = 0
    else:
        s1 = preIMC_on_preIMSmasks.shape[1] if preIMC_on_preIMSmasks.shape[1] <= preIMSmasks.shape[1] else preIMSmasks.shape[1]
        s2 = preIMC_on_preIMSmasks.shape[2] if preIMC_on_preIMSmasks.shape[2] <= preIMSmasks.shape[2] else preIMSmasks.shape[2]
        logging.info(f"\tMask crop: {(s1,s2)}")
        preIMC_on_preIMSmasks = preIMC_on_preIMSmasks[:,:s1,:s2]
        preIMSmasks = preIMSmasks[:,:s1,:s2]

        logging.info("score calculation")
        preIMC_on_preIMS_to_preIMS, preIMS_to_preIMC_on_preIMS, dice_score_preIMC_on_preIMS_to_preIMS = get_max_dice_score(preIMC_on_preIMSmasks, preIMSmasks)
        logging.info(f"Dice score: {dice_score_preIMC_on_preIMS_to_preIMS}")

preIMC_on_preIMS_area = np.sum(preIMC_on_preIMS_to_preIMS)/(output_spacing**2)
preIMS_area = np.sum(preIMS_to_preIMC_on_preIMS)/(output_spacing**2)

reg1 = skimage.measure.regionprops(preIMC_on_preIMS_to_preIMS.astype(np.uint8)[:,:,0])
if len(reg1) == 0:
    preIMC_on_preIMS_centroid = [np.nan,np.nan]
else:
    preIMC_on_preIMS_centroid = reg1[0].centroid
reg2 = skimage.measure.regionprops(preIMS_to_preIMC_on_preIMS.astype(np.uint8)[:,:,0])
if len(reg2) == 0:
    preIMS_centroid = [np.nan,np.nan]
else:
    preIMS_centroid = reg2[0].centroid

preIMC_on_preIMS_to_preIMS_dist = dist_centroids(preIMC_on_preIMS_centroid, preIMS_centroid, 1/output_spacing)









logging.info("preIMS on postIMS bounding box extraction")
preIMS_on_postIMS = skimage.io.imread(preIMS_on_postIMS_file)
rp = skimage.measure.regionprops((preIMS_on_postIMS[:,:,0]>0).astype(np.uint8))
bb1 = rp[0].bbox
bb2 = [int(bb1[0]/input_spacing_2),int(bb1[1]/input_spacing_2),int(bb1[2]/input_spacing_2),int(bb1[3]/input_spacing_2)] 
logging.info(f"\tBB1: {bb1}")
logging.info(f"\tBB2: {bb2}")

logging.info("mask extraction")
preIMS_on_postIMSmasks = extract_mask(preIMS_on_postIMS_file, bb1, rembg_session, input_spacing_1/output_spacing, multiple_rembgth=True)
postIMSmasks = extract_mask(postIMS_file, bb2, rembg_session, input_spacing_2/output_spacing, is_postIMS=True, multiple_rembgth=True)
s1 = preIMS_on_postIMSmasks.shape[1] if preIMS_on_postIMSmasks.shape[1] <= postIMSmasks.shape[1] else postIMSmasks.shape[1]
s2 = preIMS_on_postIMSmasks.shape[2] if preIMS_on_postIMSmasks.shape[2] <= postIMSmasks.shape[2] else postIMSmasks.shape[2]
logging.info(f"\tMask crop: {(s1,s2)}")
preIMS_on_postIMSmasks = preIMS_on_postIMSmasks[:,:s1,:s2]
postIMSmasks = postIMSmasks[:,:s1,:s2]

logging.info("IMC area calculation")
IMC_geojson = json.load(open(IMC_on_postIMS_location, "r"))
if isinstance(IMC_geojson,list):
    IMC_geojson=IMC_geojson[0]
boundary_points = (np.flip(np.array(IMC_geojson['geometry']['coordinates'])[0,:,:])-bb2[:2])*input_spacing_2/output_spacing
IMC_area = cv2.contourArea(boundary_points.reshape(-1,1,2).astype(np.float32))

preIMS_on_postIMSmasks = np.array([msk for msk in preIMS_on_postIMSmasks if np.sum(msk)<np.prod(preIMS_on_postIMSmasks.shape[1:])*0.9])
preIMS_on_postIMSmasks = np.array([msk for msk in preIMS_on_postIMSmasks if np.sum(msk)/IMC_area > 1.05])
postIMSmasks = np.array([msk for msk in postIMSmasks if np.sum(msk)<np.prod(postIMSmasks.shape[1:])*0.9])
postIMSmasks = np.array([msk for msk in postIMSmasks if np.sum(msk)/IMC_area > 1.05])


logging.info("score calculation")
if len(preIMS_on_postIMSmasks) == 0 or len(postIMSmasks) == 0:
    logging.info("No masks found!")
    dice_score_preIMS_on_postIMS_to_postIMS = 0
    overlap_area=0
else:
    preIMS_on_postIMS_to_postIMS, postIMS_to_preIMS_on_postIMS, dice_score_preIMS_on_postIMS_to_postIMS = get_max_dice_score(preIMS_on_postIMSmasks, postIMSmasks)
    overlap_area = np.sum(np.logical_and(preIMS_on_postIMS_to_postIMS, postIMS_to_preIMS_on_postIMS))

logging.info(f"Dice score: {dice_score_preIMS_on_postIMS_to_postIMS}")
logging.info(f"Overlap area: {overlap_area}")
logging.info(f"IMC area: {IMC_area}")
logging.info(f"Overlap area / IMC area: {overlap_area/IMC_area}")
if dice_score_preIMS_on_postIMS_to_postIMS < 0.95 or overlap_area/IMC_area < 0.98:
    logging.info("use SAM for mask extraction")
    logging.info("mask extraction")
    sample_pts = boundary_points[:-1].astype(int)
    for q1 in [0.25,0.5,0.75]:
        for q2 in [0.25,0.5,0.75]:
            sample_pts = np.vstack([sample_pts,np.array([
                np.min(boundary_points[:,0]) + q1*(np.max(boundary_points[:,0])-np.min(boundary_points[:,0])),
                np.min(boundary_points[:,1]) + q2*(np.max(boundary_points[:,1])-np.min(boundary_points[:,1]))
            ]).reshape(1,2).astype(int)])
    logging.info(f"\tSample points: {sample_pts}")

    preIMS_on_postIMSmasks_tmp = extract_mask(preIMS_on_postIMS_file, bb1, rembg_session, input_spacing_1/output_spacing, sam=sam, pts=boundary_points)
    preIMS_on_postIMSmasks = np.array([msk for msk in preIMS_on_postIMSmasks_tmp if np.sum(msk)/IMC_area > 1.05])
    if len(preIMS_on_postIMSmasks) > 0:
        preIMS_on_postIMSmasks = np.array([msk for msk in preIMS_on_postIMSmasks if np.sum(msk)<np.prod(preIMS_on_postIMSmasks_tmp.shape[1:])*0.9])
    if len(preIMS_on_postIMSmasks) > 0:
        tb1 = np.array([np.sum(np.array([ preIMS_on_postIMSmasks[i,x,y] for x,y in sample_pts ])) for i in range(preIMS_on_postIMSmasks.shape[0])])
        preIMS_on_postIMSmasks = preIMS_on_postIMSmasks[tb1==np.max(tb1)]

    postIMSmasks_tmp = extract_mask(postIMS_file, bb2, rembg_session, input_spacing_2/output_spacing, is_postIMS=True, sam=sam, pts=boundary_points)
    postIMSmasks = np.array([msk for msk in postIMSmasks_tmp if np.sum(msk)/IMC_area > 1.05])
    if len(postIMSmasks) > 0:
        postIMSmasks = np.array([msk for msk in postIMSmasks if np.sum(msk)<np.prod(postIMSmasks_tmp.shape[1:])*0.9])
    if len(postIMSmasks) > 0:
        tb1 = np.array([np.sum(np.array([ postIMSmasks[i,x,y] for x,y in sample_pts ])) for i in range(postIMSmasks.shape[0])])
        postIMSmasks = postIMSmasks[tb1==np.max(tb1)]

    if len(preIMS_on_postIMSmasks) == 0 or len(postIMSmasks) == 0:
        logging.info("No masks found!")
        preIMS_on_postIMS_to_postIMS = np.zeros(preIMS_on_postIMSmasks_tmp[0].shape+tuple([1]))
        postIMS_to_preIMS_on_postIMS = np.zeros(postIMSmasks_tmp[0].shape+tuple([1]))
        dice_score_preIMS_on_postIMS_to_postIMS = 0
    else:
        s1 = preIMS_on_postIMSmasks.shape[1] if preIMS_on_postIMSmasks.shape[1] <= postIMSmasks.shape[1] else postIMSmasks.shape[1]
        s2 = preIMS_on_postIMSmasks.shape[2] if preIMS_on_postIMSmasks.shape[2] <= postIMSmasks.shape[2] else postIMSmasks.shape[2]
        logging.info(f"\tMask crop: {(s1,s2)}")
        preIMS_on_postIMSmasks = preIMS_on_postIMSmasks[:,:s1,:s2]
        postIMSmasks = postIMSmasks[:,:s1,:s2]

        logging.info("score calculation")
        preIMS_on_postIMS_to_postIMS, postIMS_to_preIMS_on_postIMS, dice_score_preIMS_on_postIMS_to_postIMS = get_max_dice_score(preIMS_on_postIMSmasks, postIMSmasks)
        logging.info(f"Dice score: {dice_score_preIMS_on_postIMS_to_postIMS}")

preIMS_on_postIMS_area = np.sum(preIMS_on_postIMS_to_postIMS)/(output_spacing**2)
postIMS_area = np.sum(postIMS_to_preIMS_on_postIMS)/(output_spacing**2)

reg1 = skimage.measure.regionprops(preIMS_on_postIMS_to_postIMS.astype(np.uint8)[:,:,0])
if len(reg1) == 0:
    preIMS_on_postIMS_centroid = [np.nan,np.nan]
else:
    preIMS_on_postIMS_centroid = reg1[0].centroid
reg2 = skimage.measure.regionprops(postIMS_to_preIMS_on_postIMS.astype(np.uint8)[:,:,0])
if len(reg2) == 0:
    postIMS_centroid = [np.nan,np.nan]
else:
    postIMS_centroid = reg2[0].centroid

preIMS_on_postIMS_to_postIMS_dist = dist_centroids(preIMS_on_postIMS_centroid, postIMS_centroid, 1/output_spacing)







logging.info("postIMC on postIMS bounding box extraction")
postIMC_on_postIMS = skimage.io.imread(postIMC_on_postIMS_file)
rp = skimage.measure.regionprops((postIMC_on_postIMS[:,:,0]>0).astype(np.uint8))
bb1 = rp[0].bbox
bb2 = bb1
logging.info(f"\tBB1: {bb1}")
logging.info(f"\tBB2: {bb2}")

logging.info("mask extraction")
postIMC_on_postIMSmasks = extract_mask(postIMC_on_postIMS_file, bb1, rembg_session, input_spacing_2/output_spacing, multiple_rembgth=True)
postIMS_to_postIMCmasks = extract_mask(postIMS_file, bb2, rembg_session, input_spacing_2/output_spacing, is_postIMS=True, multiple_rembgth=True)
s1 = postIMC_on_postIMSmasks.shape[1] if postIMC_on_postIMSmasks.shape[1] <= postIMS_to_postIMCmasks.shape[1] else postIMS_to_postIMCmasks.shape[1]
s2 = postIMC_on_postIMSmasks.shape[2] if postIMC_on_postIMSmasks.shape[2] <= postIMS_to_postIMCmasks.shape[2] else postIMS_to_postIMCmasks.shape[2]
logging.info(f"\tMask crop: {(s1,s2)}")
postIMC_on_postIMSmasks = postIMC_on_postIMSmasks[:,:s1,:s2]
postIMS_to_postIMCmasks = postIMS_to_postIMCmasks[:,:s1,:s2]

logging.info("IMC area calculation")
IMC_geojson = json.load(open(IMC_on_postIMS_location, "r"))
if isinstance(IMC_geojson,list):
    IMC_geojson=IMC_geojson[0]
boundary_points = (np.flip(np.array(IMC_geojson['geometry']['coordinates'])[0,:,:])-bb2[:2])*input_spacing_2/output_spacing
IMC_area = cv2.contourArea(boundary_points.reshape(-1,1,2).astype(np.float32))

postIMC_on_postIMSmasks = np.array([msk for msk in postIMC_on_postIMSmasks if np.sum(msk)<np.prod(postIMC_on_postIMSmasks.shape[1:])*0.9])
postIMC_on_postIMSmasks = np.array([msk for msk in postIMC_on_postIMSmasks if np.sum(msk)/IMC_area > 1.05])
postIMS_to_postIMCmasks = np.array([msk for msk in postIMS_to_postIMCmasks if np.sum(msk)<np.prod(postIMS_to_postIMCmasks.shape[1:])*0.9])
postIMS_to_postIMCmasks = np.array([msk for msk in postIMS_to_postIMCmasks if np.sum(msk)/IMC_area > 1.05])


logging.info("score calculation")
if len(postIMC_on_postIMSmasks) == 0 or len(postIMS_to_postIMCmasks) == 0:
    logging.info("No masks found!")
    dice_score_postIMC_on_postIMS_to_postIMS = 0
    overlap_area=0
else:
    postIMC_on_postIMS_to_postIMS, postIMS_to_postIMC_on_postIMS, dice_score_postIMC_on_postIMS_to_postIMS = get_max_dice_score(postIMC_on_postIMSmasks, postIMS_to_postIMCmasks)
    overlap_area = np.sum(np.logical_and(postIMC_on_postIMS_to_postIMS, postIMS_to_postIMC_on_postIMS))

logging.info(f"Dice score: {dice_score_postIMC_on_postIMS_to_postIMS}")
logging.info(f"Overlap area: {overlap_area}")
logging.info(f"IMC area: {IMC_area}")
logging.info(f"Overlap area / IMC area: {overlap_area/IMC_area}")
if dice_score_postIMC_on_postIMS_to_postIMS < 0.95 or overlap_area/IMC_area < 0.98:
    logging.info("use SAM for mask extraction")
    logging.info("mask extraction")
    sample_pts = boundary_points[:-1].astype(int)
    for q1 in [0.25,0.5,0.75]:
        for q2 in [0.25,0.5,0.75]:
            sample_pts = np.vstack([sample_pts,np.array([
                np.min(boundary_points[:,0]) + q1*(np.max(boundary_points[:,0])-np.min(boundary_points[:,0])),
                np.min(boundary_points[:,1]) + q2*(np.max(boundary_points[:,1])-np.min(boundary_points[:,1]))
            ]).reshape(1,2).astype(int)])
    logging.info(f"\tSample points: {sample_pts}")

    postIMC_on_postIMSmasks_tmp = extract_mask(postIMC_on_postIMS_file, bb1, rembg_session, input_spacing_2/output_spacing, sam=sam, pts=boundary_points)
    postIMC_on_postIMSmasks = np.array([msk for msk in postIMC_on_postIMSmasks_tmp if np.sum(msk)/IMC_area > 1.05])
    if len(postIMC_on_postIMSmasks) > 0:
        postIMC_on_postIMSmasks = np.array([msk for msk in postIMC_on_postIMSmasks if np.sum(msk)<np.prod(postIMC_on_postIMSmasks_tmp.shape[1:])*0.9])
    if len(postIMC_on_postIMSmasks) > 0:
        tb1 = np.array([np.sum(np.array([ postIMC_on_postIMSmasks[i,x,y] for x,y in sample_pts ])) for i in range(postIMC_on_postIMSmasks.shape[0])])
        postIMC_on_postIMSmasks = postIMC_on_postIMSmasks[tb1==np.max(tb1)]

    postIMS_to_postIMCmasks_tmp = extract_mask(postIMS_file, bb2, rembg_session, input_spacing_2/output_spacing, is_postIMS=True, sam=sam, pts=boundary_points)
    postIMS_to_postIMCmasks = np.array([msk for msk in postIMS_to_postIMCmasks_tmp if np.sum(msk)/IMC_area > 1.05])
    if len(postIMS_to_postIMCmasks) > 0:
        postIMS_to_postIMCmasks = np.array([msk for msk in postIMS_to_postIMCmasks if np.sum(msk)<np.prod(postIMS_to_postIMCmasks_tmp.shape[1:])*0.9])
    if len(postIMS_to_postIMCmasks) > 0:
        tb1 = np.array([np.sum(np.array([ postIMS_to_postIMCmasks[i,x,y] for x,y in sample_pts ])) for i in range(postIMS_to_postIMCmasks.shape[0])])
        postIMS_to_postIMCmasks = postIMS_to_postIMCmasks[tb1==np.max(tb1)]

    if len(postIMC_on_postIMSmasks) == 0 or len(postIMS_to_postIMCmasks) == 0:
        logging.info("No masks found!")
        postIMC_on_postIMS_to_postIMS = np.zeros(postIMC_on_postIMSmasks_tmp[0].shape+tuple([1]))
        postIMS_to_postIMC_on_postIMS = np.zeros(postIMS_to_postIMCmasks_tmp[0].shape+tuple([1]))
        dice_score_postIMC_on_postIMS_to_postIMS = 0
    else:
        s1 = postIMC_on_postIMSmasks.shape[1] if postIMC_on_postIMSmasks.shape[1] <= postIMS_to_postIMCmasks.shape[1] else postIMS_to_postIMCmasks.shape[1]
        s2 = postIMC_on_postIMSmasks.shape[2] if postIMC_on_postIMSmasks.shape[2] <= postIMS_to_postIMCmasks.shape[2] else postIMS_to_postIMCmasks.shape[2]
        logging.info(f"\tMask crop: {(s1,s2)}")
        postIMC_on_postIMSmasks = postIMC_on_postIMSmasks[:,:s1,:s2]
        postIMS_to_postIMCmasks = postIMS_to_postIMCmasks[:,:s1,:s2]

        logging.info("score calculation")
        postIMC_on_postIMS_to_postIMS, postIMS_to_postIMC_on_postIMS, dice_score_postIMC_on_postIMS_to_postIMS = get_max_dice_score(postIMC_on_postIMSmasks, postIMS_to_postIMCmasks)
        logging.info(f"Dice score: {dice_score_postIMC_on_postIMS_to_postIMS}")

postIMC_on_postIMS_area = np.sum(postIMC_on_postIMS_to_postIMS)/(output_spacing**2)
postIMS_to_postIMC_area = np.sum(postIMS_to_postIMC_on_postIMS)/(output_spacing**2)

reg1 = skimage.measure.regionprops(postIMC_on_postIMS_to_postIMS.astype(np.uint8)[:,:,0])
if len(reg1) == 0:
    postIMC_on_postIMS_centroid = [np.nan,np.nan]
else:
    postIMC_on_postIMS_centroid = reg1[0].centroid
reg2 = skimage.measure.regionprops(postIMS_to_postIMC_on_postIMS.astype(np.uint8)[:,:,0])
if len(reg2) == 0:
    postIMS_to_postIMC_centroid = [np.nan,np.nan]
else:
    postIMS_to_postIMC_centroid = reg2[0].centroid

postIMC_on_postIMS_to_postIMS_dist = dist_centroids(postIMC_on_postIMS_centroid, postIMS_to_postIMC_centroid, 1/output_spacing)



logging.info("Create and save csv")
samplename = os.path.basename(postIMC_on_preIMC_file).replace("_transformed_on_preIMC.ome.tiff","")
df = pd.DataFrame(data = {
    'sample': samplename,
    'dice_score_postIMC_preIMC': dice_score_postIMC_on_preIMC_to_preIMC,
    'dice_score_preIMC_preIMS': dice_score_preIMC_on_preIMS_to_preIMS,
    'dice_score_preIMS_postIMS': dice_score_preIMS_on_postIMS_to_postIMS,
    'dice_score_postIMC_postIMS': dice_score_postIMC_on_postIMS_to_postIMS,
    'postIMC_on_preIMC_mask_area': postIMC_on_preIMC_area, 
    'preIMC_mask_area': preIMC_area, 
    'preIMC_on_preIMS_mask_area': preIMC_on_preIMS_area, 
    'preIMS_mask_area': preIMS_area, 
    'preIMS_on_postIMS_mask_area': preIMS_on_postIMS_area, 
    'postIMS_mask_area': postIMS_area, 
    'postIMC_on_postIMS_mask_area': postIMC_on_postIMS_area, 
    'postIMS_to_postIMC_mask_area': postIMS_to_postIMC_area, 
    'euclidean_distance_centroids_postIMC_to_preIMC': postIMC_on_preIMC_to_preIMC_dist,
    'euclidean_distance_centroids_preIMC_to_preIMS': preIMC_on_preIMS_to_preIMS_dist,
    'euclidean_distance_centroids_preIMS_to_postIMS': preIMS_on_postIMS_to_postIMS_dist,
    'euclidean_distance_centroids_postIMC_to_postIMS': postIMC_on_postIMS_to_postIMS_dist
    }, index = [0])
df.to_csv(output_df, index=False)


logging.info("Create and save images")

tmpimg = postIMC_on_preIMC_to_preIMC[:,:,0].astype(np.uint8)*85+preIMC_to_postIMC_on_preIMC[:,:,0].astype(np.uint8)*170
tifffile.imwrite(snakemake.output['postIMCmask_preIMCmask'],tmpimg)

tmpimg = preIMC_on_preIMS_to_preIMS[:,:,0].astype(np.uint8)*85+preIMS_to_preIMC_on_preIMS[:,:,0].astype(np.uint8)*170
tifffile.imwrite(snakemake.output['preIMCmask_preIMSmask'],tmpimg)

tmpimg = preIMS_on_postIMS_to_postIMS[:,:,0].astype(np.uint8)*85+postIMS_to_preIMS_on_postIMS[:,:,0].astype(np.uint8)*170
tifffile.imwrite(snakemake.output['preIMSmask_postIMSmask'],tmpimg)

tmpimg = postIMC_on_postIMS_to_postIMS[:,:,0].astype(np.uint8)*85+postIMS_to_postIMC_on_postIMS[:,:,0].astype(np.uint8)*170
tifffile.imwrite(snakemake.output['postIMCmask_postIMSmask'],tmpimg)


logging.info("Finished")