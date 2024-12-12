import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
import pandas as pd
import json
import cv2
from rembg import new_session
from segment_anything import sam_model_registry
import skimage
import numpy as np
import tifffile
from image_utils import extract_mask, get_pyrlvl_rescalemod_imgshape
from registration_utils import get_max_dice_score, dist_centroids
from utils import setNThreads, snakeMakeMock
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
input_spacing = snakemake.params["input_spacing"]
rescale = snakemake.params["rescale"]

# inputs
postIMC_on_preIMC_file = snakemake.input["postIMC_on_preIMC"]
preIMC_file = snakemake.input["preIMC"]
TMA_on_preIMC_location = snakemake.input["TMA_on_preIMC"]
if isinstance(TMA_on_preIMC_location, list):
    TMA_on_preIMC_location = TMA_on_preIMC_location[0]
logging.info(f"IMC_on_preIMC_location: {TMA_on_preIMC_location}")
preIMC_on_preIMS_file = snakemake.input["preIMC_on_preIMS"]
preIMS_file = snakemake.input["preIMS"]
TMA_on_preIMS_location = snakemake.input["TMA_on_preIMS"]
if isinstance(TMA_on_preIMS_location, list):
    TMA_on_preIMS_location = TMA_on_preIMS_location[0]
logging.info(f"TMA_on_preIMS_location: {TMA_on_preIMS_location}")
preIMS_on_postIMS_file = snakemake.input["preIMS_on_postIMS"]
postIMS_file = snakemake.input["postIMS"]
postIMC_on_postIMS_file = snakemake.input["postIMC_on_postIMS"]
TMA_on_postIMS_location = snakemake.input["TMA_on_postIMS"]
if isinstance(TMA_on_postIMS_location, list):
    TMA_on_postIMS_location = TMA_on_postIMS_location[0]
logging.info(f"IMC_on_postIMS_location: {TMA_on_postIMS_location}")
IMC_file = snakemake.input["IMC"]
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

def get_mask_stats(img1_file, img2_file, tmashape_file, rescale, rembg_session, input_spacing, img1_is_postIMS=False, img2_is_postIMS=False):
    img1_pyr_level, img1_rescale, img1_rescale_modifier, img1_shape = get_pyrlvl_rescalemod_imgshape(img1_file, rescale)
    img2_pyr_level, img2_rescale, img2_rescale_modifier, img2_shape = get_pyrlvl_rescalemod_imgshape(img2_file, rescale)

    TMA_geojson = json.load(open(tmashape_file, "r"))
    if isinstance(TMA_geojson,list):
        TMA_geojson=TMA_geojson[0]
    boundary_points = np.array(TMA_geojson['geometry']['coordinates'])[0,:,:]
    xmin=int(np.min(boundary_points[:,1])/rescale)
    xmax=int(np.max(boundary_points[:,1])/rescale)
    ymin=int(np.min(boundary_points[:,0])/rescale)
    ymax=int(np.max(boundary_points[:,0])/rescale)
    bbox = np.array([xmin,ymin,xmax,ymax])

    img1_masks = extract_mask(img1_file,(np.ceil(np.array([xmin,ymin,xmax,ymax])*img1_rescale)).astype(int), rembg_session, img1_rescale_modifier/rescale, is_postIMS=img1_is_postIMS, pyr_level=img1_pyr_level)
    img2_masks = extract_mask(img2_file,(np.ceil(np.array([xmin,ymin,xmax,ymax])*img2_rescale)).astype(int), rembg_session, img2_rescale_modifier/rescale, is_postIMS=img2_is_postIMS , pyr_level=img2_pyr_level)

    logging.info("score calculation")
    if not np.any(img1_masks) or not np.any(img2_masks):
        logging.info("No masks found!")
        dice_score_img1_to_img2 = 0
        overlap_area=0
    else:
        img1_to_img2, img2_to_img1, dice_score_img1_to_img2 = get_max_dice_score(img1_masks, img2_masks)
        overlap_area = np.sum(np.logical_and(img1_to_img2, img2_to_img1))

    # img1_area = np.sum(img1_to_img2)/(input_spacing**2)
    # img2_area = np.sum(img2_to_img1)/(input_spacing**2)

    # regionprops using cv2
    if not np.any(img1_to_img2):
        img1_area = 0
        img1_centroid = [np.nan,np.nan]
    else:
        cc = cv2.connectedComponentsWithStats(img1_to_img2.astype(np.uint8)[:,:,0])
        img1_maxind = np.argmax(cc[2][:,4])
        img1_area = cc[2][img1_maxind,4]/(input_spacing**2)
        img1_centroid = cc[3][img1_maxind]/(input_spacing)

    if not np.any(img2_to_img1):
        img2_area = 0
        img2_centroid = [np.nan,np.nan]
    else:
        cc = cv2.connectedComponentsWithStats(img2_to_img1.astype(np.uint8)[:,:,0])
        img2_maxind = np.argmax(cc[2][:,4])
        img2_area = cc[2][img2_maxind,4]/(input_spacing**2)
        img2_centroid = cc[3][img2_maxind]/(input_spacing)

    img1_to_img2_dist = dist_centroids(img1_centroid, img2_centroid, 1)

    return img1_area, img2_area, img1_to_img2_dist, dice_score_img1_to_img2, img1_to_img2, img2_to_img1


logging.info(f"postIMC on preIMC")
postIMC_on_preIMC_area, preIMC_area, postIMC_on_preIMC_to_preIMC_dist, dice_score_postIMC_on_preIMC_to_preIMC, postIMC_on_preIMC_to_preIMC, preIMC_to_postIMC_on_preIMC = get_mask_stats(postIMC_on_preIMC_file, preIMC_file, TMA_on_preIMC_location, rescale, rembg_session, input_spacing)


logging.info(f"preIMC on preIMS")
preIMC_on_preIMS_area, preIMS_area, preIMC_on_preIMS_to_preIMS_dist, dice_score_preIMC_on_preIMS_to_preIMS, preIMC_on_preIMS_to_preIMS, preIMS_to_preIMC_on_preIMS = get_mask_stats(preIMC_on_preIMS_file, preIMS_file, TMA_on_preIMS_location, rescale, rembg_session, input_spacing)

logging.info(f"preIMS on postIMS")
preIMS_on_postIMS_area, postIMS_area, preIMS_on_postIMS_to_postIMS_dist, dice_score_preIMS_on_postIMS_to_postIMS, preIMS_on_postIMS_to_postIMS, postIMS_to_preIMS_on_postIMS = get_mask_stats(preIMS_on_postIMS_file, postIMS_file, TMA_on_postIMS_location, rescale, rembg_session, input_spacing, img2_is_postIMS=True)

logging.info(f"postIMC on postIMS")
postIMC_on_postIMS_area, postIMS_to_postIMC_area, postIMC_on_postIMS_to_postIMS_dist, dice_score_postIMC_on_postIMS_to_postIMS, postIMC_on_postIMS_to_postIMS, postIMS_to_postIMC_on_postIMS = get_mask_stats(postIMC_on_postIMS_file, postIMS_file, TMA_on_postIMS_location, rescale, rembg_session, input_spacing, img2_is_postIMS=True)

# logging.info(f"Dice score: {dice_score_postIMC_on_preIMC_to_preIMC}")
# logging.info(f"Overlap area: {overlap_area}")
# logging.info(f"IMC area: {IMC_area}")
# logging.info(f"Overlap area / IMC area: {overlap_area/IMC_area}")
# if dice_score_postIMC_on_preIMC_to_preIMC < 0.95 or overlap_area/IMC_area < 0.98:
#     logging.info("use SAM for mask extraction")
#     logging.info("mask extraction")
#     sample_pts = boundary_points[:-1].astype(int)
#     for q1 in [0.25,0.5,0.75]:
#         for q2 in [0.25,0.5,0.75]:
#             sample_pts = np.vstack([sample_pts,np.array([
#                 np.min(boundary_points[:,0]) + q1*(np.max(boundary_points[:,0])-np.min(boundary_points[:,0])),
#                 np.min(boundary_points[:,1]) + q2*(np.max(boundary_points[:,1])-np.min(boundary_points[:,1]))
#             ]).reshape(1,2).astype(int)])
#     logging.info(f"\tSample points: {sample_pts}")

#     postIMC_on_preIMCmasks_tmp = extract_mask(postIMC_on_preIMC_file, bb1, rembg_session, input_spacing_1/output_spacing, sam=sam, pts=sample_pts)
#     postIMC_on_preIMCmasks = np.array([msk for msk in postIMC_on_preIMCmasks_tmp if np.sum(msk)/IMC_area > 1.05])
#     if len(postIMC_on_preIMCmasks) > 0:
#         postIMC_on_preIMCmasks = np.array([msk for msk in postIMC_on_preIMCmasks if np.sum(msk)<np.prod(postIMC_on_preIMCmasks_tmp.shape[1:])*0.9])
#     if len(postIMC_on_preIMCmasks) > 0:
#         tb1 = np.array([np.sum(np.array([ postIMC_on_preIMCmasks[i,x,y] for x,y in sample_pts ])) for i in range(postIMC_on_preIMCmasks.shape[0])])
#         postIMC_on_preIMCmasks = postIMC_on_preIMCmasks[tb1==np.max(tb1)]

#     preIMCmasks_tmp = extract_mask(preIMC_file, bb2, rembg_session, input_spacing_2/output_spacing, sam=sam, pts=sample_pts)
#     preIMCmasks = np.array([msk for msk in preIMCmasks_tmp if np.sum(msk)/IMC_area > 1.05])
#     if len(preIMCmasks) > 0:
#         preIMCmasks = np.array([msk for msk in preIMCmasks if np.sum(msk)<np.prod(preIMCmasks_tmp.shape[1:])*0.9])
#     if len(preIMCmasks) > 0:
#         tb1 = np.array([np.sum(np.array([ preIMCmasks[i,x,y] for x,y in sample_pts ])) for i in range(preIMCmasks.shape[0])])
#         preIMCmasks = preIMCmasks[tb1==np.max(tb1)]

#     if len(postIMC_on_preIMCmasks) == 0 or len(preIMCmasks) == 0:
#         logging.info("No masks found!")
#         postIMC_on_preIMC_to_preIMC = np.zeros(postIMC_on_preIMCmasks_tmp[0].shape+tuple([1]))
#         preIMC_to_postIMC_on_preIMC = np.zeros(preIMCmasks_tmp[0].shape+tuple([1]))
#         dice_score_postIMC_on_preIMC_to_preIMC = 0
#     else:
#         s1 = postIMC_on_preIMCmasks.shape[1] if postIMC_on_preIMCmasks.shape[1] <= preIMCmasks.shape[1] else preIMCmasks.shape[1]
#         s2 = postIMC_on_preIMCmasks.shape[2] if postIMC_on_preIMCmasks.shape[2] <= preIMCmasks.shape[2] else preIMCmasks.shape[2]
#         logging.info(f"\tMask crop: {(s1,s2)}")
#         postIMC_on_preIMCmasks = postIMC_on_preIMCmasks[:,:s1,:s2]
#         preIMCmasks = preIMCmasks[:,:s1,:s2]

#         logging.info("score calculation")
#         postIMC_on_preIMC_to_preIMC, preIMC_to_postIMC_on_preIMC, dice_score_postIMC_on_preIMC_to_preIMC = get_max_dice_score(postIMC_on_preIMCmasks, preIMCmasks)
#         logging.info(f"Dice score: {dice_score_postIMC_on_preIMC_to_preIMC}")

logging.info("Create and save csv")
samplename = os.path.basename(IMC_file).replace(".tiff","")
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
s1 = postIMC_on_preIMC_to_preIMC.shape[0] if postIMC_on_preIMC_to_preIMC.shape[0] <= preIMC_to_postIMC_on_preIMC.shape[0] else preIMC_to_postIMC_on_preIMC.shape[0]
s2 = postIMC_on_preIMC_to_preIMC.shape[1] if postIMC_on_preIMC_to_preIMC.shape[1] <= preIMC_to_postIMC_on_preIMC.shape[1] else preIMC_to_postIMC_on_preIMC.shape[1]
logging.info(f"\timage 1: {postIMC_on_preIMC_to_preIMC.shape}")
logging.info(f"\timage 2: {preIMC_to_postIMC_on_preIMC.shape}")
logging.info(f"\tMask crop: {(s1,s2)}")
t1 = postIMC_on_preIMC_to_preIMC[:s1,:s2,0]
t2 = preIMC_to_postIMC_on_preIMC[:s1,:s2,0]
tmpimg = t1.astype(np.uint8)*85+t2.astype(np.uint8)*170
tifffile.imwrite(snakemake.output['postIMCmask_preIMCmask'],tmpimg)

s1 = preIMC_on_preIMS_to_preIMS.shape[0] if preIMC_on_preIMS_to_preIMS.shape[0] <= preIMS_to_preIMC_on_preIMS.shape[0] else preIMS_to_preIMC_on_preIMS.shape[0]
s2 = preIMC_on_preIMS_to_preIMS.shape[1] if preIMC_on_preIMS_to_preIMS.shape[1] <= preIMS_to_preIMC_on_preIMS.shape[1] else preIMS_to_preIMC_on_preIMS.shape[1]
logging.info(f"\timage 1: {preIMC_on_preIMS_to_preIMS.shape}")
logging.info(f"\timage 2: {preIMS_to_preIMC_on_preIMS.shape}")
logging.info(f"\tMask crop: {(s1,s2)}")
t1 = preIMC_on_preIMS_to_preIMS[:s1,:s2,0]
t2 = preIMS_to_preIMC_on_preIMS[:s1,:s2,0]
tmpimg = t1.astype(np.uint8)*85+t2.astype(np.uint8)*170
tifffile.imwrite(snakemake.output['preIMCmask_preIMSmask'],tmpimg)

s1 = preIMS_on_postIMS_to_postIMS.shape[0] if preIMS_on_postIMS_to_postIMS.shape[0] <= postIMS_to_preIMS_on_postIMS.shape[0] else postIMS_to_preIMS_on_postIMS.shape[0]
s2 = preIMS_on_postIMS_to_postIMS.shape[1] if preIMS_on_postIMS_to_postIMS.shape[1] <= postIMS_to_preIMS_on_postIMS.shape[1] else postIMS_to_preIMS_on_postIMS.shape[1]
logging.info(f"\timage 1: {preIMS_on_postIMS_to_postIMS.shape}")
logging.info(f"\timage 2: {postIMS_to_preIMS_on_postIMS.shape}")
logging.info(f"\tMask crop: {(s1,s2)}")
t1 = preIMS_on_postIMS_to_postIMS[:s1,:s2,0]
t2 = postIMS_to_preIMS_on_postIMS[:s1,:s2,0]
tmpimg = t1.astype(np.uint8)*85+t2.astype(np.uint8)*170
tifffile.imwrite(snakemake.output['preIMSmask_postIMSmask'],tmpimg)

s1 = postIMC_on_postIMS_to_postIMS.shape[0] if postIMC_on_postIMS_to_postIMS.shape[0] <= postIMS_to_postIMC_on_postIMS.shape[0] else postIMS_to_postIMC_on_postIMS.shape[0]
s2 = postIMC_on_postIMS_to_postIMS.shape[1] if postIMC_on_postIMS_to_postIMS.shape[1] <= postIMS_to_postIMC_on_postIMS.shape[1] else postIMS_to_postIMC_on_postIMS.shape[1]
logging.info(f"\timage 1: {postIMC_on_postIMS_to_postIMS.shape}")
logging.info(f"\timage 2: {postIMS_to_postIMC_on_postIMS.shape}")
logging.info(f"\tMask crop: {(s1,s2)}")
t1 = postIMC_on_postIMS_to_postIMS[:s1,:s2,0]
t2 = postIMS_to_postIMC_on_postIMS[:s1,:s2,0]
tmpimg = t1.astype(np.uint8)*85+t2.astype(np.uint8)*170
tifffile.imwrite(snakemake.output['postIMCmask_postIMSmask'],tmpimg)

logging.info("Finished")