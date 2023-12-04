import pandas as pd
import matplotlib.pyplot as plt
from rembg import new_session
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
    snakemake.input["postIMC_on_preIMC"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/Cirrhosis-TMA-5_New_Detector_001_transformed_on_preIMC.ome.tiff"
    snakemake.input['preIMC'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMC/test_split_pre_preIMC.ome.tiff"
    snakemake.input['preIMC_on_preIMS'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMC/Cirrhosis-TMA-5_New_Detector_001_transformed_on_preIMS.ome.tiff"
    snakemake.input['preIMS'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMS/test_split_pre_preIMS.ome.tiff"
    snakemake.input['preIMS_on_postIMS'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMS/Cirrhosis-TMA-5_New_Detector_002_transformed_on_postIMS.ome.tiff"
    snakemake.input['postIMS'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMS/test_split_pre_postIMS.ome.tiff"
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
preIMC_on_preIMS_file = snakemake.input["preIMC_on_preIMS"]
preIMS_file = snakemake.input["preIMS"]
preIMS_on_postIMS_file = snakemake.input["preIMS_on_postIMS"]
postIMS_file = snakemake.input["postIMS"]
# outputs
output_df = snakemake.output["registration_metrics"]

logging.info("Setup rembg model")
# prepare model for rembg
model_name = "isnet-general-use"
os.environ["OMP_NUM_THREADS"] = str(snakemake.threads)
rembg_session = new_session(model_name)


logging.info("postIMC on preIMC bounding box extraction")
postIMC_on_preIMC = skimage.io.imread(postIMC_on_preIMC_file)
rp = skimage.measure.regionprops((postIMC_on_preIMC[:,:,0]>0).astype(np.uint8))
bb1 = rp[0].bbox
bb2 = [int(bb1[0]/input_spacing_2),int(bb1[1]/input_spacing_2),int(bb1[2]/input_spacing_2),int(bb1[3]/input_spacing_2)] 

logging.info("mask extraction")
postIMC_on_preIMCmasks = extract_mask(postIMC_on_preIMC_file, bb1, rembg_session, input_spacing_1/output_spacing)
preIMCmasks = extract_mask(preIMC_file, bb2, rembg_session, input_spacing_2/output_spacing)
s1 = postIMC_on_preIMCmasks.shape[1] if postIMC_on_preIMCmasks.shape[1] <= preIMCmasks.shape[1] else preIMCmasks.shape[1]
s2 = postIMC_on_preIMCmasks.shape[2] if postIMC_on_preIMCmasks.shape[2] <= preIMCmasks.shape[2] else preIMCmasks.shape[2]
logging.info(f"\tMask crop: {(s1,s2)}")
postIMC_on_preIMCmasks = postIMC_on_preIMCmasks[:,:s1,:s2]
preIMCmasks = preIMCmasks[:,:s1,:s2]

logging.info("score calculation")
postIMC_on_preIMC_to_preIMC, preIMC_to_postIMC_on_preIMC, dice_score_postIMC_on_preIMC_to_preIMC = get_max_dice_score(postIMC_on_preIMCmasks, preIMCmasks)

postIMC_on_preIMC_area = np.sum(postIMC_on_preIMC_to_preIMC)/(output_spacing**2)
preIMC_area = np.sum(preIMC_to_postIMC_on_preIMC)/(output_spacing**2)

reg1 = skimage.measure.regionprops(postIMC_on_preIMC_to_preIMC.astype(np.uint8)[:,:,0])
postIMC_on_preIMC_centroid = reg1[0].centroid
reg2 = skimage.measure.regionprops(preIMC_to_postIMC_on_preIMC.astype(np.uint8)[:,:,0])
preIMC_centroid = reg2[0].centroid

postIMC_on_preIMC_to_preIMC_dist = dist_centroids(postIMC_on_preIMC_centroid, preIMC_centroid, 1/output_spacing)


logging.info("preIMC on preIMS bounding box extraction")
preIMC_on_preIMS = skimage.io.imread(preIMC_on_preIMS_file)
rp = skimage.measure.regionprops((preIMC_on_preIMS[:,:,0]>0).astype(np.uint8))
bb1 = rp[0].bbox
bb2 = [int(bb1[0]/input_spacing_2),int(bb1[1]/input_spacing_2),int(bb1[2]/input_spacing_2),int(bb1[3]/input_spacing_2)] 

logging.info("mask extraction")
preIMC_on_preIMSmasks = extract_mask(preIMC_on_preIMS_file, bb1, rembg_session, input_spacing_1/output_spacing)
preIMSmasks = extract_mask(preIMS_file, bb2, rembg_session, input_spacing_2/output_spacing)
s1 = preIMC_on_preIMSmasks.shape[1] if preIMC_on_preIMSmasks.shape[1] <= preIMSmasks.shape[1] else preIMSmasks.shape[1]
s2 = preIMC_on_preIMSmasks.shape[2] if preIMC_on_preIMSmasks.shape[2] <= preIMSmasks.shape[2] else preIMSmasks.shape[2]
logging.info(f"\tMask crop: {(s1,s2)}")
preIMC_on_preIMSmasks = preIMC_on_preIMSmasks[:,:s1,:s2]
preIMSmasks = preIMSmasks[:,:s1,:s2]


logging.info("score calculation")
preIMC_on_preIMS_to_preIMS, preIMS_to_preIMC_on_preIMS, dice_score_preIMC_on_preIMS_to_preIMS = get_max_dice_score(preIMC_on_preIMSmasks, preIMSmasks)

preIMC_on_preIMS_area = np.sum(preIMC_on_preIMS_to_preIMS)/(output_spacing**2)
preIMS_area = np.sum(preIMS_to_preIMC_on_preIMS)/(output_spacing**2)

reg1 = skimage.measure.regionprops(preIMC_on_preIMS_to_preIMS.astype(np.uint8)[:,:,0])
preIMC_on_preIMS_centroid = reg1[0].centroid
reg2 = skimage.measure.regionprops(preIMS_to_preIMC_on_preIMS.astype(np.uint8)[:,:,0])
preIMS_centroid = reg2[0].centroid

preIMC_on_preIMS_to_preIMS_dist = dist_centroids(preIMC_on_preIMS_centroid, preIMS_centroid, 1/output_spacing)


logging.info("preIMS on postIMS bounding box extraction")
preIMS_on_postIMS = skimage.io.imread(preIMS_on_postIMS_file)
rp = skimage.measure.regionprops((preIMS_on_postIMS[:,:,0]>0).astype(np.uint8))
bb1 = rp[0].bbox
bb2 = [int(bb1[0]/input_spacing_2),int(bb1[1]/input_spacing_2),int(bb1[2]/input_spacing_2),int(bb1[3]/input_spacing_2)] 

logging.info("mask extraction")
preIMS_on_postIMSmasks = extract_mask(preIMS_on_postIMS_file, bb1, rembg_session, input_spacing_1/output_spacing)
postIMSmasks = extract_mask(postIMS_file, bb2, rembg_session, input_spacing_2/output_spacing, is_postIMS=True)
s1 = preIMS_on_postIMSmasks.shape[1] if preIMS_on_postIMSmasks.shape[1] <= postIMSmasks.shape[1] else postIMSmasks.shape[1]
s2 = preIMS_on_postIMSmasks.shape[2] if preIMS_on_postIMSmasks.shape[2] <= postIMSmasks.shape[2] else postIMSmasks.shape[2]
logging.info(f"\tMask crop: {(s1,s2)}")
preIMS_on_postIMSmasks = preIMS_on_postIMSmasks[:,:s1,:s2]
postIMSmasks = postIMSmasks[:,:s1,:s2]

logging.info("score calculation")
preIMS_on_postIMS_to_postIMS, postIMS_to_preIMS_on_postIMS, dice_score_preIMS_on_postIMS_to_postIMS = get_max_dice_score(preIMS_on_postIMSmasks, postIMSmasks)

preIMS_on_postIMS_area = np.sum(preIMS_on_postIMS_to_postIMS)/(output_spacing**2)
postIMS_area = np.sum(postIMS_to_preIMS_on_postIMS)/(output_spacing**2)

reg1 = skimage.measure.regionprops(preIMS_on_postIMS_to_postIMS.astype(np.uint8)[:,:,0])
preIMS_on_postIMS_centroid = reg1[0].centroid
reg2 = skimage.measure.regionprops(postIMS_to_preIMS_on_postIMS.astype(np.uint8)[:,:,0])
postIMS_centroid = reg2[0].centroid

preIMS_on_postIMS_to_postIMS_dist = dist_centroids(preIMS_on_postIMS_centroid, postIMS_centroid, 1/output_spacing)


logging.info("Create and save csv")
samplename = os.path.basename(postIMC_on_preIMC_file).replace("_transformed_on_preIMC.ome.tiff","")
df = pd.DataFrame(data = {
    'sample': samplename,
    'dice_score_postIMC_preIMC': dice_score_postIMC_on_preIMC_to_preIMC,
    'dice_score_preIMC_preIMS': dice_score_preIMC_on_preIMS_to_preIMS,
    'dice_score_preIMS_postIMS': dice_score_preIMS_on_postIMS_to_postIMS,
    'postIMC_on_preIMC_mask_area': postIMC_on_preIMC_area, 
    'preIMC_mask_area': preIMC_area, 
    'preIMC_on_preIMS_mask_area': preIMC_on_preIMS_area, 
    'preIMS_mask_area': preIMS_area, 
    'preIMS_on_postIMS_mask_area': preIMS_on_postIMS_area, 
    'postIMS_mask_area': postIMS_area, 
    'euclidean_distance_centroids_postIMC_to_preIMC': postIMC_on_preIMC_to_preIMC_dist,
    'euclidean_distance_centroids_preIMC_to_preIMS': preIMC_on_preIMS_to_preIMS_dist,
    'euclidean_distance_centroids_preIMS_to_postIMS': preIMS_on_postIMS_to_postIMS_dist
    }, index = [0])
df.to_csv(output_df, index=False)


logging.info("Create and save images")

tmpimg = postIMC_on_preIMCmasks[0,:,:].astype(np.uint8)*85+preIMCmasks[0,:,:].astype(np.uint8)*170
tifffile.imwrite(snakemake.output['postIMCmask_preIMCmask'],tmpimg)

tmpimg = preIMC_on_preIMSmasks[0,:,:].astype(np.uint8)*85+preIMSmasks[0,:,:].astype(np.uint8)*170
tifffile.imwrite(snakemake.output['preIMCmask_preIMSmask'],tmpimg)

tmpimg = preIMS_on_postIMSmasks[0,:,:].astype(np.uint8)*85+postIMSmasks[0,:,:].astype(np.uint8)*170
tifffile.imwrite(snakemake.output['preIMSmask_postIMSmask'],tmpimg)

logging.info("Finished")