import pandas as pd
import cv2
import matplotlib.pyplot as plt
from rembg import remove, new_session
import skimage
import numpy as np
import tifffile
from image_registration_IMS_to_preIMS_utils import readimage_crop, prepare_image_for_sam, get_max_dice_score, dist_centroids, subtract_postIMS_grid
import sys,os
import logging, traceback
logging.basicConfig(filename=snakemake.log["stdout"],
                    level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    )
from logging_utils import handle_exception, StreamToLogger
sys.excepthook = handle_exception
sys.stdout = StreamToLogger(logging.getLogger(),logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger(),logging.ERROR)

logging.info("Start")

# postIMC_on_preIMC_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/Cirrhosis-TMA-5_New_Detector_001_transformed_on_preIMC.ome.tiff"
postIMC_on_preIMC_file = snakemake.input["postIMC_on_preIMC"]
# preIMC_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMC/test_split_pre_preIMC.ome.tiff"
preIMC_file = snakemake.input["preIMC"]
# preIMC_on_preIMS_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMC/Cirrhosis-TMA-5_New_Detector_001_transformed_on_preIMS.ome.tiff"
preIMC_on_preIMS_file = snakemake.input["preIMC_on_preIMS"]
# preIMS_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMS/test_split_pre_preIMS.ome.tiff"
preIMS_file = snakemake.input["preIMS"]
# preIMS_on_postIMS_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMS/Cirrhosis-TMA-5_New_Detector_001_transformed_on_postIMS.ome.tiff"
preIMS_on_postIMS_file = snakemake.input["preIMS_on_postIMS"]
# postIMS_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMS/test_split_pre_postIMS.ome.tiff"
postIMS_file = snakemake.input["postIMS"]


# # postIMS_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMS/test_split_pre_postIMS_reduced.ome.tiff"
# postIMS_file = snakemake.input["postIMS_downscaled"]
# # postIMC_file="/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/Cirrhosis-TMA-5_New_Detector_001_transformed.ome.tiff"
# postIMC_file = snakemake.input["postIMC_transformed"]
# # preIMS_file =  "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMS/test_split_pre-preIMS_to_postIMS_registered_reduced.ome.tiff"
# preIMS_file = snakemake.input["preIMS_downscaled"]
# # resize factor to speedup computations
# rescale = 0.22537
rescale = snakemake.params["downscale"]

output_df = snakemake.output["registration_metrics"]

logging.info("Setup rembg model")
# prepare model for rembg
model_name = "isnet-general-use"
os.environ["OMP_NUM_THREADS"] = str(snakemake.threads)
rembg_session = new_session(model_name)

def extract_mask(file, bb, session, is_postIMS=False):
    w = readimage_crop(file, bb)
    w = prepare_image_for_sam(w, rescale)
    if is_postIMS:
        w = subtract_postIMS_grid(w)
        w = cv2.blur(w, (5,5))
    w = np.stack([w, w, w], axis=2)
    wr = remove(w, only_mask=True, session=session)
    masks = wr>127
    masks = skimage.morphology.remove_small_holes(masks,100**2*np.pi)
    masks = cv2.morphologyEx(masks.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5,5),np.uint8)).astype(bool)
    masks = np.stack([masks])
    return masks


logging.info("postIMC on preIMC bounding box extraction")
postIMC_on_preIMC = skimage.io.imread(postIMC_on_preIMC_file)
rp = skimage.measure.regionprops((postIMC_on_preIMC[:,:,0]>0).astype(np.uint8))
bb1 = rp[0].bbox

logging.info("mask extraction")
postIMC_on_preIMCmasks = extract_mask(postIMC_on_preIMC_file, bb1, rembg_session)
preIMCmasks = extract_mask(preIMC_file, bb1, rembg_session)

logging.info("score calculation")
postIMC_on_preIMC_to_preIMC, preIMC_to_postIMC_on_preIMC, dice_score_postIMC_on_preIMC_to_preIMC = get_max_dice_score(postIMC_on_preIMCmasks, preIMCmasks)

postIMC_on_preIMC_area = np.sum(postIMC_on_preIMC_to_preIMC)
preIMC_area = np.sum(preIMC_to_postIMC_on_preIMC)

reg1 = skimage.measure.regionprops(postIMC_on_preIMC_to_preIMC.astype(np.uint8)[:,:,0])
postIMC_on_preIMC_centroid = reg1[0].centroid
reg2 = skimage.measure.regionprops(preIMC_to_postIMC_on_preIMC.astype(np.uint8)[:,:,0])
preIMC_centroid = reg2[0].centroid

postIMC_on_preIMC_to_preIMC_dist = dist_centroids(postIMC_on_preIMC_centroid, preIMC_centroid, 1)


logging.info("preIMC on preIMS bounding box extraction")
preIMC_on_preIMS = skimage.io.imread(preIMC_on_preIMS_file)
rp = skimage.measure.regionprops((preIMC_on_preIMS[:,:,0]>0).astype(np.uint8))
bb1 = rp[0].bbox

logging.info("mask extraction")
preIMC_on_preIMSmasks = extract_mask(preIMC_on_preIMS_file, bb1, rembg_session)
preIMSmasks = extract_mask(preIMS_file, bb1, rembg_session)

logging.info("score calculation")
preIMC_on_preIMS_to_preIMS, preIMS_to_preIMC_on_preIMS, dice_score_preIMC_on_preIMS_to_preIMS = get_max_dice_score(preIMC_on_preIMSmasks, preIMSmasks)

preIMC_on_preIMS_area = np.sum(preIMC_on_preIMS_to_preIMS)
preIMS_area = np.sum(preIMS_to_preIMC_on_preIMS)

reg1 = skimage.measure.regionprops(preIMC_on_preIMS_to_preIMS.astype(np.uint8)[:,:,0])
preIMC_on_preIMS_centroid = reg1[0].centroid
reg2 = skimage.measure.regionprops(preIMS_to_preIMC_on_preIMS.astype(np.uint8)[:,:,0])
preIMS_centroid = reg2[0].centroid

preIMC_on_preIMS_to_preIMS_dist = dist_centroids(preIMC_on_preIMS_centroid, preIMS_centroid, 1)


logging.info("preIMS on postIMS bounding box extraction")
preIMS_on_postIMS = skimage.io.imread(preIMS_on_postIMS_file)
rp = skimage.measure.regionprops((preIMS_on_postIMS[:,:,0]>0).astype(np.uint8))
bb1 = rp[0].bbox

logging.info("mask extraction")
preIMS_on_postIMSmasks = extract_mask(preIMS_on_postIMS_file, bb1, rembg_session)
postIMSmasks = extract_mask(postIMS_file, bb1, rembg_session, is_postIMS=True)

logging.info("score calculation")
preIMS_on_postIMS_to_postIMS, postIMS_to_preIMS_on_postIMS, dice_score_preIMS_on_postIMS_to_postIMS = get_max_dice_score(preIMS_on_postIMSmasks, postIMSmasks)

preIMS_on_postIMS_area = np.sum(preIMS_on_postIMS_to_postIMS)
postIMS_area = np.sum(postIMS_to_preIMS_on_postIMS)

reg1 = skimage.measure.regionprops(preIMS_on_postIMS_to_postIMS.astype(np.uint8)[:,:,0])
preIMS_on_postIMS_centroid = reg1[0].centroid
reg2 = skimage.measure.regionprops(postIMS_to_preIMS_on_postIMS.astype(np.uint8)[:,:,0])
postIMS_centroid = reg2[0].centroid

preIMS_on_postIMS_to_postIMS_dist = dist_centroids(preIMS_on_postIMS_centroid, postIMS_centroid, 1)

# logging.info("postIMS mask extraction")
# # postIMS
# postIMSw = readimage_crop(postIMS_file, bb1)
# postIMSw = prepare_image_for_sam(postIMSw, rescale)
# postIMS = postIMSw.copy()
# tmpmask = skimage.morphology.isotropic_dilation(preIMSmasks[0], np.ceil(rescale*100))
# postIMS[np.logical_not(tmpmask)] = 0
# postIMS = skimage.filters.median(postIMS, skimage.morphology.disk(rescale * 5))
# postIMS = normalize_image(postIMS)*255
# postIMS = postIMS.astype(np.uint8)
# postIMS = np.stack([postIMS, postIMS, postIMS], axis=2)
# postIMSmasks, scores1 = sam_core(postIMS, sam)
# postIMSmasks = np.stack([preprocess_mask(msk,rescale) for msk in postIMSmasks ])


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

tmpimg = postIMC_on_preIMCmasks[0,:,:].astype(np.uint8)*127+preIMCmasks[0,:,:].astype(np.uint8)*127
tifffile.imwrite(snakemake.output['postIMCmask_preIMCmask'],tmpimg)

tmpimg = preIMC_on_preIMSmasks[0,:,:].astype(np.uint8)*127+preIMSmasks[0,:,:].astype(np.uint8)*127
tifffile.imwrite(snakemake.output['preIMCmask_preIMSmask'],tmpimg)

tmpimg = preIMS_on_postIMSmasks[0,:,:].astype(np.uint8)*127+postIMSmasks[0,:,:].astype(np.uint8)*127
tifffile.imwrite(snakemake.output['preIMSmask_postIMSmask'],tmpimg)

# # save images for QC
# tifffile.imwrite("",img1w)
# tifffile.imwrite("",img2w)
# tifffile.imwrite("",mask_image1[:,:,0])
# tifffile.imwrite("",mask_image2[:,:,0])


# fig, ax = plt.subplots(nrows=3, ncols=2)
# ax[0,0].imshow(img3)
# ax[0,0].set_title("preIMS")
# ax[1,0].imshow(mask_image3, cmap='gray')
# ax[1,0].set_title("preIMS mask")
# ax[0,1].imshow(img2)
# ax[0,1].set_title("postIMC")
# ax[1,1].imshow(mask_image2, cmap='gray')
# ax[1,1].set_title("postIMC mask")
# ax[2,1].imshow(mask_image2.astype(np.uint8) - mask_image3.astype(np.uint8), cmap='gray')
# ax[2,1].set_title("postIMC mask - postIMS mask")
# plt.show()



# fig, ax = plt.subplots(nrows=1, ncols=4)
# ax[0].imshow(masks1[0], cmap='gray')
# ax[0].set_title("0")
# ax[1].imshow(masks1[1], cmap='gray')
# ax[1].set_title("1")
# ax[2].imshow(masks1[2], cmap='gray')
# ax[2].set_title("2")
# ax[3].imshow(img1, cmap='gray')
# ax[3].set_title("3")
# plt.show()


logging.info("Finished")