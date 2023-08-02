import pandas as pd
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import torch
from rembg import remove, new_session
import skimage
import numpy as np
import tifffile
import zarr
from wsireg.utils.im_utils import grayscale
from image_registration_IMS_to_preIMS_utils import normalize_image, readimage_crop, prepare_image_for_sam, apply_filter, preprocess_mask, get_max_dice_score, smooth_mask, dist_centroids,sam_core
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

# image_file1 ="/home/retger/Downloads/QuPath-0.4.1-Linux/Projects/cirrhosis_TMA/cirrhosis_TMA-postIMS_registered.ome.tiff"
# postIMS_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMS/test_split_pre_postIMS_reduced.ome.tiff"
# postIMS_file = "/home/retger/Downloads/sherborne/results/Lipid_TMA_3781/data/postIMS/Lipid_TMA_3781_postIMS_reduced.ome.tiff"
postIMS_file = snakemake.input["postIMS_downscaled"]
# image_file2 ="/home/retger/Downloads/QuPath-0.4.1-Linux/Projects/cirrhosis_TMA/cirrhosis_TMA-postIMC_to_postIMS_registered.ome.tiff"
# postIMC_file="/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/Cirrhosis-TMA-5_New_Detector_001_transformed.ome.tiff"
# postIMC_file = "/home/retger/Downloads/sherborne/results/Lipid_TMA_3781/data/postIMC/Lipid_TMA_37819_009_transformed.ome.tiff"
postIMC_file = snakemake.input["postIMC_transformed"]
# imc_mask1 ="/home/retger/Downloads/QuPath-0.4.1-Linux/Projects/cirrhosis_TMA/Cirrhosis_TMA_5_01262022_004_aggr_transformed.ome.tiff"
# imc_mask1="/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_001_transformed.ome.tiff"
# imc_mask2 ="/home/retger/Downloads/QuPath-0.4.1-Linux/Projects/cirrhosis_TMA/Cirrhosis_TMA_5_01262022_003_aggr_transformed.ome.tiff"
# preIMS_file = "/home/retger/Downloads/sherborne/results/Lipid_TMA_3781/data/preIMS/Lipid_TMA_3781-preIMS_to_postIMS_registered_reduced.ome.tiff"
# preIMS_file =  "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMS/test_split_pre-preIMS_to_postIMS_registered_reduced.ome.tiff"
preIMS_file = snakemake.input["preIMS_downscaled"]
# resize factor to speedup computations
# rescale = 1
rescale = snakemake.params["downscale"]

output_df = snakemake.output["registration_metrics"]

logging.info("Setup rembg model")
# prepare model for rembg
model_name = "isnet-general-use"
os.environ["OMP_NUM_THREADS"] = str(snakemake.threads)
rembg_session = new_session(model_name)

logging.info("Setup sam model")
# prepare SAM
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cpu'
torch.set_num_threads(snakemake.threads)
MODEL_TYPE = "vit_h"

# download model from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# CHECKPOINT_PATH = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/Misc/sam_vit_h_4b8939.pth"
# CHECKPOINT_PATH = "/home/retger/Downloads/sherborne/results/Misc/sam_vit_h_4b8939.pth"
CHECKPOINT_PATH = snakemake.input["sam_weights"]

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
logging.info("postIMC bounding box extraction")
# read IMC to get bounding box (image was cropped in previous step)
postIMC = skimage.io.imread(postIMC_file)
rp = skimage.measure.regionprops((np.max(postIMC, axis=2)>0).astype(np.uint8))
bb1 = rp[0].bbox

logging.info("postIMC mask extraction")
# postIMC
postIMCw = readimage_crop(postIMC_file, bb1)
postIMCw = prepare_image_for_sam(postIMCw, rescale)
postIMC = np.stack([postIMCw, postIMCw, postIMCw], axis=2)
postIMCr = remove(postIMC, only_mask=True, session=rembg_session)
postIMCmasks = postIMCr>127
postIMCmasks = np.stack([skimage.morphology.remove_small_holes(postIMCmasks,100**2*np.pi*rescale)])

logging.info("preIMS mask extraction")
# preIMS
preIMSw = readimage_crop(preIMS_file, bb1)
preIMSw = prepare_image_for_sam(preIMSw, rescale)
preIMS = np.stack([preIMSw, preIMSw, preIMSw], axis=2)
preIMSr = remove(preIMS, only_mask=True, session=rembg_session)
preIMSmasks = preIMSr>127
preIMSmasks = np.stack([skimage.morphology.remove_small_holes(preIMSmasks,100**2*np.pi*rescale)])

logging.info("postIMS mask extraction")
# postIMS
postIMSw = readimage_crop(postIMS_file, bb1)
postIMSw = prepare_image_for_sam(postIMSw, rescale)
postIMS = postIMSw.copy()
tmpmask = skimage.morphology.isotropic_dilation(preIMSmasks[0], np.ceil(rescale*100))
postIMS[np.logical_not(tmpmask)] = 0
postIMS = skimage.filters.median(postIMS, skimage.morphology.disk(rescale * 5))
postIMS = normalize_image(postIMS)*255
postIMS = postIMS.astype(np.uint8)
postIMS = np.stack([postIMS, postIMS, postIMS], axis=2)
postIMSmasks, scores1 = sam_core(postIMS, sam)
postIMSmasks = np.stack([preprocess_mask(msk,rescale) for msk in postIMSmasks ])

postIMSmask_areas = [np.sum(postIMSmasks[i,:,:]>0) for i in range(3)]
preIMSmask_area = np.sum(preIMSmasks)
mask_diff = np.array(postIMSmask_areas) - preIMSmask_area
postIMSmasks = postIMSmasks[mask_diff==np.min(mask_diff),:,:]
logging.info(f"Min area difference postIMS-preIMS: {np.min(mask_diff)} (normalized by postIMS area: {np.min(mask_diff)/np.sum(postIMSmasks):6.4f})")

# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].imshow(postIMS[1000:1500,2000:2500])
# ax[0].set_title("postIMS")
# ax[1].imshow(postIMSmasks[0,1000:1500,2000:2500], cmap='gray')
# ax[1].set_title("postIMS mask")
# plt.show()


logging.info("Calculate overlap between preIMS and postIMS")
# get best overlap of masks
preIMSmask_to_postIMS, postIMSmask_to_preIMS, dice_score_postIMC_preIMS = get_max_dice_score(preIMSmasks, postIMSmasks)


# logging.info("second iteration postIMS mask extraction")
# tmppostIMSmask = np.logical_not(postIMSmask_to_preIMS)
# tmppostIMSmask = skimage.morphology.isotropic_erosion(tmppostIMSmask,100)
# postIMSwr = postIMSw.copy()
# postIMSwr[tmppostIMSmask[:,:,0]] = 0
# filt_realx, filt_imag = skimage.filters.gabor(postIMSwr, frequency=0.49)
# filt_realy, filt_imag = skimage.filters.gabor(postIMSwr, frequency=0.49, theta = np.pi/2)
# gabc = np.maximum(filt_realy,filt_realx)
# gabc = skimage.filters.median(gabc, skimage.morphology.disk(2))
# gabc = skimage.util.invert(gabc)
# postIMSgabc = np.stack([gabc.copy(), gabc.copy(), gabc.copy()], axis=2)
# si(postIMSwr)
# postIMSmasks, scores1 = sam_core(postIMSgabc, sam)

logging.info("Calculate dice coefficients")
preIMSmask, postIMSmask, dice_score_preIMS_postIMS = get_max_dice_score(preIMSmasks, postIMSmasks)
postIMCmask, preIMSmask, dice_score_postIMC_preIMS = get_max_dice_score(postIMCmasks, preIMSmasks)
postIMCmask, postIMSmask, dice_score_postIMC_postIMS = get_max_dice_score(postIMCmasks, postIMSmasks)

logging.info("Calculate Areas")
preIMSmask_area = np.sum(preIMSmask)*rescale
postIMCmask_area = np.sum(postIMCmask)*rescale
postIMSmask_area = np.sum(postIMSmask)*rescale

logging.info("Calculate ratio area")
postIMS_to_preIMS_area_ratio = postIMSmask_area/preIMSmask_area
preIMS_to_postIMC_area_ratio = preIMSmask_area/postIMCmask_area

logging.info("Calculate Proportions overlap")
# proportion non overlap postIMS
prop_postIMS_notin_postIMC = np.sum(np.logical_and(postIMSmask, np.logical_not(postIMCmask)))/np.sum(postIMSmask)
# proportion non overlap postIMC
prop_postIMC_notin_postIMS = np.sum(np.logical_and(postIMCmask, np.logical_not(postIMSmask)))/np.sum(postIMCmask)
# proportion non overlap postIMC
prop_postIMC_notin_preIMS = np.sum(np.logical_and(postIMCmask, np.logical_not(preIMSmask)))/np.sum(postIMCmask)
# proportion non overlap postIMC
prop_preIMS_notin_postIMS = np.sum(np.logical_and(preIMSmask, np.logical_not(postIMSmask)))/np.sum(preIMSmask)



logging.info("Calculate Centroids")

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].imshow(postIMSmask[:,:,0])
# ax[0].set_title("postIMS")
# ax[1].imshow(postIMSmaskm)
# ax[1].set_title("postIMC")
# plt.show()

disklen = 200
postIMSmaskm = smooth_mask(postIMSmask, disklen)
regpoppostIMS = skimage.measure.regionprops(postIMSmask.astype(np.uint8)[:,:,0],postIMSmaskm)
postIMScentw = regpoppostIMS[0].centroid_weighted
postIMScentw = (postIMScentw[0][0],postIMScentw[1][0])
postIMScent = regpoppostIMS[0].centroid

postIMCmaskm = smooth_mask(postIMCmask, disklen)
regpoppostIMC = skimage.measure.regionprops(postIMCmask.astype(np.uint8)[:,:,0],postIMCmaskm)
postIMCcentw = regpoppostIMC[0].centroid_weighted
postIMCcentw = (postIMCcentw[0][0],postIMCcentw[1][0])
postIMCcent = regpoppostIMC[0].centroid

preIMSmaskm = smooth_mask(preIMSmask, disklen)
regpoppreIMS = skimage.measure.regionprops(preIMSmask.astype(np.uint8)[:,:,0],preIMSmaskm)
preIMScentw = regpoppreIMS[0].centroid_weighted
preIMScentw = (preIMScentw[0][0],preIMScentw[1][0])
preIMScent = regpoppreIMS[0].centroid

logging.info("Calculate Mean error of centroids")
postIMS_to_preIMS_dist = dist_centroids(postIMScent, preIMScent, rescale)
postIMS_to_postIMC_dist = dist_centroids(postIMScent, postIMCcent, rescale)
preIMS_to_postIMC_dist = dist_centroids(preIMScent, postIMCcent, rescale)

postIMS_to_preIMS_dist_weighted = dist_centroids(postIMScentw, preIMScentw, rescale)
postIMS_to_postIMC_dist_weighted = dist_centroids(postIMScentw, postIMCcentw, rescale)
preIMS_to_postIMC_dist_weighted = dist_centroids(preIMScentw, postIMCcentw, rescale)
# t1 = np.logical_and(mask_image1, np.logical_not(mask_image2))
# t2 = np.logical_and(mask_image2, np.logical_not(mask_image1))


# regionprops(t1.astype(np.uint8))[0].centroid
# cent1
# regionprops(t2.astype(np.uint8))[0].centroid
# cent2

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].imshow(t1)
# ax[0].set_title("postIMS")
# ax[1].imshow(t2)
# ax[1].set_title("postIMC")
# plt.show()

logging.info("Create and save csv")
samplename = os.path.basename(postIMC_file).replace("_transformed.ome.tiff","")
df = pd.DataFrame(data = {
    'sample': samplename,
    'dice_score_postIMC_preIMS': dice_score_postIMC_preIMS,
    'dice_score_postIMC_postIMS': dice_score_postIMC_postIMS,
    'dice_score_preIMS_postIMS': dice_score_preIMS_postIMS,
    'postIMS_mask_area': postIMSmask_area, 
    'postIMC_mask_area': postIMCmask_area, 
    'preIMS_mask_area': preIMSmask_area, 
    'postIMS_to_preIMS_area_ratio': postIMS_to_preIMS_area_ratio,
    'preIMS_to_postIMC_area_ratio': preIMS_to_postIMC_area_ratio,
    'euclidean_distance_centroids_postIMS_to_preIMS': postIMS_to_preIMS_dist,
    'euclidean_distance_centroids_weighted_postIMS_to_preIMS': postIMS_to_preIMS_dist_weighted,
    'euclidean_distance_centroids_postIMS_to_postIMC': postIMS_to_postIMC_dist,
    'euclidean_distance_centroids_weighted_postIMS_to_postIMC': postIMS_to_postIMC_dist_weighted,
    'euclidean_distance_centroids_preIMS_to_postIMC': preIMS_to_postIMC_dist,
    'euclidean_distance_centroids_weighted_preIMS_to_postIMC': preIMS_to_postIMC_dist_weighted,
    'proportion_postIMS_mask_in_postIMC': prop_postIMS_notin_postIMC,
    'proportion_preIMS_mask_in_postIMS': prop_preIMS_notin_postIMS,
    'proportion_postIMC_mask_in_postIMS': prop_postIMC_notin_postIMS,
    'proportion_postIMC_mask_in_preIMS': prop_postIMC_notin_preIMS,
    }, index = [0])
df.to_csv(output_df, index=False)


logging.info("Create and save images")
tifffile.imwrite(snakemake.output['postIMS_postIMSmask'],np.stack([postIMSw,postIMSmask[:,:,0].astype(np.uint8)*255],axis=0))
tifffile.imwrite(snakemake.output['postIMC_postIMCmask'],np.stack([postIMCw,postIMCmask[:,:,0].astype(np.uint8)*255],axis=0))
tifffile.imwrite(snakemake.output['preIMS_preIMSmask'],np.stack([preIMSw,preIMSmask[:,:,0].astype(np.uint8)*255],axis=0))

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