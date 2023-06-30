import pandas as pd
import os
# import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import torch
from rembg import remove, new_session
import skimage
import numpy as np
import tifffile
import zarr
from wsireg.utils.im_utils import grayscale

# image_file1 ="/home/retger/Downloads/QuPath-0.4.1-Linux/Projects/cirrhosis_TMA/cirrhosis_TMA-postIMS_registered.ome.tiff"
# postIMS_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMS/test_split_pre_postIMS_reduced.ome.tiff"
postIMS_file = snakemake.input["postIMS_downscaled"]
# image_file2 ="/home/retger/Downloads/QuPath-0.4.1-Linux/Projects/cirrhosis_TMA/cirrhosis_TMA-postIMC_to_postIMS_registered.ome.tiff"
# postIMC_file="/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/Cirrhosis-TMA-5_New_Detector_001_transformed.ome.tiff"
postIMC_file = snakemake.input["postIMC_transformed"]
# imc_mask1 ="/home/retger/Downloads/QuPath-0.4.1-Linux/Projects/cirrhosis_TMA/Cirrhosis_TMA_5_01262022_004_aggr_transformed.ome.tiff"
# imc_mask1="/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_001_transformed.ome.tiff"
# imc_mask2 ="/home/retger/Downloads/QuPath-0.4.1-Linux/Projects/cirrhosis_TMA/Cirrhosis_TMA_5_01262022_003_aggr_transformed.ome.tiff"
preIMS_file =  "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMS/test_split_pre-preIMS_to_postIMS_registered_reduced.ome.tiff"
preIMS_file = snakemake.input["preIMS_downscaled"]
# resize factor to speedup computations
# rescale = 1
rescale = snakemake.params["downscale"]

output_df = snakemake.output["registration_metrics"]

# prepare model for rembg
model_name = "isnet-general-use"
rembg_session = new_session(model_name)

# prepare SAM
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cpu'
torch.set_num_threads(snakemake.threads)
MODEL_TYPE = "vit_h"

# download model from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# CHECKPOINT_PATH = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/Misc/sam_vit_h_4b8939.pth"
CHECKPOINT_PATH = snakemake.input["sam_weights"]

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)

def normalize_image(image):
    return (image-np.nanmin(image))/(np.nanmax(image)- np.nanmin(image))

def readimage_crop(image: str, bbox: list[int]):
    store = tifffile.imread(image, aszarr=True)
    z = zarr.open(store, mode='r')
    image_crop = z[0][bbox[0]:bbox[2],bbox[1]:bbox[3],:]
    return image_crop

def prepare_image_for_sam(image: np.ndarray, scale): 
    img = grayscale(image, True)
    img = skimage.transform.rescale(img, scale, preserve_range = True)   
    img = (img-np.nanmin(img))/(np.nanmax(img)- np.nanmin(img))
    img = skimage.exposure.equalize_adapthist(img)
    img = normalize_image(img)*255
    img = img.astype(np.uint8)
    return img

def apply_filter(image: np.ndarray):
    img = skimage.filters.sobel(image)
    img = normalize_image(img)*255
    img = img.astype(np.uint8)
    img = np.stack([img, img, img], axis=2)
    return img


def preprocess_mask(mask: np.ndarray, image_resolution):
    mask1tmp = skimage.morphology.isotropic_opening(mask, np.ceil(image_resolution*5))
    mask1tmp, count = skimage.measure.label(mask1tmp, connectivity=2, return_num=True)
    counts = np.unique(mask1tmp, return_counts = True)
    countsred = counts[1][counts[0] > 0]
    indsred = counts[0][counts[0] > 0]
    maxind = indsred[countsred == np.max(countsred)][0]
    mask1tmp = mask1tmp == maxind
    return mask1tmp


def sam_core(img: np.ndarray, sam):
    predictor = SamPredictor(sam)
    predictor.set_image(img)

    input_points = np.array([
        [img.shape[0]//2,img.shape[1]//2]
        ])
    input_labels = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )
    return masks, scores

def get_max_dice_score(masks1, masks2):
    dice_scores = np.zeros((masks1.shape[0],masks2.shape[1]))
    for i in range(masks1.shape[0]):
        for j in range(masks2.shape[0]):
            mask_image1 = masks1[i,:,:]
            mask_image2 = masks2[j,:,:]

            union = np.logical_and(mask_image1,mask_image2)
            dice_score = (2*np.sum(union))/(np.sum(mask_image1)+np.sum(mask_image2))
            dice_scores[i,j] = dice_score

    indices = np.where(dice_scores == dice_scores.max())
    h, w = masks1.shape[-2:]
    mask_image1 = masks1[indices[0][0]].reshape(h,w,1)
    h, w = masks2.shape[-2:]
    mask_image2 = masks2[indices[1][0]].reshape(h,w,1)

    return mask_image1, mask_image2, dice_scores[indices[0][0],indices[1][0]]

# read IMC to get bounding box (image was cropped in previous step)
postIMC = skimage.io.imread(postIMC_file)
rp = skimage.measure.regionprops((np.max(postIMC, axis=2)>0).astype(np.uint8))
bb1 = rp[0].bbox


# postIMS
postIMSw = readimage_crop(postIMS_file, bb1)
postIMSw = prepare_image_for_sam(postIMSw, rescale)
postIMS = apply_filter(postIMSw)
postIMSmasks, scores1 = sam_core(postIMS, sam)
postIMSmasks = np.stack([preprocess_mask(msk,rescale) for msk in postIMSmasks ])

# preIMS
preIMSw = readimage_crop(preIMS_file, bb1)
preIMSw = prepare_image_for_sam(preIMSw, rescale)
preIMS = np.stack([preIMSw, preIMSw, preIMSw], axis=2)
preIMSr = remove(preIMS, only_mask=True, session=rembg_session)
preIMSmasks = np.stack([preIMSr>127])

# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].imshow(img3)
# ax[0].set_title("postIMS")
# ax[1].imshow(masks3[0,:,:], cmap='gray')
# ax[1].set_title("postIMS mask")
# plt.show()


# postIMC
postIMCw = readimage_crop(postIMC_file, bb1)
postIMCw = prepare_image_for_sam(postIMCw, rescale)
postIMC = np.stack([postIMCw, postIMCw, postIMCw], axis=2)
postIMCr = remove(postIMC, only_mask=True, session=rembg_session)
postIMCmasks = np.stack([postIMCr>127])

# get best overlap of masks
preIMSmask_to_postIMS, postIMSmask_to_preIMS, dice_score_postIMC_preIMS = get_max_dice_score(preIMSmasks, postIMSmasks)


tmppostIMSmask = np.logical_not(postIMSmask_to_preIMS)
tmppostIMSmask = skimage.morphology.isotropic_erosion(tmppostIMSmask,100)
postIMSwr = postIMSw.copy()
postIMSwr[tmppostIMSmask[:,:,0]] = 0
filt_realx, filt_imag = skimage.filters.gabor(postIMSwr, frequency=0.49)
filt_realy, filt_imag = skimage.filters.gabor(postIMSwr, frequency=0.49, theta = np.pi/2)
gabc = np.maximum(filt_realy,filt_realx)
gabc = skimage.filters.median(gabc, skimage.morphology.disk(2))
gabc = skimage.util.invert(gabc)
postIMSgabc = np.stack([gabc.copy(), gabc.copy(), gabc.copy()], axis=2)
postIMSmasks, scores1 = sam_core(postIMSgabc, sam)

preIMSmask, postIMSmask, dice_score_preIMS_postIMS = get_max_dice_score(preIMSmasks, postIMSmasks)
postIMCmask, preIMSmask, dice_score_postIMC_preIMS = get_max_dice_score(postIMCmasks, preIMSmasks)
postIMCmask, postIMSmask, dice_score_postIMC_postIMS = get_max_dice_score(postIMCmasks, postIMSmasks)

preIMSmask_area = np.sum(preIMSmask)*rescale
postIMCmask_area = np.sum(postIMCmask)*rescale
postIMSmask_area = np.sum(postIMSmask)*rescale


# proportion non overlap postIMS
prop_postIMS_notin_postIMC = np.sum(np.logical_and(postIMSmask, np.logical_not(postIMCmask)))/np.sum(postIMSmask)
# proportion non overlap postIMC
prop_postIMC_notin_postIMS = np.sum(np.logical_and(postIMCmask, np.logical_not(postIMSmask)))/np.sum(postIMCmask)
# proportion non overlap postIMC
prop_postIMC_notin_preIMS = np.sum(np.logical_and(postIMCmask, np.logical_not(preIMSmask)))/np.sum(postIMCmask)
# proportion non overlap postIMC
prop_preIMS_notin_postIMS = np.sum(np.logical_and(preIMSmask, np.logical_not(postIMSmask)))/np.sum(preIMSmask)


# proportion_ratio = prop_postIMS_notin_postIMC / prop_postIMC_notin_postIMS

postIMScent = skimage.measure.regionprops(postIMSmask.astype(np.uint8))[0].centroid
postIMCcent = skimage.measure.regionprops(postIMCmask.astype(np.uint8))[0].centroid
preIMScent = skimage.measure.regionprops(preIMSmask.astype(np.uint8))[0].centroid
def dist_centroids(cent1, cent2, rescale):
    euclid_dist_pixel = ((cent1[0]-cent2[0])**2 + (cent1[1]-cent2[1])**2)**0.5
    euclid_dist = euclid_dist_pixel*rescale
    return euclid_dist

postIMS_to_preIMS_dist = dist_centroids(postIMScent, preIMScent, rescale)
postIMS_to_postIMC_dist = dist_centroids(postIMScent, postIMCcent, rescale)
preIMS_to_postIMC_dist = dist_centroids(preIMScent, postIMCcent, rescale)


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

samplename = os.path.basename(postIMC_file).replace("_transformed.ome.tiff","")

df = pd.DataFrame(data = {
    'sample': samplename,
    'dice_score_postIMC_preIMS': dice_score_postIMC_preIMS,
    'dice_score_postIMC_postIMS': dice_score_postIMC_postIMS,
    'dice_score_preIMS_postIMS': dice_score_preIMS_postIMS,
    'postIMS_mask_area': postIMSmask_area, 
    'postIMC_mask_area': postIMCmask_area, 
    'preIMS_mask_area': preIMSmask_area, 
    'euclidean_distance_centroids_postIMS_to_preIMS': postIMS_to_preIMS_dist,
    'euclidean_distance_centroids_postIMS_to_postIMC': postIMS_to_postIMC_dist,
    'euclidean_distance_centroids_preIMS_to_postIMC': preIMS_to_postIMC_dist,
    'proportion_postIMS_mask_in_postIMC': prop_postIMS_notin_postIMC,
    'proportion_preIMS_mask_in_postIMS': prop_preIMS_notin_postIMS,
    'proportion_postIMC_mask_in_postIMS': prop_postIMC_notin_postIMS,
    'proportion_postIMC_mask_in_preIMS': prop_postIMC_notin_preIMS,
    }, index = [0])
df.to_csv(output_df)


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

