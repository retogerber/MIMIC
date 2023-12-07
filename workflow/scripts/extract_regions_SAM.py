import matplotlib.pyplot as plt
import time
import numpy as np
import json
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import skimage
import skimage.exposure
import numpy as np
import cv2
from image_utils import readimage_crop, convert_and_scale_image, subtract_postIMS_grid, extract_mask, get_image_shape, sam_core, preprocess_mask
from utils import setNThreads, snakeMakeMock
import sys,os
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["pixel_expansion"] = 501
    snakemake.params["min_area"] = 24**2
    snakemake.params["max_area"] = 512**2
    snakemake.params["input_spacing"] = 0.22537
    snakemake.params["input_spacing_IMC_location"] = 0.22537
    snakemake.params["output_spacing"] = 1
    snakemake.params["remove_postIMS_grid"] = False
    snakemake.params["region_extraction_method"]="sam_prefilter"
    snakemake.input["sam_weights"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/Misc/sam_vit_h_4b8939.pth"
    snakemake.input["microscopy_image"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/test_split_pre_postIMC.ome.tiff"
    snakemake.input["IMC_location"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_location/test_split_pre_IMC_mask_on_postIMC_A1.geojson"
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
pixel_expansion = snakemake.params["pixel_expansion"]
min_area = snakemake.params["min_area"]
max_area = snakemake.params["max_area"]
input_spacing = snakemake.params["input_spacing"]
input_spacing_IMC_location = snakemake.params["input_spacing_IMC_location"]
output_spacing = snakemake.params["output_spacing"]
used_region_extraction_method = snakemake.params["region_extraction_method"]
assert(used_region_extraction_method in ["sam_prefilter","sam"])

# inputs
DEVICE = 'cpu'
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = snakemake.input["sam_weights"]
microscopy_file = snakemake.input['microscopy_image']
IMC_location=snakemake.input["IMC_location"]
if isinstance(IMC_location, list):
    IMC_location = IMC_location[0]

# output
contours_file_out = snakemake.output['contours_out']


logging.info("core bounding box extraction")
IMC_geojson = json.load(open(IMC_location, "r"))
if isinstance(IMC_geojson,list):
    IMC_geojson=IMC_geojson[0]
boundary_points = np.array(IMC_geojson['geometry']['coordinates'])[0,:,:]
xmin=np.min(boundary_points[:,1])
xmax=np.max(boundary_points[:,1])
ymin=np.min(boundary_points[:,0])
ymax=np.max(boundary_points[:,0])

s1f = input_spacing/input_spacing_IMC_location
bb1 = [int(xmin/s1f-pixel_expansion/input_spacing),int(ymin/s1f-pixel_expansion/input_spacing),int(xmax/s1f+pixel_expansion/input_spacing),int(ymax/s1f+pixel_expansion/input_spacing)]
imxmax, imymax, _ = get_image_shape(microscopy_file)
imxmax=int(imxmax/input_spacing)
imymax=int(imymax/input_spacing)
bb1[0] = bb1[0] if bb1[0]>=0 else 0
bb1[1] = bb1[1] if bb1[1]>=0 else 0
bb1[2] = bb1[2] if bb1[2]<=imxmax else imxmax
bb1[3] = bb1[3] if bb1[3]<=imymax else imymax
logging.info(f"bounding box whole image: {bb1}")

m2full_shape = get_image_shape(microscopy_file)
bb3 = [int(xmin/s1f-1251/input_spacing),int(ymin/s1f-1251/input_spacing),int(xmax/s1f+1251/input_spacing),int(ymax/s1f+1251/input_spacing)]
bb3[0] = bb3[0] if bb3[0]>=0 else 0
bb3[1] = bb3[1] if bb3[1]>=0 else 0
bb3[2] = bb3[2] if bb3[2]<=m2full_shape[0] else m2full_shape[0]
bb3[3] = bb3[3] if bb3[3]<=m2full_shape[1] else m2full_shape[1]
logging.info(f"bounding box mask whole image: {bb3}")


logging.info("load microscopy image")
# microscopy_image = readimage_crop(microscopy_file, bb1)
microscopy_image = readimage_crop(microscopy_file, bb3)
microscopy_image = convert_and_scale_image(microscopy_image, input_spacing/output_spacing)

logging.info("Extract mask for microscopy image")
mask_2 = extract_mask(microscopy_file, bb3, rescale = input_spacing/output_spacing, is_postIMS=False)[0,:,:]
xb = int((mask_2.shape[0]-microscopy_image.shape[0])/2)
yb = int((mask_2.shape[1]-microscopy_image.shape[1])/2)
wn = microscopy_image.shape[0]
hn = microscopy_image.shape[1]
assert(xb>=0)
assert(yb>=0)
if xb==0 and yb==0:
    mask_2 = cv2.resize(mask_2.astype(np.uint8), (hn,wn), interpolation=cv2.INTER_NEAREST)
elif xb==0:
    mask_2 = cv2.resize(mask_2[:,yb:-yb].astype(np.uint8), (hn,wn), interpolation=cv2.INTER_NEAREST)
elif yb==0:
    mask_2 = cv2.resize(mask_2[xb:-xb,:].astype(np.uint8), (hn,wn), interpolation=cv2.INTER_NEAREST)
else:
    mask_2 = cv2.resize(mask_2[xb:-xb,yb:-yb].astype(np.uint8), (hn,wn), interpolation=cv2.INTER_NEAREST)
mask_2_proportion = np.sum(mask_2)/np.prod(mask_2.shape)
logging.info(f"proportion image covered by mask (rembg): {mask_2_proportion:5.4}")

bb0 = [int(xmin/s1f),int(ymin/s1f),int(xmax/s1f),int(ymax/s1f)]
IMC_mask_proportion = ((bb0[2]-bb0[0])*(bb0[3]-bb0[1]))/((bb3[2]-bb3[0])*(bb3[3]-bb3[1]))
if mask_2_proportion < 1.5*IMC_mask_proportion:
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    saminp = readimage_crop(microscopy_file, bb3)
    # to gray scale, rescale
    saminp = convert_and_scale_image(saminp, input_spacing/output_spacing)
    saminp = np.stack([saminp, saminp, saminp], axis=2)
    # run SAM segmentation model
    masks, scores1 = sam_core(saminp, sam)
    # postprocess
    masks = np.stack([skimage.morphology.convex_hull_image(preprocess_mask(msk,1)) for msk in masks ])
    tmpareas = np.array([np.sum(im) for im in masks])
    indmax = np.argmax(tmpareas/(masks.shape[1]*masks.shape[2]))
    mask_2 = masks[indmax,:,:].astype(np.uint8)
    mask_2_proportion = np.sum(mask_2)/np.prod(mask_2.shape)
    logging.info(f"proportion image covered by mask (SAM): {mask_2_proportion:5.4}")
    if mask_2_proportion < 1.5*IMC_mask_proportion:
        mask_2 = np.ones(microscopy_image.shape, dtype=np.uint8)
        logging.info(f"Mask detection failed, using full image and 'region_extraction_method'='sam'")
        used_region_extraction_method='sam'

xmax = microscopy_image.shape[0]
ymax = microscopy_image.shape[1]
imcbbox_outer = [0,0,xmax,ymax]
logging.info(f"imcbbox_outer: {imcbbox_outer}")

logging.info(f"snakemake params remove_postIMS_grid: {snakemake.params['remove_postIMS_grid']}")
if snakemake.params["remove_postIMS_grid"]:
    img = microscopy_image[imcbbox_outer[0]:imcbbox_outer[2],imcbbox_outer[1]:imcbbox_outer[3]].copy()
    mask_2_on_2 = cv2.resize(mask_2, (microscopy_image.shape[1], microscopy_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.morphologyEx(src=mask_2_on_2, dst=mask_2_on_2, op = cv2.MORPH_DILATE, kernel = skimage.morphology.square(20))
    img[mask_2_on_2==0]=0
    out = subtract_postIMS_grid(img)
    out[mask_2_on_2==0]=0
    microscopy_image[imcbbox_outer[0]:imcbbox_outer[2],imcbbox_outer[1]:imcbbox_outer[3]] = out 

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)


def get_regions_sam_prefilter(img, mask, sam, min_area=24**2,max_area=512**2, quantile=0.05, dilation_kernel_size=20, opening_kernel_radius=5, erosion_kernel_size=50):

    # expand mask
    mask_exp = cv2.morphologyEx(src=mask.astype(np.uint8), op = cv2.MORPH_DILATE, kernel = skimage.morphology.square(dilation_kernel_size))
    mbb = cv2.boundingRect(mask_exp.astype(np.uint8))
    # get background pixels (outside of expanded mask)
    background_pixels = img[mbb[1]:mbb[1]+mbb[3],mbb[0]:mbb[0]+mbb[2]][mask_exp[mbb[1]:mbb[1]+mbb[3],mbb[0]:mbb[0]+mbb[2]]==0]

    # assume most pixels selected are actually background
    thr=np.quantile(background_pixels, quantile)
    thrimg = img>thr
    # remove small objects
    thrimg2 = cv2.morphologyEx(src=thrimg.astype(np.uint8), op = cv2.MORPH_OPEN, kernel = skimage.morphology.disk(opening_kernel_radius))
    # erode mask
    mask_ixp = cv2.morphologyEx(src=mask.astype(np.uint8), op = cv2.MORPH_ERODE, kernel = skimage.morphology.square(erosion_kernel_size))
    # remove objects outside of mask
    thrimg2[mask_ixp==0]=0
    # get connected components
    _,_,stats,centroids = cv2.connectedComponentsWithStats(thrimg2.astype(np.uint8), connectivity=8)
    # filter by size
    to_keep = np.logical_and(stats[:,4] > min_area/2, stats[:,4] < max_area)
    centroids_filt = centroids[to_keep,:]

    predictor = SamPredictor(sam)
    img_stacked = np.stack([img, img, img], axis=2)
    predictor.set_image(img_stacked)

    # loop over centroids
    regions=list()
    bboxes=list()
    for p in range(len(centroids_filt)):
        masks, _, _ = predictor.predict(
            point_coords=centroids_filt[p,:].reshape(1,2),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        if np.sum(masks[0])>min_area and np.sum(masks[0])<max_area:
            cts,_ = cv2.findContours(masks[0].astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for ct in cts:
                ca = cv2.contourArea(ct) 
                if ca>=min_area and ca<=max_area:
                    ct = ct.reshape(-1,2)
                    bboxes.append(cv2.boundingRect(ct))
                    regions.append(ct)
    return regions, bboxes


def get_regions_sam(img, sam, min_area=24**2,max_area=512**2):
    points_per_side = int(np.ceil(np.max(img.shape)/np.sqrt(min_area)))
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=points_per_side, stability_score_thresh=0.9, pred_iou_thresh=0.8)
    img_stacked = np.stack([img, img, img], axis=2)
    t1=time.time()
    masks = mask_generator.generate(img_stacked)
    t2=time.time()
    logging.info(f"Time for SAM: {t2-t1}")

    bboxes=list()
    regions=list()
    for mask in masks:
        if mask['area']>=min_area and mask['area']<=max_area:
            cts,_ = cv2.findContours(mask['segmentation'].astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for ct in cts:
                ca = cv2.contourArea(ct) 
                if ca>=min_area and ca<=max_area:
                    ct = ct.reshape(-1,2)
                    bboxes.append(cv2.boundingRect(ct))
                    regions.append(ct)
    return regions, bboxes

logging.info("Extract regions")

# crop to IMC expanded location
bb31 = [bb1[0]-bb3[0],bb1[1]-bb3[1],bb1[2]-bb3[0],bb1[3]-bb3[1]]
s1f = input_spacing/output_spacing
bb31 = [round(bb31[0]*s1f),round(bb31[1]*s1f),round(bb31[2]*s1f),round(bb31[3]*s1f)]

# microscopy_image2 = readimage_crop(microscopy_file, bb1)
# microscopy_image2 = convert_and_scale_image(microscopy_image2, input_spacing/output_spacing)

# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].imshow(microscopy_image[bb31[0]:bb31[2],bb31[1]:bb31[3]])
# ax[1].imshow(microscopy_image2)
# plt.show()



if used_region_extraction_method == "sam_prefilter":
    regions, bboxes = get_regions_sam_prefilter(microscopy_image[bb31[0]:bb31[2],bb31[1]:bb31[3]], mask_2[bb31[0]:bb31[2],bb31[1]:bb31[3]], sam, min_area=min_area, max_area=max_area)
else:
    regions, bboxes = get_regions_sam(microscopy_image[bb31[0]:bb31[2],bb31[1]:bb31[3]], sam, min_area=min_area, max_area=max_area)

# arrowed_microscopy_image_1 = np.stack([microscopy_image, microscopy_image, microscopy_image], axis=2)
# for k in range(len(regions)):
#     arrowed_microscopy_image_1 = cv2.drawContours(
#         arrowed_microscopy_image_1, 
#         [regions[k]], 
#         -1, 
#         255,
#         -1)
# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].imshow(arrowed_microscopy_image_1)
# ax[1].imshow(microscopy_image, cmap='gray')
# plt.show()



logging.info("Save regions")
# save regions
regionsls = [reg.tolist() for reg in regions]
with open(contours_file_out, 'w') as f:
    json.dump({'regions': regionsls, 'bboxes': bboxes}, f)


logging.info("Finished")