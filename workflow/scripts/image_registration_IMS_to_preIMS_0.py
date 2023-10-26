from rembg import remove, new_session
from segment_anything import sam_model_registry, SamPredictor
import torch
import cv2
import json
import skimage
import numpy as np
from image_registration_IMS_to_preIMS_utils import *
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
torch.set_num_threads(snakemake.threads)
cv2.setNumThreads(snakemake.threads)

logging.info("Setup rembg")
model_name = "isnet-general-use"
os.environ["OMP_NUM_THREADS"] = str(snakemake.threads)
rembg_session = new_session(model_name)

logging.info("Setup segment anything")
CHECKPOINT_PATH = snakemake.input["sam_weights"]
DEVICE = 'cpu'
MODEL_TYPE = "vit_h"


# parameters
stepsize = float(snakemake.params["IMS_pixelsize"])
pixelsize = stepsize*float(snakemake.params["IMS_shrink_factor"])
resolution = float(snakemake.params["microscopy_pixelsize"])

postIMS_file = snakemake.input["postIMS_downscaled"]
postIMSr_file = snakemake.output["postIMSmask_downscaled"]

imc_mask_files = snakemake.input["IMCmask"]

# upscale in all directions from IMC location
# if IMC is 1000x1000 microns, adding 1350 (microns) to all directions
# should include whole TMA of around 2.5 mm in diameter
expand_microns = 1350

logging.info("Input:")
logging.info(f"\tstepsize: {stepsize}")
logging.info(f"\tpixelsize: {pixelsize}")
logging.info(f"\tresolution: {resolution}")
logging.info(f"\tpostIMS_file: {postIMS_file}")
logging.info(f"\tpostIMSr_file: {postIMSr_file}")
logging.info(f"\timc_mask_files: {imc_mask_files}")
logging.info(f"\texpand_microns: {expand_microns}")


logging.info("get IMC locations")
imcbboxls = list()
for imcmaskfile in imc_mask_files:
    IMC_geojson = json.load(open(imcmaskfile, "r"))
    if isinstance(IMC_geojson,list):
        IMC_geojson=IMC_geojson[0]
    boundary_points = np.array(IMC_geojson['geometry']['coordinates'])[0,:,:]
    xmin=int(np.min(boundary_points[:,1]))
    xmax=int(np.max(boundary_points[:,1]))
    ymin=int(np.min(boundary_points[:,0]))
    ymax=int(np.max(boundary_points[:,0]))
    bbox = np.array([xmin,ymin,xmax,ymax])
    if resolution != 1:
        bbox = (bbox*resolution).astype(int)
    logging.info(f"    {bbox}:{imcmaskfile}")
    imcbboxls.append(bbox)

def extract_mask(img: np.ndarray, session, rescale: float):
    """
    extract postIMS tissue location using rembg
    """
    # to grayscale, rescale
    w = prepare_image_for_sam(img, rescale)
    # using fft, remove IMS ablation grid
    #w = subtract_postIMS_grid(w)
    w = cv2.blur(w, (5,5))
    w = np.stack([w, w, w], axis=2)
    # detect background
    wr = remove(w, only_mask=True, session=session)
    # threshold
    tmpimg = wr>127
    # upscale to not touch border, remove small holes
    tmpimg2 = np.zeros((tmpimg.shape[0]+10,tmpimg.shape[1]+10))
    tmpimg2[5:(tmpimg.shape[0]+5),5:(tmpimg.shape[1]+5)] = tmpimg
    tmpimg = skimage.morphology.remove_small_holes(tmpimg2>0,500**2*np.pi*(1/resolution))[5:(tmpimg.shape[0]+5),5:(tmpimg.shape[1]+5)]
    tmpimg = tmpimg.astype(np.uint8)*255
    tmpimg = preprocess_mask(tmpimg,1)
    tmpimg = tmpimg.astype(np.uint8)*255
    tmpimg2 = np.zeros(np.array(list(tmpimg.shape))+20, dtype=np.uint8)
    tmpimg2[10:(tmpimg2.shape[0]-10),10:(tmpimg2.shape[1]-10)] = tmpimg.astype(np.uint8)
    # upscale to not touch border, get convex hull
    tmpimg2 = skimage.morphology.convex_hull_image(tmpimg2>127)
    tmpimg2 = tmpimg2.astype(np.uint8)*255
    tmpimg = tmpimg2[10:(tmpimg2.shape[0]-10),10:(tmpimg2.shape[1]-10)]
    # morphological closing
    tmpimg = cv2.morphologyEx(tmpimg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5,5),np.uint8)).astype(bool)
    return tmpimg 

logging.info("Remove background individually for each IMC location with rembg")
postIMS_shape = (np.array(get_image_shape(postIMS_file)[:2])*resolution).astype(int)
postIMSstitch = np.zeros(postIMS_shape)
rembg_mask_areas = []
for i in range(len(imcbboxls)):
    xmin = int(imcbboxls[i][0]-expand_microns)
    xmin = xmin if xmin>=0 else 0
    ymin = int(imcbboxls[i][1]-expand_microns)
    ymin = ymin if ymin>=0 else 0
    xmax = int(imcbboxls[i][2]+expand_microns)
    xmax = xmax if xmax<=postIMS_shape[0] else postIMS_shape[0]
    ymax = int(imcbboxls[i][3]+expand_microns)
    ymax = ymax if ymax<=postIMS_shape[1] else postIMS_shape[1]
    print(f"i: {i}, {os.path.basename(imc_mask_files[i])}, coords:[{xmin}:{xmax},{ymin}:{ymax}]")
    tmpinpimg = readimage_crop(postIMS_file, (np.ceil(np.array([xmin,ymin,xmax,ymax])/resolution)).astype(int))
    tmpimg = extract_mask(tmpinpimg, rembg_session, resolution)
    rembg_mask_areas.append(np.sum(tmpimg>0))
    wn, hn = postIMSstitch[xmin:xmax,ymin:ymax].shape
    if tmpimg[:wn,:hn].shape[0]-wn > -2 and tmpimg[:wn,:hn].shape[1]-hn > -2:
        tmpimg = cv2.resize(tmpimg[:wn,:hn].astype(np.uint8), (hn,wn), interpolation=cv2.INTER_NEAREST).astype(bool)
    elif tmpimg[:wn,:hn].shape[0]-wn < -1 and tmpimg[:wn,:hn].shape[1]-hn < -1:
        raise ValueError("shapes are too different!")
    else:
        tmpimg = tmpimg[:wn,:hn]
    logging.info(f"\tshape 1: {postIMSstitch[xmin:xmax,ymin:ymax].shape}")
    logging.info(f"\tshape 2: {tmpimg.shape}")
    postIMSstitch[xmin:xmax,ymin:ymax] = np.max(np.stack([postIMSstitch[xmin:xmax,ymin:ymax],tmpimg], axis=0),axis=0)

# threshold
postIMSrs = postIMSstitch>0

logging.info("Check mask")
IMCrs_filled = list()
for i in range(len(imcbboxls)):
    xmin = int(imcbboxls[i][0])
    ymin = int(imcbboxls[i][1])
    xmax = int(imcbboxls[i][2])
    ymax = int(imcbboxls[i][3])
    IMCrs_filled.append(np.min(postIMSrs[xmin:xmax,ymin:ymax]) == 1)

postIMSstitch = np.zeros(postIMS_shape)
inds = np.array(list(range(len(imcbboxls))))

logging.info("Run segment anything model on IMC locations")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_mask_areas = []
for i in inds:
    # bounding box
    xmin = int(imcbboxls[i][0]-expand_microns)
    xmin = xmin if xmin>=0 else 0
    ymin = int(imcbboxls[i][1]-expand_microns)
    ymin = ymin if ymin>=0 else 0
    xmax = int(imcbboxls[i][2]+expand_microns)
    xmax = xmax if xmax<=postIMS_shape[0] else postIMS_shape[0]
    ymax = int(imcbboxls[i][3]+expand_microns)
    ymax = ymax if ymax<=postIMS_shape[1] else postIMS_shape[1]
    logging.info(f"i: {i}, {os.path.basename(imc_mask_files[i])}, coords:[{xmin}:{xmax},{ymin}:{ymax}]")

    # read image
    saminp = readimage_crop(postIMS_file, (np.ceil(np.array([xmin,ymin,xmax,ymax])/resolution)).astype(int))
    # to gray scale, rescale
    saminp = prepare_image_for_sam(saminp, resolution)
    # remove postIMS IMS ablation grid
    #saminp = subtract_postIMS_grid(saminp)
    saminp = np.stack([saminp, saminp, saminp], axis=2)
    # run SAM segmentation model
    postIMSmasks, scores1 = sam_core(saminp, sam)
    # postprocess
    postIMSmasks = np.stack([preprocess_mask(msk,1) for msk in postIMSmasks ])
    tmpareas = np.array([np.sum(im) for im in postIMSmasks])
    imcarea = (imcbboxls[i][2]-imcbboxls[i][0])*(imcbboxls[i][3]-imcbboxls[i][1])
    tmpinds = np.array(list(range(3)))
    tmpinds = tmpinds[tmpareas > imcarea*1.02]
    tmpind = tmpinds[scores1[tmpinds]==np.max(scores1[tmpinds])]
    tmpimg = postIMSmasks[tmpind,:,:][0,:,:].astype(np.uint8)*255
    sam_mask_areas.append(np.sum(tmpimg>0))
    wn, hn = postIMSstitch[xmin:xmax,ymin:ymax].shape
    if tmpimg[:wn,:hn].shape[0]-wn > -2 and tmpimg[:wn,:hn].shape[1]-hn > -2:
        tmpimg = cv2.resize(tmpimg[:wn,:hn].astype(np.uint8), (hn,wn), interpolation=cv2.INTER_NEAREST).astype(bool)
    elif tmpimg[:wn,:hn].shape[0]-wn < -1 and tmpimg[:wn,:hn].shape[1]-hn < -1:
        raise ValueError("shapes are too different!")
    else:
        tmpimg = tmpimg[:wn,:hn]
    logging.info(f"\tshape 1: {postIMSstitch[xmin:xmax,ymin:ymax].shape}")
    logging.info(f"\tshape 2: {tmpimg.shape}")
    postIMSstitch[xmin:xmax,ymin:ymax] = np.max(np.stack([postIMSstitch[xmin:xmax,ymin:ymax],tmpimg], axis=0),axis=0)

postIMSsamr = postIMSstitch>0

logging.info("Check mask")
IMCrsam_filled = list()
for i in range(len(imcbboxls)):
    xmin = int(imcbboxls[i][0])
    ymin = int(imcbboxls[i][1])
    xmax = int(imcbboxls[i][2])
    ymax = int(imcbboxls[i][3])
    IMCrsam_filled.append(np.min(postIMSsamr[xmin:xmax,ymin:ymax]) == 1)

tmpbool = np.logical_not(np.logical_and(IMCrs_filled,IMCrsam_filled))
inds = np.array(list(range(len(imcbboxls))))[tmpbool]
if len(inds)>0:
    logging.info(f"The following masks were not found in both rembg and sam:")
    for i in inds:
        logging.info(f"{os.path.basename(imc_mask_files[i])};\trembg: {IMCrs_filled[i]};\tSAM:{IMCrsam_filled[i]}")

ratio_sam_to_rembg = [np.log10(sam_mask_areas[i]/rembg_mask_areas[i]) for i in range(len(imcbboxls))]
logging.info(f"Difference of the mask areas:")
logging.info(f"SAM-rembg\t,(SAM-rembg)/(0.5*(SAM+rembg))\t,Name")
for i in range(len(imcbboxls)):
    logging.info(f"{sam_mask_areas[i]-rembg_mask_areas[i]}\t, {(sam_mask_areas[i]-rembg_mask_areas[i])/(0.5*(sam_mask_areas[i]+rembg_mask_areas[i])):.4f}\t, {os.path.basename(imc_mask_files[i])}")
    # if (ratio_sam_to_rembg[i]<np.log10(1/1.1)) or (ratio_sam_to_rembg[i]>np.log10(1.1/1)):
    # logging.info(f"Difference of the mask areas (SAM-rembg) for {os.path.basename(imc_mask_files[i])}: {sam_mask_areas[i]-rembg_mask_areas[i]} ((SAM-rembg)/(0.5*(SAM+rembg)): {(sam_mask_areas[i]-rembg_mask_areas[i])/(0.5*(sam_mask_areas[i]+rembg_mask_areas[i])):.4f}")




logging.info(f"Get convex hull")
lbs = skimage.measure.label(postIMSsamr)
rps = skimage.measure.regionprops(lbs)
cvi = lbs*0
for i in range(len(rps)):
    tbb = rps[i].bbox
    ti = skimage.morphology.convex_hull_image(lbs[tbb[0]:tbb[2],tbb[1]:tbb[3]]==rps[i].label)
    cvi[tbb[0]:tbb[2],tbb[1]:tbb[3]] = np.logical_or(ti,cvi[tbb[0]:tbb[2],tbb[1]:tbb[3]])
logging.info("Save mask")

# rescale
if resolution != 1:
    wn, hn = get_image_shape(postIMS_file)[:2]
    cvi = cv2.resize(cvi, (hn,wn), interpolation=cv2.INTER_NEAREST)

saveimage_tile(cvi, postIMSr_file, resolution)
logging.info("Finished")
