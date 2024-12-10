import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
from rembg import new_session
from segment_anything import sam_model_registry
from ome_types import from_tiff
import torch
import cv2
import json
import skimage
import numpy as np
from image_utils import convert_and_scale_image, get_image_shape, extract_mask, readimage_crop, sam_core, saveimage_tile, preprocess_mask,get_pyr_levels
from utils import setNThreads, snakeMakeMock
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()

    snakemake.params["IMS_pixelsize"] = 30
    snakemake.params["IMS_shrink_factor"] = 0.8
    snakemake.params["microscopy_pixelsize"] = 0.22537
    snakemake.params["rescale"] = 2
    snakemake.params["postIMSmask_extraction_constraint"] = "min_preIMS"
    snakemake.params["postIMSmask_extraction_constraint_parameter"] = 5
    snakemake.input["sam_weights"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/Misc/sam_vit_h_4b8939.pth"
    snakemake.input["postIMS_downscaled"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/cirrhosis_TMA/data/postIMS/cirrhosis_TMA_postIMS_reduced.ome.tiff"
    snakemake.input["preIMS_downscaled"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/cirrhosis_TMA/data/preIMS/cirrhosis_TMA-preIMS_to_postIMS_registered.ome.tiff"
    snakemake.input["IMCmask"] = ["/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMC_location/cirrhosis_TMA_IMC_mask_on_postIMS_E1.geojson","/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMC_location/cirrhosis_TMA_IMC_mask_on_postIMS_E2.geojson"]
    snakemake.output["postIMSmask_downscaled"] = ""

    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# parameters
stepsize = float(snakemake.params["IMS_pixelsize"])
pixelsize = stepsize*float(snakemake.params["IMS_shrink_factor"])
resolution = float(snakemake.params["microscopy_pixelsize"])
rescale = float(snakemake.params["compute_rescale"])
out_rescale = float(snakemake.params["out_rescale"])
# upscale in all directions from TMA location 
expand_microns = stepsize*15
postIMSmask_extraction_constraint = snakemake.params["postIMSmask_extraction_constraint"]
postIMSmask_extraction_constraint = "none"
postIMSmask_extraction_constraint_parameter = snakemake.params["postIMSmask_extraction_constraint_parameter"]
postIMSmask_extraction_constraint_parameter = 0

# inputs
postIMS_file = snakemake.input["postIMS_downscaled"]
preIMS_file = snakemake.input["preIMS_downscaled"]
imc_mask_files = snakemake.input["IMCmask"]

model_name = "isnet-general-use"
os.environ["OMP_NUM_THREADS"] = str(snakemake.threads)
rembg_session = new_session(model_name)
CHECKPOINT_PATH = snakemake.input["sam_weights"]

DEVICE = 'cpu'
MODEL_TYPE = "vit_h"

# outputs
postIMSr_file = snakemake.output["postIMSmask_downscaled"]
preIMSr_file = snakemake.output["postIMSmask_downscaled"].replace("postIMS_reduced_mask.ome.tiff","preIMS_reduced_mask.ome.tiff").replace("data/postIMS/","data/preIMS/")


postIMS_ome = from_tiff(postIMS_file)
postIMS_resolution = postIMS_ome.images[0].pixels.physical_size_x
logging.info(f"postIMS resolution: {postIMS_resolution}")
assert postIMS_resolution == resolution
preIMS_ome = from_tiff(preIMS_file)
preIMS_resolution = preIMS_ome.images[0].pixels.physical_size_x
logging.info(f"preIMS resolution: {preIMS_resolution}")
assert preIMS_resolution == resolution

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
    if rescale != 1:
        bbox = (bbox/rescale).astype(int)
    logging.info(f"    {bbox}:{imcmaskfile}")
    imcbboxls.append(bbox)

def get_pyrlvl_rescalemod_imgshape(imgfile, rescale):
    preIMS_pyr_levels = get_pyr_levels(imgfile)
    preIMS_pyr_xsize = [get_image_shape(imgfile, pyr_level=i)[0] for i in preIMS_pyr_levels]
    preIMS_pyr_xscalefactor = [1]+[1/np.round(preIMS_pyr_xsize[i]/preIMS_pyr_xsize[i-1],3) for i in range(1,len(preIMS_pyr_xsize))]
    for i in range(2,len(preIMS_pyr_xscalefactor)):
        preIMS_pyr_xscalefactor[i] *= preIMS_pyr_xscalefactor[i-1]
    preIMS_pyr_xres = [resolution*preIMS_pyr_xscalefactor[i] for i in range(len(preIMS_pyr_xscalefactor))]

    if rescale in preIMS_pyr_xscalefactor:
        preIMS_pyr_level = preIMS_pyr_levels[preIMS_pyr_xscalefactor.index(rescale)]
        preIMS_rescale_modifier = preIMS_pyr_xscalefactor[preIMS_pyr_xscalefactor.index(rescale)]
        preIMS_imgshape = get_image_shape(imgfile, pyr_level=preIMS_pyr_level)
        preIMS_rescale=1
    else:   
        preIMS_pyr_level = 0
        preIMS_rescale_modifier = 1
        preIMS_imgshape = get_image_shape(imgfile)
        preIMS_imgshape = (int(preIMS_imgshape[0]/rescale),int(preIMS_imgshape[1]/rescale),preIMS_imgshape[2])
        preIMS_rescale=rescale
    return preIMS_pyr_level, preIMS_rescale, preIMS_rescale_modifier, preIMS_imgshape

logging.info("Create bounding box of preIMS tissue")
preIMS_pyr_level, preIMS_rescale, preIMS_rescale_modifier, preIMS_shape = get_pyrlvl_rescalemod_imgshape(preIMS_file, rescale)
logging.info("Create bounding box of postIMS tissue")
postIMS_pyr_level, postIMS_rescale, postIMS_rescale_modifier, postIMS_shape = get_pyrlvl_rescalemod_imgshape(postIMS_file, rescale)

 
corebboxls = list()
preIMSstitch = np.zeros(preIMS_shape[:2])
for i,bb1 in enumerate(imcbboxls):
    logging.info(f"\t{imc_mask_files[i]}")
    logging.info(f"\t\t{bb1}")
    # bounding box
    bbn = [0]*4
    # scale up by 1.35 mm in each direction, leading to image size of about 3.7mm * 3.7mm, which should be enough to include whole TMA core
    bbn[0] = max(0, int(np.floor((bb1[0] - 1350/resolution/rescale))))
    bbn[1] = max(0, int(np.floor((bb1[1] - 1350/resolution/rescale))))
    bbn[2] = min(preIMS_shape[0], int(np.ceil((bb1[2] + 1350/resolution/rescale))))
    bbn[3] = min(preIMS_shape[1], int(np.ceil((bb1[3] + 1350/resolution/rescale))))
    logging.info(f"\t\t{bbn}")

    img_mask = extract_mask(preIMS_file, np.array(bbn*preIMS_rescale).astype(int), session=rembg_session, rescale=preIMS_rescale_modifier/4, is_postIMS = False, pyr_level=preIMS_pyr_level)[0,:,:]
    img_mask = preprocess_mask(img_mask,1)
    wn=bbn[2]-bbn[0]
    hn=bbn[3]-bbn[1]
    img_mask = cv2.resize(img_mask.astype(np.uint8), (hn,wn), interpolation=cv2.INTER_NEAREST)

    bm = cv2.boundingRect(img_mask)
    bbnn = bbn.copy()
    bbnn[0] += bm[1]
    bbnn[1] += bm[0]
    bbnn[2] = bbnn[0]+bm[3]
    bbnn[3] = bbnn[1]+bm[2]

    logging.info(f"\t\t{bbnn}")
    # edges
    if bbnn[0]>bb1[0]:
        bbnn[0] = bbn[0]
    if bbnn[1]>bb1[1]:
        bbnn[1] = bbn[1]
    if bbnn[2]<bb1[2]:
        bbnn[2] = bbn[2]
    if bbnn[3]<bb1[3]:
        bbnn[3] = bbn[3]
    logging.info(f"\t\t{bbnn}")

    assert((bbnn[2]-bbnn[0]) <= (bbn[2]-bbn[0]))
    assert((bbnn[3]-bbnn[1]) <= (bbn[3]-bbn[1]))

    assert((bbnn[2]-bbnn[0]) >= (bb1[2]-bb1[0]))
    assert((bbnn[3]-bbnn[1]) >= (bb1[3]-bb1[1]))

    assert(bbnn[0]<bb1[0])
    assert(bbnn[1]<bb1[1])
    assert(bbnn[2]>bb1[2])
    assert(bbnn[3]>bb1[3])

    bbnn = np.array(bbnn)
    logging.info(f"\t\t{bbnn}")
    corebboxls.append(bbnn)
    preIMSstitch[bbn[0]:bbn[2],bbn[1]:bbn[3]] = np.max(np.stack([preIMSstitch[bbn[0]:bbn[2],bbn[1]:bbn[3]],img_mask], axis=0),axis=0)

if out_rescale != 1 or rescale != 1:
    wn, hn = (np.array(get_image_shape(postIMS_file)[:2])/out_rescale).astype(int)
    cvi = cv2.resize(preIMSstitch, (hn,wn), interpolation=cv2.INTER_NEAREST)
saveimage_tile(preIMSstitch, preIMSr_file, resolution*out_rescale)

if postIMSmask_extraction_constraint == "preIMS":
    logging.info("Only extract postIMS tissue location using preIMS mask")
    logging.info("Downscale preIMS mask")
    postIMSstitch = np.zeros(postIMS_shape[:2])
    preIMSstitch_red = cv2.resize(preIMSstitch, (postIMSstitch.shape[1],postIMSstitch.shape[0]), interpolation=cv2.INTER_NEAREST)
    logging.info(f"Get convex hull")
    lbs = skimage.measure.label(preIMSstitch_red)
    rps = skimage.measure.regionprops(lbs)
    cvi = lbs*0
    for i in range(len(rps)):
        tbb = rps[i].bbox
        ti = skimage.morphology.convex_hull_image(lbs[tbb[0]:tbb[2],tbb[1]:tbb[3]]==rps[i].label)
        logging.info(f"Number of pixels in postIMS mask before: {np.sum(ti>0)}")
        cvi[tbb[0]:tbb[2],tbb[1]:tbb[3]] = np.logical_or(ti,cvi[tbb[0]:tbb[2],tbb[1]:tbb[3]])
        logging.info(f"Number of pixels in postIMS mask after: {np.sum(cvi[tbb[0]:tbb[2],tbb[1]:tbb[3]]>0)}")

    logging.info("Upscale mask")
    # rescale
    if out_rescale != 1 or rescale != 1:
        wn, hn = (np.array(get_image_shape(postIMS_file)[:2])/out_rescale).astype(int)
        cvi = cv2.resize(cvi, (hn,wn), interpolation=cv2.INTER_NEAREST)
    logging.info("Save mask")
    saveimage_tile(cvi, postIMSr_file, resolution*out_rescale)
    sys.exit(0)


# def extract_mask(img: np.ndarray, session, rescale: float):
#     """
#     extract postIMS tissue location using rembg
#     """
#     # to grayscale, rescale
#     w = convert_and_scale_image(img, rescale)
#     # using fft, remove IMS ablation grid
#     #w = subtract_postIMS_grid(w)
#     w = cv2.blur(w, (5,5))
#     w = np.stack([w, w, w], axis=2)
#     # detect background
#     wr = remove(w, only_mask=True, session=session)
#     # threshold
#     tmpimg = wr>127
#     # upscale to not touch border, remove small holes
#     tmpimg2 = np.zeros((tmpimg.shape[0]+10,tmpimg.shape[1]+10))
#     tmpimg2[5:(tmpimg.shape[0]+5),5:(tmpimg.shape[1]+5)] = tmpimg
#     tmpimg = skimage.morphology.remove_small_holes(tmpimg2>0,500**2*np.pi*(1/resolution))[5:(tmpimg.shape[0]+5),5:(tmpimg.shape[1]+5)]
#     tmpimg = tmpimg.astype(np.uint8)*255
#     tmpimg = preprocess_mask(tmpimg,1)
#     tmpimg = tmpimg.astype(np.uint8)*255
#     tmpimg2 = np.zeros(np.array(list(tmpimg.shape))+20, dtype=np.uint8)
#     tmpimg2[10:(tmpimg2.shape[0]-10),10:(tmpimg2.shape[1]-10)] = tmpimg.astype(np.uint8)
#     # upscale to not touch border, get convex hull
#     tmpimg2 = skimage.morphology.convex_hull_image(tmpimg2>127)
#     tmpimg2 = tmpimg2.astype(np.uint8)*255
#     tmpimg = tmpimg2[10:(tmpimg2.shape[0]-10),10:(tmpimg2.shape[1]-10)]
#     # morphological closing
#     tmpimg = cv2.morphologyEx(tmpimg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5,5),np.uint8)).astype(bool)
#     return tmpimg 

logging.info("Remove background individually for each IMC location with rembg")
postIMSstitch = np.zeros(postIMS_shape[:2])
preIMSstitch_red = cv2.resize(preIMSstitch, (postIMSstitch.shape[1],postIMSstitch.shape[0]), interpolation=cv2.INTER_NEAREST)
rembg_mask_areas = []
for i in range(len(corebboxls)):
    xmin = max(0, int(corebboxls[i][0] - expand_microns/resolution/rescale))
    ymin = max(0, int(corebboxls[i][1] - expand_microns/resolution/rescale))
    xmax = min(postIMS_shape[0], int(corebboxls[i][2] + expand_microns/resolution/rescale))
    ymax = min(postIMS_shape[1], int(corebboxls[i][3] + expand_microns/resolution/rescale))
    logging.info(f"i: {i}, {os.path.basename(imc_mask_files[i])}, coords:[{xmin}:{xmax},{ymin}:{ymax}]")
    tmpimg = extract_mask(postIMS_file,(np.ceil(np.array([xmin,ymin,xmax,ymax])*postIMS_rescale)).astype(int), rembg_session, postIMS_rescale_modifier/4 , pyr_level=postIMS_pyr_level)[0,:,:]
    rembg_mask_areas.append(np.sum(tmpimg>0))
    wn, hn = postIMSstitch[xmin:xmax,ymin:ymax].shape
    tmpimg = cv2.resize(tmpimg[:wn,:hn].astype(np.uint8), (hn,wn), interpolation=cv2.INTER_NEAREST).astype(bool)
    logging.info(f"\tshape 1: {postIMSstitch[xmin:xmax,ymin:ymax].shape}")
    logging.info(f"\tshape 2: {tmpimg.shape}")
    logging.info(f"\tarea: {np.sum(tmpimg>0)}")
    postIMSstitch[xmin:xmax,ymin:ymax] = np.max(np.stack([postIMSstitch[xmin:xmax,ymin:ymax],tmpimg], axis=0),axis=0)

# threshold
postIMSrs = postIMSstitch>0
if out_rescale != 1 or rescale != 1:
    wn, hn = (np.array(get_image_shape(postIMS_file)[:2])/out_rescale).astype(int)
    postIMSrs = cv2.resize(postIMSrs.astype(np.uint8), (hn,wn), interpolation=cv2.INTER_NEAREST)
saveimage_tile(postIMSrs, postIMSr_file.replace("reduced_mask","reduced_mask_rembg"), resolution*out_rescale)


logging.info("Check mask")
IMCrs_filled = list()
for i in range(len(imcbboxls)):
    xmin = int(imcbboxls[i][0])
    ymin = int(imcbboxls[i][1])
    xmax = int(imcbboxls[i][2])
    ymax = int(imcbboxls[i][3])
    IMCrs_filled.append(np.min(postIMSrs[xmin:xmax,ymin:ymax]) == 1)

postIMSstitch = np.zeros(postIMS_shape[:2])
inds = np.array(list(range(len(imcbboxls))))
logging.info("Run segment anything model on IMC locations")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_mask_areas = []
for i in inds:
    # bounding box
    xmin = max(0, int(corebboxls[i][0] - expand_microns/resolution/rescale))
    ymin = max(0, int(corebboxls[i][1] - expand_microns/resolution/rescale))
    xmax = min(postIMS_shape[0], int(corebboxls[i][2] + expand_microns/resolution/rescale))
    ymax = min(postIMS_shape[1], int(corebboxls[i][3] + expand_microns/resolution/rescale))
    logging.info(f"i: {i}, {os.path.basename(imc_mask_files[i])}, coords:[{xmin}:{xmax},{ymin}:{ymax}]")

    # read image
    saminp = readimage_crop(postIMS_file, (np.ceil(np.array([xmin,ymin,xmax,ymax])*postIMS_rescale)).astype(int), pyr_level=postIMS_pyr_level)
    # to gray scale, rescale
    saminp = convert_and_scale_image(saminp, postIMS_rescale_modifier/4)
    saminp = np.stack([saminp, saminp, saminp], axis=2)

    imcarea = (imcbboxls[i][2]-imcbboxls[i][0])*(imcbboxls[i][3]-imcbboxls[i][1])
    pts = np.array([
        [imcbboxls[i][0]-corebboxls[i][0],imcbboxls[i][1]-corebboxls[i][1]],
        [imcbboxls[i][2]-corebboxls[i][0],imcbboxls[i][1]-corebboxls[i][1]],
        [imcbboxls[i][2]-corebboxls[i][0],imcbboxls[i][3]-corebboxls[i][1]],
        [imcbboxls[i][0]-corebboxls[i][0],imcbboxls[i][3]-corebboxls[i][1]],
        [((imcbboxls[i][2]-corebboxls[i][0])-(imcbboxls[i][0]-corebboxls[i][0]))//2,((imcbboxls[i][3]-corebboxls[i][1])-(imcbboxls[i][1]-corebboxls[i][1]))//2],
    ])*(postIMS_rescale*postIMS_rescale_modifier/4)
    pts = pts.astype(int)
    postIMSmasks, scores1 = sam_core(saminp, sam, pts)
    postIMSmasks = np.stack([preprocess_mask(msk,1) for msk in postIMSmasks ])
    pts_in_mask = list()
    for j in range(len(postIMSmasks)):
        pts_in_mask.append(np.sum([postIMSmasks[j][p[0],p[1]] for p in pts]))
    tmpperimeters = np.array([skimage.measure.regionprops(im)[0].perimeter for im in postIMSmasks ])
    tmpareas = np.array([np.sum(im>0) for im in postIMSmasks])
    tmpcircularity = np.array([4*np.pi*a/p**2 for a,p in zip(tmpareas,tmpperimeters)])
    thr1=(np.max([postIMSmasks.shape[1], postIMSmasks.shape[2]])/2)**2*np.pi
    tmpinds = np.array(list(range(3)))
    tmpinds = tmpinds[np.logical_and(np.logical_and(tmpareas > imcarea*1.02, tmpareas < thr1), np.array(pts_in_mask)>=len(pts)-2)]
    logging.info(f"\t areas: {tmpareas}")
    logging.info(f"\t scores: {scores1}")
    logging.info(f"\t number of points in mask: {pts_in_mask}")
    logging.info(f"\t tmpinds: {tmpinds}")

    if len(tmpinds) == 0:
        logging.info(f"\tCould not find a mask for {os.path.basename(imc_mask_files[i])}")
        logging.info(f"\tTry to find mask without points")
         # run SAM segmentation model
        postIMSmasks, scores1 = sam_core(saminp, sam)
        # postprocess
        postIMSmasks = np.stack([preprocess_mask(msk,1) for msk in postIMSmasks ])
        tmpareas = np.array([np.sum(im) for im in postIMSmasks])
        tmpinds = np.array(list(range(3)))
        tmpinds = tmpinds[np.logical_and(tmpareas > imcarea*1.02, tmpareas < thr1)]

        logging.info(f"\t areas: {tmpareas}")
        logging.info(f"\t scores: {scores1}")
        logging.info(f"\t tmpinds: {tmpinds}")

    assert len(tmpinds) > 0
    harmonic_scores = np.array([a*b for a,b in zip(scores1[tmpinds],tmpcircularity[tmpinds])])
    tmpind = tmpinds[harmonic_scores==np.max(harmonic_scores)]
    tmpimg = postIMSmasks[tmpind,:,:][0,:,:].astype(np.uint8)*255
    sam_mask_areas.append(np.sum(tmpimg>0))
    wn, hn = postIMSstitch[xmin:xmax,ymin:ymax].shape
    tmpimg = cv2.resize(tmpimg[:wn,:hn].astype(np.uint8), (hn,wn), interpolation=cv2.INTER_NEAREST).astype(bool)
    logging.info(f"\tshape 1: {postIMSstitch[xmin:xmax,ymin:ymax].shape}")
    logging.info(f"\tshape 2: {tmpimg.shape}")
    logging.info(f"\tarea postIMSmask: {np.sum(tmpimg>0)}")
    logging.info(f"\tarea preIMSmask: {np.sum(preIMSstitch_red[xmin:xmax,ymin:ymax]>0)}")
    postIMSstitch[xmin:xmax,ymin:ymax] = np.max(np.stack([postIMSstitch[xmin:xmax,ymin:ymax],tmpimg], axis=0),axis=0)
    # cv2.imwrite("tmp.png",tmpimg.astype(np.uint8)*255)


if out_rescale != 1 or rescale != 1:
    wn, hn = (np.array(get_image_shape(postIMS_file)[:2])/out_rescale).astype(int)
    postIMSstitch_out = cv2.resize(postIMSstitch.astype(np.uint8), (hn,wn), interpolation=cv2.INTER_NEAREST)
saveimage_tile(postIMSstitch_out>0, postIMSr_file.replace("reduced_mask","reduced_mask_no_constraints"), resolution*out_rescale)

if not postIMSmask_extraction_constraint is None:
    logging.info("Apply mask extraction constraint")
    if postIMSmask_extraction_constraint == "min_preIMS":
        preIMSstitch_red = cv2.morphologyEx(preIMSstitch_red.astype(np.uint8), cv2.MORPH_DILATE, np.ones((int(postIMSmask_extraction_constraint_parameter),int(postIMSmask_extraction_constraint_parameter)),np.uint8))
    if postIMSmask_extraction_constraint == "max_preIMS":
        preIMSstitch_red = cv2.morphologyEx(preIMSstitch_red.astype(np.uint8), cv2.MORPH_ERODE, np.ones((int(postIMSmask_extraction_constraint_parameter),int(postIMSmask_extraction_constraint_parameter)),np.uint8))
    logging.info(f"Number of pixels in preIMS mask: {np.sum(preIMSstitch_red>0)}")
    logging.info(f"Number of pixels in postIMS mask before: {np.sum(postIMSstitch>0)}")
    postIMSstitch = np.max(np.stack([postIMSstitch,preIMSstitch_red],axis=0),axis=0)
    logging.info(f"Number of pixels in postIMS mask after: {np.sum(postIMSstitch>0)}")
    postIMSstitch = cv2.morphologyEx(postIMSstitch.astype(np.uint8), cv2.MORPH_OPEN, np.ones((25,25),np.uint8))
    logging.info(f"Number of pixels in postIMS mask after closing: {np.sum(postIMSstitch>0)}")

# postIMSstitchdownscaled = cv2.resize(postIMSstitch.astype(np.uint8), (int(postIMSstitch.shape[1]*resolution),int(postIMSstitch.shape[0]*resolution)), interpolation=cv2.INTER_NEAREST)
# cv2.imwrite("tmp.png",postIMSstitch.astype(np.uint8)*255)
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


if out_rescale != 1 or rescale != 1:
    wn, hn = (np.array(get_image_shape(postIMS_file)[:2])/out_rescale).astype(int)
    postIMSsamr = cv2.resize(postIMSsamr.astype(np.uint8), (hn,wn), interpolation=cv2.INTER_NEAREST)
saveimage_tile(postIMSsamr, postIMSr_file.replace("reduced_mask","reduced_mask_no_convexhull"), resolution*out_rescale)



logging.info(f"Get convex hull")
lbs = skimage.measure.label(postIMSsamr)
rps = skimage.measure.regionprops(lbs)
cvi = lbs*0
for i in range(len(rps)):
    tbb = rps[i].bbox
    ti = skimage.morphology.convex_hull_image(lbs[tbb[0]:tbb[2],tbb[1]:tbb[3]]==rps[i].label)
    logging.info(f"Number of pixels in postIMS mask before: {np.sum(ti>0)}")
    cvi[tbb[0]:tbb[2],tbb[1]:tbb[3]] = np.logical_or(ti,cvi[tbb[0]:tbb[2],tbb[1]:tbb[3]])
    logging.info(f"Number of pixels in postIMS mask after: {np.sum(cvi[tbb[0]:tbb[2],tbb[1]:tbb[3]]>0)}")
logging.info("Save mask")


if out_rescale != 1 or rescale != 1:
    wn, hn = (np.array(get_image_shape(postIMS_file)[:2])/out_rescale).astype(int)
    cvi = cv2.resize(cvi, (hn,wn), interpolation=cv2.INTER_NEAREST)
saveimage_tile(cvi, postIMSr_file, resolution*out_rescale)
logging.info("Finished")
