import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
from rembg import new_session
import cv2
import json
import numpy as np
from image_utils import get_image_shape, extract_mask, saveimage_tile, preprocess_mask
from utils import setNThreads, snakeMakeMock
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["microscopy_pixelsize"] = 0.22537
    snakemake.params["IMC_location_pixelsize"] = 0.22537
    snakemake.params["max_TMA_diameter"] = 2500
    snakemake.input["postIMC"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/test_split_pre_postIMC.ome.tiff"
    snakemake.input["IMC_location"] = [f"/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_location/test_split_pre_registered_IMC_mask_on_postIMC_{core}.geojson" for core in ["A1","B1"]]
    snakemake.output["TMA_location"] = [f"/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/TMA_location/test_split_pre_TMA_location_on_postIMC_{core}.geojson" for core in ["A1","B1"]]
    snakemake.output["postIMC_mask"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/test_split_pre_postIMC_mask.ome.tiff"

    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# parameters
resolution_microscopy = float(snakemake.params["microscopy_pixelsize"])
resolution_IMC_location = float(snakemake.params["IMC_location_pixelsize"])
max_TMA_diameter = float(snakemake.params["max_TMA_diameter"])

# inputs
postIMC_file = snakemake.input["postIMC"]
imc_mask_files = snakemake.input["IMC_location"]

model_name = "isnet-general-use"
os.environ["OMP_NUM_THREADS"] = str(snakemake.threads)
rembg_session = new_session(model_name)

# outputs
TMA_locations_out = snakemake.output["TMA_location"]
if isinstance(TMA_locations_out, str):
    TMA_locations_out = [TMA_locations_out]
postIMC_mask_file = snakemake.output["postIMC_mask"]
if isinstance(postIMC_mask_file, list):
    postIMC_mask_file = postIMC_mask_file[0]


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
    # bbox to microscopy resolution
    if resolution_microscopy/resolution_IMC_location != 1:
        bbox = (bbox/resolution_IMC_location*resolution_microscopy).astype(int)
    logging.info(f"    {bbox}:{imcmaskfile}")
    imcbboxls.append(bbox)


logging.info("Create bounding box of postIMC tissue")
imgshape = get_image_shape(postIMC_file)
logging.info(f"postIMC shape: {imgshape}")
corebboxls = list()
postIMCstitch = np.zeros(imgshape[:2])
max_TMA_diameter_px = max_TMA_diameter/resolution_microscopy
for i,bb1 in enumerate(imcbboxls):
    logging.info(f"\t{imc_mask_files[i]}")
    logging.info(f"\t\t{bb1}")
    # max needed expansion to include whole TMA core
    expand_px = np.max([max_TMA_diameter_px-(bb1[2]-bb1[0]),max_TMA_diameter_px-(bb1[3]-bb1[1])])
    # bounding box
    bbn = [0]*4
    bbn[0] = max(0, int(np.floor((bb1[0] - expand_px))))
    bbn[1] = max(0, int(np.floor((bb1[1] - expand_px))))
    bbn[2] = min(imgshape[0], int(np.ceil((bb1[2] + expand_px))))
    bbn[3] = min(imgshape[1], int(np.ceil((bb1[3] + expand_px))))
    logging.info(f"\t\t{bbn}")

    img_mask = extract_mask(postIMC_file, bbn, session=rembg_session, rescale=resolution_microscopy/4, is_postIMS = False)[0,:,:]
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
    postIMCstitch[bbn[0]:bbn[2],bbn[1]:bbn[3]] = np.max(np.stack([postIMCstitch[bbn[0]:bbn[2],bbn[1]:bbn[3]],img_mask], axis=0),axis=0)

logging.info("Save TMA locations")
for imcmaskfile, bb, bbout in zip(imc_mask_files, corebboxls, TMA_locations_out):
    IMC_geojson = json.load(open(imcmaskfile, "r"))
    if isinstance(IMC_geojson,list):
        IMC_geojson=IMC_geojson[0]
    boundary_points = np.array(IMC_geojson['geometry']['coordinates'])[0,:,:]
    # corner points from bbox bb
    bpts = np.array([
        [bb[1],bb[0]],
        [bb[1],bb[2]],
        [bb[3],bb[2]],
        [bb[3],bb[0]]
    ])
    IMC_geojson['geometry']['coordinates'][0] = bpts.tolist()
    if not os.path.exists(os.path.dirname(bbout)):
        os.makedirs(os.path.dirname(bbout))
    with open(bbout, "w") as f:
        json.dump(IMC_geojson, f)

logging.info("Save postIMC mask") 
saveimage_tile(postIMCstitch, postIMC_mask_file, 1)
logging.info("Finished")
