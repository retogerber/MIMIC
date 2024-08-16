import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
import cv2
from wsireg.writers.ome_tiff_writer import OmeTiffWriter
from wsireg.reg_transforms.reg_transform_seq import RegTransformSeq
from wsireg.reg_images.loader import reg_image_loader
import numpy as np
from shapely.geometry import shape
import json
from image_utils import get_image_shape, readimage_crop, extract_mask
from utils import setNThreads, snakeMakeMock
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["input_spacing"] = 0.22537
    snakemake.params["IMC_location_spacing"] = 0.22537
    snakemake.params["output_spacing"] = 0.22537
    snakemake.params["transform_target"] = "preIMS"
    snakemake.params["transform_source"] = "postIMC"
    snakemake.input["postIMC_to_postIMS_transform"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/registrations/postIMC_to_postIMS/B5/NASH_HCC_TMA_B5-postIMC_to_postIMS_transformations_mod.json"
    snakemake.input["postIMC"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/postIMC/NASH_HCC_TMA_postIMC.ome.tiff"
    snakemake.input['IMC_location'] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/IMC_location/NASH_HCC_TMA_IMC_mask_on_postIMC_B5.geojson"
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
input_spacing = snakemake.params["input_spacing"]
output_spacing = snakemake.params["output_spacing"]
IMC_location_spacing = snakemake.params["IMC_location_spacing"]
transform_target = snakemake.params["transform_target"]
transform_source = snakemake.params["transform_source"]

# inputs
transform_file_postIMC_to_postIMS=snakemake.input["postIMC_to_postIMS_transform"]
img_file = snakemake.input["postIMC"]
IMC_geojson_file=snakemake.input['IMC_location']
if isinstance(IMC_geojson_file, list):
    IMC_geojson_file = IMC_geojson_file[0]

# outputs
img_out = snakemake.output["postIMC_transformed"]
img_basename = os.path.basename(img_out).split(".")[0]
img_dirname = os.path.dirname(img_out)

logging.info("Load transform")
# setup transformation sequence
rtsn=RegTransformSeq(transform_file_postIMC_to_postIMS)
rtsn.set_output_spacing((float(output_spacing),float(output_spacing)))

rtls = rtsn.reg_transforms
all_linear = np.array([r.is_linear for r in rtls]).all()
if all_linear:
    assert(len(rtls)==5 or len(rtls)==3)
    is_split_transform = len(rtls)==5
else:
    # len=4 : direct registration
    # len=6 : additional separate registration between preIMC and preIMS
    assert(len(rtls)==6 or len(rtls)==4)
    is_split_transform = len(rtls)==6


# get info of IMC location
IMC_geojson = json.load(open(IMC_geojson_file, "r"))
if isinstance(IMC_geojson,list):
    IMC_geojson=IMC_geojson[0]
IMC_geojson_polygon = shape(IMC_geojson['geometry'])

logging.info("Create bounding box")
# bounding box
bb1 = IMC_geojson_polygon.bounds
# reorder axis
bb1 = np.array([bb1[1],bb1[0],bb1[3],bb1[2]])/(IMC_location_spacing/input_spacing)
bbn = [0]*4
# scale up by 1.35 mm in each direction, leading to image size of about 3.7mm * 3.7mm, which should be enough to include whole TMA core
bbn[0] = int(np.floor(bb1[0] - 1350/input_spacing))
bbn[1] = int(np.floor(bb1[1] - 1350/input_spacing))
bbn[2] = int(np.ceil(bb1[2] + 1350/input_spacing ))
bbn[3] = int(np.ceil(bb1[3] + 1350/input_spacing ))


logging.info("Setup transformation for image")
if transform_target == "preIMC":
    n_end = 1
elif transform_target == "preIMS":
    if all_linear:
        n_end = 4 if is_split_transform else 2
    else:
        n_end = 5 if is_split_transform else 3
elif transform_target == "postIMS":
    if all_linear:
        n_end = 5 if is_split_transform else 3
    else:
        n_end = 6 if is_split_transform else 4
else:
    raise ValueError("Unknown transform target: " + transform_target)

if transform_source == "postIMC":
    n_start = 0
elif transform_source == "preIMC":
    n_start = 1
elif transform_source == "preIMS":
    if all_linear:
        n_start = 4 if is_split_transform else 2
    else:
        n_start = 5 if is_split_transform else 3
else:
    raise ValueError("Unknown transform source: " + transform_source)

rtls = rtsn.reg_transforms
rtls = rtls[n_start:n_end]
rtsn = RegTransformSeq(rtls, transform_seq_idx=list(range(len(rtls))))
if len(rtls)>0:
    rtsn.set_output_spacing((float(output_spacing),float(output_spacing)))


logging.info("Read Image")
# read image
imgshape = get_image_shape(img_file)
# edges
if bbn[0]<0:
    bbn[0] = 0
if bbn[1]<0:
    bbn[1] = 0
if bbn[2]>imgshape[0]:
    bbn[2] = imgshape[0]
if bbn[3]>imgshape[1]:
    bbn[3] = imgshape[1]

img_mask = extract_mask(img_file, bbn, session=None, rescale=input_spacing/4, is_postIMS = transform_source=="postIMS")[0,:,:]
wn = int(img_mask.shape[0]*(4/input_spacing))
hn = int(img_mask.shape[1]*(4/input_spacing))
img_mask = cv2.resize(img_mask.astype(np.uint8), (hn,wn), interpolation=cv2.INTER_NEAREST)

bm = cv2.boundingRect(img_mask)
bbnn = bbn.copy()
bbnn[0] += bm[1]
bbnn[1] += bm[0]
bbnn[2] = bbnn[0]+bm[3]
bbnn[3] = bbnn[1]+bm[2]

# edges
if bbnn[0]>bb1[0]:
    bbnn[0] = bbn[0]
if bbnn[1]>bb1[1]:
    bbnn[1] = bbn[1]
if bbnn[2]<bb1[2]:
    bbnn[2] = bbn[2]
if bbnn[3]<bb1[3]:
    bbnn[3] = bbn[3]


assert((bbnn[2]-bbnn[0]) <= (bbn[2]-bbn[0]))
assert((bbnn[3]-bbnn[1]) <= (bbn[3]-bbn[1]))

assert((bbnn[2]-bbnn[0]) >= (bb1[2]-bb1[0]))
assert((bbnn[3]-bbnn[1]) >= (bb1[3]-bb1[1]))

assert(bbnn[0]<=bb1[0])
assert(bbnn[1]<=bb1[1])
assert(bbnn[2]>=bb1[2])
assert(bbnn[3]>=bb1[3])

logging.info(f"final bounding box: {bbnn}")

img = readimage_crop(img_file, bbnn)

logging.info("Create new image")
# create empty image and fill in core
imgnew = np.zeros(imgshape, dtype=np.uint8)
imgnew[bbnn[0]:bbnn[2],bbnn[1]:bbnn[3]] = img


logging.info("Transform and save image")
# transform and save image
ri = reg_image_loader(imgnew, input_spacing)
writer = OmeTiffWriter(ri, reg_transform_seq=rtsn)
writer.write_image_by_plane(img_basename, output_dir=img_dirname, tile_size=1024)

logging.info("Finished")
