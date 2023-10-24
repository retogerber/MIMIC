import cv2
import SimpleITK as sitk
from wsireg.writers.ome_tiff_writer import OmeTiffWriter
from wsireg.reg_shapes import RegShapes
from wsireg.reg_transforms.reg_transform_seq import RegTransformSeq
from wsireg.reg_images.loader import reg_image_loader
from image_registration_IMS_to_preIMS_utils import get_image_shape, readimage_crop
import numpy as np
from tifffile import imread
from shapely.geometry import shape
import json
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

# input_spacing = 0.22537
input_spacing = snakemake.params["input_spacing"]
# output_spacing=0.22537
output_spacing = snakemake.params["output_spacing"]
# IMC_location_spacing=0.22537
IMC_location_spacing = snakemake.params["IMC_location_spacing"]



# transform_target = "preIMC"
transform_target = snakemake.params["transform_target"]
# transform_source = "postIMC"
transform_source = snakemake.params["transform_source"]

cv2.setNumThreads(snakemake.threads)
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(snakemake.threads)
transform_file_postIMC_to_postIMS = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/registrations/postIMC_to_postIMS/3/NASH_HCC_TMA_3-postIMC_to_postIMS_transformations.json"
# transform_file_postIMC_to_postIMS = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/registrations/postIMC_to_postIMS/1/NASH_HCC_TMA_1-postIMC_to_postIMS_transformations.json"
# transform_file_postIMC_to_postIMS_single = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/registrations/postIMC_to_postIMS/test_split_pre-postIMC_to_postIMS_transformations.json"
# transform_file_postIMC_to_postIMS = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/registrations/postIMC_to_postIMS/A1/test_split_pre_A1-postIMC_to_postIMS_transformations.json"
transform_file_postIMC_to_postIMS=snakemake.input["postIMC_to_postIMS_transform"]

img_file = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/postIMC/NASH_HCC_TMA_postIMC.ome.tiff"
# img_file = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/preIMC/NASH_HCC_TMA_preIMC.ome.tiff"
# img_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/test_split_pre_postIMC.ome.tiff"
img_file = snakemake.input["postIMC"]

IMC_geojson_file = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/IMC_location/NASH_HCC_TMA_IMC_mask_on_postIMC_B5.geojson"
# IMC_geojson_file = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/IMC_location/NASH_HCC_TMA_IMC_mask_on_postIMC_D2.geojson"
# IMC_geojson_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_location/test_split_pre_IMC_mask_on_postIMC_A1.geojson"
IMC_geojson_file=snakemake.input['IMC_location']
if isinstance(IMC_geojson_file, list):
    IMC_geojson_file = IMC_geojson_file[0]

# img_out = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/A1_postIMC_transformed.ome.tiff"
img_out = snakemake.output["postIMC_transformed"]
img_basename = os.path.basename(img_out).split(".")[0]
img_dirname = os.path.dirname(img_out)

logging.info("Load transform")
# setup transformation sequence
rtsn=RegTransformSeq(transform_file_postIMC_to_postIMS)
#rtsn.set_output_spacing((microscopy_pixelsize,microscopy_pixelsize))
#rtsn.set_output_spacing((1.0,1.0))
rtsn.set_output_spacing((float(output_spacing),float(output_spacing)))

# if transform_target != "postIMS":
rtls = rtsn.reg_transforms
is_split_transform = len(rtls)==6



# logging.info("Setup transformation for IMC location")
# if transform_source == "postIMC":
#     n_start = 0
# elif transform_source == "preIMC":
#     n_start = 1
# elif transform_source == "preIMS":
#     n_start = 5 if is_split_transform else 3
# else:
#     raise ValueError("Unknown transform source: " + transform_source)

# rtls = rtsn.reg_transforms
# rtls = rtls[:n_start]
# logging.info(f"rtls: {rtls}")
# rtsngeo = RegTransformSeq(rtls, transform_seq_idx=list(range(len(rtls))))
# if len(rtls)>0:
#     rtsngeo.set_output_spacing((float(output_spacing),float(output_spacing)))


# logging.info("Read json, transform and create shape")
# # get info of IMC location
IMC_geojson = json.load(open(IMC_geojson_file, "r"))
if isinstance(IMC_geojson,list):
    IMC_geojson=IMC_geojson[0]

# if len(rtls)>0:
#     rs = RegShapes(IMC_geojson_file, source_res=1, target_res=output_spacing)
#     rs.transform_shapes(rtsngeo)

#     IMC_geojson['geometry']['coordinates'] = [rs.transformed_shape_data[0]['array'].tolist()]
# else:
#     IMC_geojson['geometry']['coordinates'] = (np.array(IMC_geojson['geometry']['coordinates'])/output_spacing).tolist()

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
    n_end = 5 if is_split_transform else 3
elif transform_target == "postIMS":
    n_end = 6 if is_split_transform else 4
else:
    raise ValueError("Unknown transform target: " + transform_target)
if transform_source == "postIMC":
    n_start = 0
elif transform_source == "preIMC":
    n_start = 1
elif transform_source == "preIMS":
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
img = readimage_crop(img_file, bbn)


logging.info("Create new image")
# create empty image and fill in core
imgnew = np.zeros(imgshape, dtype=np.uint8)
imgnew[bbn[0]:bbn[2],bbn[1]:bbn[3]] = img


logging.info("Transform and save image")
# transform and save image
ri = reg_image_loader(imgnew, input_spacing)
writer = OmeTiffWriter(ri, reg_transform_seq=rtsn)
writer.write_image_by_plane(img_basename, output_dir=img_dirname, tile_size=1024)

logging.info("Finished")
