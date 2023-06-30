from wsireg.writers.ome_tiff_writer import OmeTiffWriter
from wsireg.reg_transforms.reg_transform_seq import RegTransformSeq
from wsireg.reg_images.loader import reg_image_loader
import numpy as np
from tifffile import imread
import os
from shapely.geometry import shape
import json


# microscopy_pixelsize = 0.22537
microscopy_pixelsize = snakemake.params["microscopy_pixelsize"]

# transform_file_postIMC_to_postIMS = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/registrations/postIMC_to_postIMS/A1/test_split_pre_A1-postIMC_to_postIMS_transformations.json"
transform_file_postIMC_to_postIMS=snakemake.input["postIMC_to_postIMS_transform"]

# img_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/test_split_pre_postIMC.ome.tiff"
img_file = snakemake.input["postIMC"]


# IMC_geojson_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_location/test_split_pre_IMC_mask_on_postIMC_A1.geojson"
IMC_geojson_file=snakemake.input['IMC_location']

# img_out = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/A1_postIMC_transformed.ome.tiff"
img_out = snakemake.output["postIMC_transformed"]
img_basename = os.path.basename(img_out).split(".")[0]
img_dirname = os.path.dirname(img_out)

# read image
img=imread(img_file)

# get info of IMC location
IMC_geojson = json.load(open(IMC_geojson_file, "r"))
IMC_geojson_polygon = shape(IMC_geojson['geometry'])
# bounding box
bb1 = IMC_geojson_polygon.bounds
# reorder axis
bb1 = [bb1[1],bb1[0],bb1[3],bb1[2]]
bbn = [0]*4
# scale up by 1.35 mm in each direction, leading to image size of about 3.7mm * 3.7mm, which should be enough to include whole TMA core
bbn[0] = int(np.floor(bb1[0]/microscopy_pixelsize - 1350/microscopy_pixelsize))
bbn[1] = int(np.floor(bb1[1]/microscopy_pixelsize - 1350/microscopy_pixelsize))
bbn[2] = int(np.ceil(bb1[2]/microscopy_pixelsize + 1350/microscopy_pixelsize ))
bbn[3] = int(np.ceil(bb1[3]/microscopy_pixelsize + 1350/microscopy_pixelsize ))
# edges
if bbn[0]<0:
    bbn[0] = 0
if bbn[1]<0:
    bbn[1] = 0
if bbn[2]>img.shape[0]/microscopy_pixelsize:
    bbn[2] = img.shape[0]/microscopy_pixelsize
if bbn[3]>img.shape[1]/microscopy_pixelsize:
    bbn[3] = img.shape[1]/microscopy_pixelsize

# create empty image and fill in core
imgnew = img*0
imgnew[bbn[0]:bbn[2],bbn[1]:bbn[3]] = img[bbn[0]:bbn[2],bbn[1]:bbn[3]]

# setup transformation sequence
rtsn=RegTransformSeq(transform_file_postIMC_to_postIMS)
rtsn.set_output_spacing((microscopy_pixelsize,microscopy_pixelsize))
rtsn.set_output_spacing((1.0,1.0))

# transform and save image
ri = reg_image_loader(imgnew, microscopy_pixelsize)
writer = OmeTiffWriter(ri, reg_transform_seq=rtsn)
writer.write_image_by_plane(img_basename, output_dir=img_dirname, tile_size=1024)
