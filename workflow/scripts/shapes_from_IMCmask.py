import SimpleITK as sitk
import wsireg
import numpy as np
from tifffile import imread
import cv2
from shapely.geometry import shape
import json
import pickle
from utils import setNThreads, snakeMakeMock
import sys,os
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params['input_spacing'] = 0.22537
    snakemake.params['input_spacing_IMC_location'] = 0.22537
    snakemake.params['output_spacing'] = 1
    snakemake.input["IMCmask"] = ""
    snakemake.input["IMC_location"] = ""
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
input_spacing = float(snakemake.params['input_spacing'])
input_spacing_IMC_location = float(snakemake.params['input_spacing_IMC_location'])
output_spacing = float(snakemake.params['output_spacing'])

# inputs
cell_image_fp=snakemake.input["IMCmask"]
IMC_geojson_file=snakemake.input['IMC_location']
if isinstance(IMC_geojson_file, list):
    IMC_geojson_file = IMC_geojson_file[0]


logging.info("Read Mask")
# read mask
cell_mask = imread(cell_image_fp)

if input_spacing/output_spacing != 1:
    wn = int(cell_mask.shape[0]*(input_spacing/output_spacing))
    hn = int(cell_mask.shape[1]*input_spacing/output_spacing)
    cell_mask = cv2.resize(cell_mask, (hn,wn), interpolation=cv2.INTER_NEAREST)


logging.info("Find bounding box")
IMC_geojson = json.load(open(IMC_geojson_file, "r"))
if isinstance(IMC_geojson,list):
    IMC_geojson=IMC_geojson[0]
IMC_geojson_polygon = shape(IMC_geojson['geometry'])
bb1 = IMC_geojson_polygon.bounds
# reorder axis
bb1 = np.array([bb1[1],bb1[0],bb1[3],bb1[2]])
if input_spacing_IMC_location/output_spacing != 1:
    bb1 = bb1*(input_spacing_IMC_location/output_spacing)
xmin=int(bb1[0])
ymin=int(bb1[1])
xmax=int(bb1[2])
ymax=int(bb1[3])

logging.info(f"bbox: {bb1}")
logging.info(f"image shape: {cell_mask.shape}")

assert(xmin>=0)
assert(ymin>=0)
assert(cell_mask.shape[0]>xmax)
assert(cell_mask.shape[1]>ymax)


# subset mask to area containing cells
cell_mask = cell_mask[xmin:(xmax+1),ymin:(ymax+1)]

unique_cells = np.unique(cell_mask)

logging.info("Cells raster to polygons")
# convert cell mask raster data to polygons for transformation
cell_shapes = []
cell_indices = []
for cell_idx in unique_cells:
    if cell_idx != 0:
        cell_mask_thresh = np.zeros(cell_mask.shape, dtype=np.uint8)
        cell_mask_thresh[cell_mask == cell_idx] = 255
        
        cell_poly, _ = cv2.findContours(
            cell_mask_thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        if len(cell_poly) == 1:
            cell_shapes.append(np.squeeze(cell_poly).astype(np.double))
            cell_indices.append(cell_idx)

logging.info("Translate polygons")
# translate to original image space
cell_shapes_translated=[]
cell_indices_translated=[]
for i,single_shape in enumerate(cell_shapes):
    if len(single_shape.shape)==2:
        single_shape[:,0]+=ymin
        single_shape[:,1]+=xmin
        cell_shapes_translated.append(single_shape)
        cell_indices_translated.append(cell_indices[i])

    
logging.info("Convert cells to shapes and save")
# cell masks to RegShapes model
rs = wsireg.reg_shapes.RegShapes(cell_shapes_translated)
output_fp_shapes=snakemake.output["IMCmask_shape_transformed"]
rs.save_shape_data(output_fp_shapes, transformed=False)

logging.info("Save cell indices")
output_fp_indices=snakemake.output["cell_indices"]
with open(output_fp_indices, "wb") as f:
    pickle.dump(cell_indices_translated, f)


logging.info("Finished")
