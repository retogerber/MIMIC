from wsireg.writers.ome_tiff_writer import OmeTiffWriter
from wsireg.reg_transforms.reg_transform import RegTransform
from wsireg.reg_transforms.reg_transform_seq import RegTransformSeq
from wsireg.reg_shapes import RegShapes
from wsireg.reg_images.loader import reg_image_loader
from wsireg.parameter_maps import transformations
import numpy as np
import os
from ome_types import from_tiff
from ome_types import to_xml
from tifffile import imread
from tifffile import imwrite
import sys
import json
from tqdm import tqdm
import cv2
import pickle
from pathlib import Path

# cell mask file
#cell_image_fp="/home/retger/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_006_transformed.ome.tiff"
cell_image_fp=snakemake.input["IMCmask"]

# read mask
cell_mask = imread(cell_image_fp)

# find bounding box
cell_mask_bin = cell_mask.copy()>0
xs=np.sum(cell_mask_bin,axis=1)
xmin=np.min(np.where(xs>0))
xmax=np.max(np.where(xs>0))
ys=np.sum(cell_mask_bin,axis=0)
ymin=np.min(np.where(ys>0))
ymax=np.max(np.where(ys>0))

# subset mask to area containing cells
cell_mask_sub = cell_mask.copy()[xmin:(xmax+1),ymin:(ymax+1)]

# convert cell mask raster data to polygons for transformation
cell_shapes = []
cell_indices = []
for cell_idx in tqdm(np.unique(cell_mask)):
    if cell_idx != 0:
        cell_mask_thresh = cell_mask_sub.copy()
        cell_mask_thresh[cell_mask_thresh < cell_idx] = 0
        cell_mask_thresh[cell_mask_thresh > cell_idx] = 0
        cell_mask_thresh[cell_mask_thresh == cell_idx] = 255

        cell_poly, _ = cv2.findContours(
            cell_mask_thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        if len(cell_poly) == 1:
            cell_shapes.append(np.squeeze(cell_poly).astype(np.double))
            cell_indices.append(cell_idx)

# translate to original image space
cell_shapes_translated=[]
cell_indices_translated=[]
for i,single_shape in enumerate(cell_shapes):
    if len(single_shape.shape)==2:
        single_shape[:,0]+=ymin
        single_shape[:,1]+=xmin
        cell_shapes_translated.append(single_shape)
        cell_indices_translated.append(cell_indices[i])

    
# cell masks to RegShapes model
rs = RegShapes(cell_shapes_translated)
output_fp_shapes=snakemake.output["IMCmask_shape_transformed"]
rs.save_shape_data(output_fp_shapes, transformed=False)

output_fp_indices=snakemake.output["cell_indices"]
with open(output_fp_indices, "wb") as f:
    pickle.dump(cell_indices_translated, f)


