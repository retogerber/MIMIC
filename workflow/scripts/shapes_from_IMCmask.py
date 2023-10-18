from wsireg.reg_shapes import RegShapes
import numpy as np
from tifffile import imread
import skimage
import cv2
import pickle
import sys,os
import logging, traceback
logging.basicConfig(filename=snakemake.log["stdout"],
                    level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    )
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
cv2.setNumThreads(snakemake.threads)
# cell mask file
# cell_image_fp = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_001_transformed_on_postIMS.ome.tiff"
#cell_image_fp="/home/retger/imc_to_ims_workflow/results/cirrhosis_TMA/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_006_transformed.ome.tiff"
cell_image_fp=snakemake.input["IMCmask"]

resolution = float(snakemake.params['resolution'])
logging.info("Read Mask")
# read mask
cell_mask = imread(cell_image_fp)

if resolution != 1:
    wn = int(cell_mask.shape[0]*resolution)
    hn = int(cell_mask.shape[1]*resolution)
    cell_mask = cv2.resize(cell_mask, (hn,wn), interpolation=cv2.INTER_NEAREST_EXACT)


logging.info("Find bounding box")
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

logging.info("Cells raster to polygons")
# convert cell mask raster data to polygons for transformation
cell_shapes = []
cell_indices = []
for cell_idx in np.unique(cell_mask):
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
rs = RegShapes(cell_shapes_translated)
output_fp_shapes=snakemake.output["IMCmask_shape_transformed"]
rs.save_shape_data(output_fp_shapes, transformed=False)

logging.info("Save cell indices")
output_fp_indices=snakemake.output["cell_indices"]
with open(output_fp_indices, "wb") as f:
    pickle.dump(cell_indices_translated, f)


logging.info("Finished")
