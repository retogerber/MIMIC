import matplotlib.pyplot as plt
import numpy as np
import json
from segment_anything import sam_model_registry
import skimage
import skimage.exposure
import SimpleITK as sitk
import re
import numpy as np
import cv2
from image_utils import readimage_crop, convert_and_scale_image, saveimage_tile, subtract_postIMS_grid, extract_mask, get_image_shape, sam_core, preprocess_mask
from utils import setNThreads, snakeMakeMock
import sys,os
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["input_spacing_1"] = 1
    snakemake.params["input_spacing_2"]
    snakemake.params["input_spacing_IMC_location"] = 0.22537
    snakemake.params["output_spacing"] = 1
    snakemake.input["sam_weights"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/Misc/sam_vit_h_4b8939.pth"
    snakemake.input["microscopy_image_1"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/preIMS/NASH_HCC_TMA-2_020_transformed_on_postIMS.ome.tiff"
    snakemake.input["microscopy_image_2"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/postIMS/NASH_HCC_TMA_postIMS.ome.tiff"
    snakemake.input["IMC_location"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/IMC_location/NASH_HCC_TMA_IMC_mask_on_postIMS_C9.geojson"

    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
input_spacing_1 = snakemake.params["input_spacing_1"]
input_spacing_2 = snakemake.params["input_spacing_2"]
input_spacing_IMC_location = snakemake.params["input_spacing_IMC_location"]
output_spacing = snakemake.params["output_spacing"]
# inputs
microscopy_file_1 = snakemake.input['microscopy_image_1']
microscopy_file_2 = snakemake.input['microscopy_image_2']
IMC_location=snakemake.input["IMC_location"]
if isinstance(IMC_location, list):
    IMC_location = IMC_location[0]
CHECKPOINT_PATH = snakemake.input["sam_weights"]
DEVICE = 'cpu'
MODEL_TYPE = "vit_h"

m = re.search("[a-zA-Z]*(?=.ome.tiff$)",os.path.basename(microscopy_file_1))
comparison_to = m.group(0)
comparison_from = os.path.basename(os.path.dirname(microscopy_file_1))
assert(comparison_to in ["preIMC","preIMS","postIMS"])
assert(comparison_from in ["postIMC","preIMC","preIMS"])


logging.info("core bounding box extraction")
IMC_geojson = json.load(open(IMC_location, "r"))
if isinstance(IMC_geojson,list):
    IMC_geojson=IMC_geojson[0]
boundary_points = np.array(IMC_geojson['geometry']['coordinates'])[0,:,:]
xmin=np.min(boundary_points[:,1])
xmax=np.max(boundary_points[:,1])
ymin=np.min(boundary_points[:,0])
ymax=np.max(boundary_points[:,0])

s1f = input_spacing_1/input_spacing_IMC_location
bb1 = [int(xmin/s1f-201/input_spacing_1),int(ymin/s1f-201/input_spacing_1),int(xmax/s1f+201/input_spacing_1),int(ymax/s1f+201/input_spacing_1)]
logging.info(f"bounding box whole image 1: {bb1}")

s2f = input_spacing_2/input_spacing_IMC_location
bb2 = [int(xmin/s2f-201/input_spacing_2),int(ymin/s2f-201/input_spacing_2),int(xmax/s2f+201/input_spacing_2),int(ymax/s2f+201/input_spacing_2)]
logging.info(f"bounding box whole image 2: {bb2}")

m2full_shape = get_image_shape(microscopy_file_1)
bb3 = [int(xmin/s1f-1251/input_spacing_1),int(ymin/s1f-1251/input_spacing_1),int(xmax/s1f+1251/input_spacing_1),int(ymax/s1f+1251/input_spacing_1)]
bb3[0] = bb3[0] if bb3[0]>=0 else 0
bb3[1] = bb3[1] if bb3[1]>=0 else 0
bb3[2] = bb3[2] if bb3[2]<=m2full_shape[0] else m2full_shape[0]
bb3[3] = bb3[3] if bb3[3]<=m2full_shape[1] else m2full_shape[1]
logging.info(f"bounding box mask whole image 1: {bb3}")


logging.info("load microscopy image 1")
microscopy_image_1 = readimage_crop(microscopy_file_1, bb1)
microscopy_image_1 = convert_and_scale_image(microscopy_image_1, input_spacing_1/output_spacing)

logging.info("load microscopy image 2")
microscopy_image_2 = readimage_crop(microscopy_file_2, bb2)
microscopy_image_2 = convert_and_scale_image(microscopy_image_2, input_spacing_2/output_spacing)

logging.info("Extract mask for microscopy image 1")
mask_2 = extract_mask(microscopy_file_1, bb3, rescale = input_spacing_1/output_spacing, is_postIMS=False)[0,:,:]
xb = int((mask_2.shape[0]-microscopy_image_1.shape[0])/2)
yb = int((mask_2.shape[1]-microscopy_image_1.shape[1])/2)
wn = microscopy_image_1.shape[0]
hn = microscopy_image_1.shape[1]
mask_2 = cv2.resize(mask_2[xb:-xb,yb:-yb].astype(np.uint8), (hn,wn), interpolation=cv2.INTER_NEAREST)
mask_2_proportion = np.sum(mask_2)/np.prod(mask_2.shape)
logging.info(f"proportion image 1 covered by mask (rembg): {mask_2_proportion:5.4}")

bb0 = [int(xmin/s1f),int(ymin/s1f),int(xmax/s1f),int(ymax/s1f)]
IMC_mask_proportion = ((bb0[2]-bb0[0])*(bb0[3]-bb0[1]))/((bb1[2]-bb1[0])*(bb1[3]-bb1[1]))
if mask_2_proportion < IMC_mask_proportion:
    logging.info("Extract mask for microscopy image 1 with SAM")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    saminp = readimage_crop(microscopy_file_1, bb1)
    # to gray scale, rescale
    saminp = convert_and_scale_image(saminp, input_spacing_1/output_spacing)
    saminp = np.stack([saminp, saminp, saminp], axis=2)
    # run SAM segmentation model
    masks, scores1 = sam_core(saminp, sam)
    # postprocess
    masks = np.stack([skimage.morphology.convex_hull_image(preprocess_mask(msk,1)) for msk in masks ])
    tmpareas = np.array([np.sum(im) for im in masks])
    indmax = np.argmax(tmpareas/(masks.shape[1]*masks.shape[2]))
    mask_2 = masks[indmax,:,:].astype(np.uint8)
    mask_2_proportion = np.sum(mask_2)/np.prod(mask_2.shape)
    logging.info(f"proportion image 1 covered by mask (SAM): {mask_2_proportion:5.4}")
    if mask_2_proportion < IMC_mask_proportion:
        logging.info("Mask not detected! Using all pixels.")
        mask_2 = np.ones(microscopy_image_1.shape, dtype=np.uint8)

xmax = min([microscopy_image_1.shape[0],microscopy_image_2.shape[0]])
ymax = min([microscopy_image_1.shape[1],microscopy_image_2.shape[1]])
imcbbox_outer = [0,0,xmax,ymax]
logging.info(f"imcbbox_outer: {imcbbox_outer}")

logging.info(f"Detect postIMS grid")
img = microscopy_image_2[imcbbox_outer[0]:imcbbox_outer[2],imcbbox_outer[1]:imcbbox_outer[3]].copy()
mask_2_on_2 = cv2.resize(mask_2, (microscopy_image_2.shape[1], microscopy_image_2.shape[0]), interpolation=cv2.INTER_NEAREST)
cv2.morphologyEx(mask_2_on_2, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8),mask_2_on_2, iterations=1)
img[mask_2_on_2==0]=0
out = subtract_postIMS_grid(img)
out[mask_2_on_2==0]=0


logging.info(f"Create mask from postIMS grid")
diffimg = img.astype(float)-out.astype(float)
diffimg = cv2.normalize(src=diffimg, dst=diffimg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
diffimg = cv2.medianBlur(diffimg, ksize=3)
diffimg_red = diffimg[mask_2_on_2>0]
binm = cv2.threshold( diffimg_red, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
binimg=(diffimg>binm[0]).astype(np.uint8)
logging.info(f"shape: {binimg.shape}")
logging.info(f"type: {binimg.dtype}")

cv2.morphologyEx(src=binimg, op=cv2.MORPH_CLOSE, kernel=np.ones((5,5),np.uint8),dst=binimg, iterations=1)
out[binimg==0]=img[binimg==0]
mask_grid = np.logical_and(binimg>0,mask_2_on_2>0)


def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.8f} "
        + f": {method.GetOptimizerPosition()}"
    )

logging.info("Run SITK registration")
fixed = sitk.GetImageFromArray(microscopy_image_2[imcbbox_outer[0]:imcbbox_outer[2],imcbbox_outer[1]:imcbbox_outer[3]].astype(float))
moving = sitk.GetImageFromArray(microscopy_image_1[imcbbox_outer[0]:imcbbox_outer[2],imcbbox_outer[1]:imcbbox_outer[3]].astype(float))

R = sitk.ImageRegistrationMethod()
R.SetMetricAsMeanSquares()
R.SetMetricSamplingStrategy(R.NONE)
R.SetMetricFixedMask(sitk.GetImageFromArray(mask_grid.astype(np.uint8)))
R.SetInterpolator(sitk.sitkLinear)
R.SetOptimizerAsGradientDescentLineSearch(
    learningRate=1, numberOfIterations=1000, 
    convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    lineSearchEpsilon=0.001, lineSearchMaximumIterations=50,
)
R.SetOptimizerScalesFromPhysicalShift()
R.SetInitialTransform(sitk.AffineTransform(2))
R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

# run registration
transform = R.Execute(fixed, moving)

# transform image
transformed_image = sitk.GetArrayFromImage(sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, moving.GetPixelID()))
transformed_image[~mask_grid]=0

fixed_img = microscopy_image_2[imcbbox_outer[0]:imcbbox_outer[2],imcbbox_outer[1]:imcbbox_outer[3]].copy()
fixed_img[~mask_grid]=0

# import plotly.subplots as sp
# import plotly.express as px
# # Create a subplot with 1 row and 2 columns
# fig = sp.make_subplots(rows=1, cols=2)
# # Add transformed_image to the first subplot
# fig.add_trace(
#     px.imshow(transformed_image).data[0],
#     row=1, col=1
# )
# # Add fixed_image to the second subplot
# fig.add_trace(
#     px.imshow(fixed_img).data[0],
#     row=1, col=2
# )
# fig.show()

logging.info("Save image")
# Stitch the images side by side
stitched_image = np.hstack((transformed_image, fixed_img))
saveimage_tile(stitched_image, snakemake.output["registered_out"] ,1)

translation = -np.array(transform.GetTranslation())
affinemat = transform.GetMatrix()

reg_measure_dic = {
    f"{comparison_from}_to_{comparison_to}_sitk_global_x_shift": str(translation[0]),
    f"{comparison_from}_to_{comparison_to}_sitk_global_y_shift": str(translation[1]),
    f"{comparison_from}_to_{comparison_to}_sitk_global_affine_matrix_00": str(affinemat[0]),
    f"{comparison_from}_to_{comparison_to}_sitk_global_affine_matrix_01": str(affinemat[1]),
    f"{comparison_from}_to_{comparison_to}_sitk_global_affine_matrix_10": str(affinemat[2]),
    f"{comparison_from}_to_{comparison_to}_sitk_global_affine_matrix_11": str(affinemat[3]),
    }

logging.info("Save json")
json.dump(reg_measure_dic, open(snakemake.output["error_stats"],"w"))

logging.info("Finished")