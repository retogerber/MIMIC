import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
# sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0],"workflow","scripts","utils")))
import numpy as np
import json
import skimage
import skimage.exposure
import shapely
import rembg
from segment_anything import sam_model_registry, SamPredictor
import torch
import SimpleITK as sitk
import numpy as np
import cv2
import matplotlib.pyplot as plt
from image_utils import readimage_crop, get_image_shape
from utils import setNThreads, snakeMakeMock
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()

    snakemake.params["input_spacing_preIMC"] = 0.22537
    snakemake.params["input_spacing_IMC"] = 0.22537
    snakemake.params["input_spacing_IMC_location"] = 0.22537
    snakemake.params["output_spacing"] = 1
    snakemake.params["non_tissue_proportion_lower"] = 0.05
    snakemake.params["non_tissue_proportion_upper"] = 0.5
    snakemake.params["pixel_expansion"] = 501
    snakemake.input["preIMC"] = "results/NASH_HCC_TMA/data/preIMC/NASH_HCC_TMA_preIMC.ome.tiff"
    snakemake.input["IMC_mask"] = "results/NASH_HCC_TMA/data/IMC_mask/NASH_HCC_TMA_IMC_transformed_on_preIMC.ome.tiff"
    snakemake.input["IMC_location_on_preIMC"] = "results/NASH_HCC_TMA/data/IMC_location/NASH_HCC_TMA_IMC_mask_on_preIMC_B5.geojson"
    snakemake.input['sam_weights'] = "results/Misc/sam_vit_h_4b8939.pth"
    snakemake.input['contours_in'] = "results/NASH_HCC_TMA/data/preIMC/NASH_HCC_TMA-2_011_preIMC_landmark_regions.json"

    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
# logging_utils.logging_setup(None)
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
input_spacing_preIMC = snakemake.params["input_spacing_preIMC"]
input_spacing_IMC = snakemake.params["input_spacing_IMC"]
input_spacing_IMC_location = snakemake.params["input_spacing_IMC_location"]
output_spacing = snakemake.params["output_spacing"]
non_tissue_proportion_lower = snakemake.params["non_tissue_proportion_lower"]
non_tissue_proportion_upper = snakemake.params["non_tissue_proportion_upper"]
pixel_expansion = snakemake.params["pixel_expansion"]

# inputs
microscopy_file = snakemake.input['preIMC']
IMC_file=snakemake.input["IMC_mask"]
IMC_location=snakemake.input["IMC_location_on_preIMC"]
if isinstance(IMC_location, list):
    IMC_location = IMC_location[0]
contours_file_in=snakemake.input['contours_in']
CHECKPOINT_PATH = snakemake.input["sam_weights"]

# outputs
IMC_to_preIMC_error_output = snakemake.output["IMC_to_preIMC_error"]
IMC_to_preIMC_error_plot = snakemake.output["IMC_to_preIMC_error_plot"]

logging.info("core bounding box extraction")
IMC_geojson = json.load(open(IMC_location, "r"))
if isinstance(IMC_geojson,list):
    IMC_geojson=IMC_geojson[0]
boundary_points = np.array(IMC_geojson['geometry']['coordinates'])[0,:,:]
xmin=np.min(boundary_points[:,1])
xmax=np.max(boundary_points[:,1])
ymin=np.min(boundary_points[:,0])
ymax=np.max(boundary_points[:,0])

expand_microns = 21
logging.info(f"bounding box expansion: {expand_microns} microns")
s1f = input_spacing_preIMC/input_spacing_IMC_location
bb1 = [int(xmin/s1f-expand_microns/input_spacing_preIMC),int(ymin/s1f-expand_microns/input_spacing_preIMC),int(xmax/s1f+expand_microns/input_spacing_preIMC),int(ymax/s1f+expand_microns/input_spacing_preIMC)]
logging.info(f"bounding box preIMC: {bb1}")
bb2 = [int(xmin/s1f-expand_microns/input_spacing_IMC),int(ymin/s1f-expand_microns/input_spacing_IMC),int(xmax/s1f+expand_microns/input_spacing_IMC),int(ymax/s1f+expand_microns/input_spacing_IMC)]
logging.info(f"bounding box IMC: {bb2}")

logging.info("preIMC image")
preIMC_shape = get_image_shape(microscopy_file)
preIMC_image = readimage_crop(microscopy_file, bb1)

logging.info("preIMC image processing")
logging.info("Calculate standard deviation of hue channel")
radius_microns = 3
preIMC_image_H = cv2.cvtColor(preIMC_image, cv2.COLOR_RGB2HSV)[:,:,0].astype(int)
radius = int(radius_microns/input_spacing_preIMC)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
kernel = kernel/np.sum(kernel)
preIMC_image_H_std = np.sqrt(cv2.filter2D(preIMC_image_H**2, ddepth=cv2.CV_64F, kernel=kernel) - cv2.filter2D(preIMC_image_H, ddepth=cv2.CV_64F, kernel=kernel)**2)
preIMC_image_H_std[np.isnan(preIMC_image_H_std)]=0

logging.info("Get Threshold for tissue using rembg")
model_name = "isnet-general-use"
os.environ["OMP_NUM_THREADS"] = str(snakemake.threads)
rembg_session = rembg.new_session(model_name)
wr = rembg.remove(preIMC_image_H_std, only_mask=True, session=rembg_session)
# weight based on rembg mask
th1 = skimage.filters.threshold_otsu(wr, nbins=256)
th = np.quantile(preIMC_image_H_std[wr<th1],0.005)
logging.info(f"Threshold: {th}")

logging.info("Thresholding and filtering")
preIMC_image = (preIMC_image_H_std>th).astype(np.uint8)
preIMC_image = cv2.erode(preIMC_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius//2*2+1, radius//2*2+1)), preIMC_image)
preIMC_image = preIMC_image.astype(bool)
preIMC_image = ~skimage.morphology.remove_small_objects(~preIMC_image,(5/input_spacing_preIMC)**2*np.pi)
preIMC_image = preIMC_image.astype(np.uint8)*255

logging.info("Setup segment anything model")
DEVICE = 'cpu'
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)

logging.info("Connected components")
ccout = cv2.connectedComponentsWithStats(np.bitwise_not(preIMC_image), connectivity=8)

# get contour of largest area
logging.info("Contours")
ptls = []
for i in range(1,ccout[0]):
    ct = cv2.findContours((ccout[1]==i).astype(np.uint8)*255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pt = shapely.Polygon(np.squeeze(ct[0][0])).representative_point()
    pt = np.array(pt.xy).T[0]
    ptls.append(pt)
pts = np.array(ptls)

logging.info("Read image")
img = readimage_crop(microscopy_file, bb1)

logging.info("Predict")
predictor = SamPredictor(sam)
logging.info("\tset image")
predictor.set_image(img)
logging.info("\tset points and bboxes")
boxs = [[ccout[2][i,0],ccout[2][i,1],ccout[2][i,0]+ccout[2][i,2],ccout[2][i,1]+ccout[2][i,3]] for i in range(1,ccout[0])]
tpts = torch.tensor(pts[:,np.newaxis,:])
ilabs = torch.tensor(np.array(np.ones(len(pts), dtype=int))[:,np.newaxis])
transformed_boxes = predictor.transform.apply_boxes_torch(torch.tensor(boxs), img.shape[:2])
transformed_pts = predictor.transform.apply_coords_torch(tpts, img.shape[:2])
logging.info(f"\tnumber of points: {len(pts)}")
masks, _, _ = predictor.predict_torch(
    point_coords=transformed_pts,
    point_labels=ilabs,
    multimask_output=False,
    boxes=transformed_boxes
)
mask = (np.array(torch.sum(masks,axis=(0,1)))>0).astype(np.uint8)
preIMC_image = mask*255

logging.info("Read preIMC contours")
data = json.load(open(contours_file_in, "r"))
regionsls = data['regions']
regions = [np.array(reg,dtype=np.uint64) for reg in regionsls]
bboxes = data['bboxes']
regions_scaled = [(reg.astype(int)-pixel_expansion)/input_spacing_preIMC for reg in regions]
regions_scaled = [np.clip(reg,0,np.array(preIMC_image.shape)-1).astype(np.uint64) for reg in regions_scaled]
cv2.imwrite("mask1.png", (mask>0)*255)
for k,reg in enumerate(regions_scaled):
    mask = cv2.drawContours(
        mask, 
        [reg], 
        -1, 
        255,
        -1)

cv2.imwrite("mask2.png", (mask>0)*255)

prop_area_non_tissue = np.sum(preIMC_image>0)/np.prod(preIMC_image.shape)
logging.info(f"Non tissue area proportion: {prop_area_non_tissue}")
if prop_area_non_tissue<non_tissue_proportion_lower or prop_area_non_tissue>non_tissue_proportion_upper:
    logging.info(f"Non tissue area proportion is not within bounds: {non_tissue_proportion_lower} < {prop_area_non_tissue} < {non_tissue_proportion_upper}")
    logging.info("write metrics")
    metric_dict = {
        "IMC_to_preIMC_mean_error": np.nan,
        "IMC_to_preIMC_quantile05_error": np.nan,
        "IMC_to_preIMC_quantile25_error": np.nan,
        "IMC_to_preIMC_quantile50_error": np.nan,
        "IMC_to_preIMC_quantile75_error": np.nan,
        "IMC_to_preIMC_quantile95_error": np.nan,
        "IMC_to_preIMC_n_points": np.nan,
        "proportion_area_no_tissue_preIMC_IMC_beforereg": np.nan,
        "proportion_area_no_tissue_preIMC_IMC_afterreg": np.nan,
        "preIMC_proportion_area_no_tissue": prop_area_non_tissue,
        "IMC_proportion_area_no_tissue": np.nan,
        "IMCmask_distance_tissue_to_nearest_cell_quantile05": np.nan,
        "IMCmask_distance_tissue_to_nearest_cell_quantile25": np.nan,
        "IMCmask_distance_tissue_to_nearest_cell_quantile50": np.nan,
        "IMCmask_distance_tissue_to_nearest_cell_quantile75": np.nan,
        "IMCmask_distance_tissue_to_nearest_cell_quantile95": np.nan,
        "IMCmask_distance_tissue_to_nearest_cell_quantile100": np.nan 
    }
    json.dump(metric_dict, open(IMC_to_preIMC_error_output,"w"))
    from pathlib import Path
    Path(IMC_to_preIMC_error_plot).touch()
    sys.exit(0)

logging.info("invert preIMC image")
preIMC_image = cv2.bitwise_not(preIMC_image)

logging.info("IMC image")
IMC_image = readimage_crop(IMC_file, bb2)

logging.info("rescale images to micron resolution for registration")
wn = int(IMC_image.shape[0]*input_spacing_preIMC)
hn = int(IMC_image.shape[1]*input_spacing_preIMC)
IMC_image = cv2.resize(IMC_image, (hn,wn), interpolation=cv2.INTER_NEAREST)
wn = int(preIMC_image.shape[0]*input_spacing_preIMC)
hn = int(preIMC_image.shape[1]*input_spacing_preIMC)
preIMC_image = cv2.resize(preIMC_image, (hn,wn), interpolation=cv2.INTER_NEAREST)


npw = np.sum(np.logical_and(preIMC_image==0,IMC_image>0))
pnpw = npw/np.sum(preIMC_image==0)
logging.info(f"Proportion of area with no tissue in preIMC and cell mask in IMC: {pnpw}")



def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.8f} "
        + f": {method.GetOptimizerPosition()}"
    )

def prepare_register(init_transform_inp):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation()
    R.SetMetricSamplingStrategy(R.REGULAR)
    R.SetMetricSamplingPercentage(0.5, seed=1234)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsGradientDescentLineSearch(learningRate=1e-3, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10, estimateLearningRate = sitk.ImageRegistrationMethod.Once)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(init_transform_inp)
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    return R

logging.info("register IMC to preIMC")
IMCimg = (IMC_image==0).astype(np.uint8)
# distance to zero
IMCimg = cv2.distanceTransform(IMCimg, cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
IMCmask_distqs = np.nanquantile(IMCimg[IMCimg>0],[0.05, 0.25, 0.5, 0.75, 0.95, 1])

# invert
IMCimg = 1/(IMCimg+1)
IMCimg[IMCimg==1]=0
preIMCimg = (preIMC_image==0).astype(np.uint8)
preIMCimg = cv2.distanceTransform(preIMCimg, cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE) 
preIMCimg = 1/(preIMCimg+1)
preIMCimg[preIMCimg==1]=0
IMCimg = sitk.GetImageFromArray(IMCimg)
preIMCimg = sitk.GetImageFromArray(preIMCimg)
init_transform_euler1 = sitk.Euler2DTransform()
init_transform_euler1.SetCenter([IMC_image.shape[0]//2,IMC_image.shape[1]//2])
R = prepare_register(init_transform_euler1)
try:
    transform = R.Execute(preIMCimg, IMCimg)
except Exception as e:
    logging.info(f"Registration failed: {e}")
    logging.info("write metrics")
    metric_dict = {
        "IMC_to_preIMC_mean_error": np.nan,
        "IMC_to_preIMC_quantile05_error": np.nan,
        "IMC_to_preIMC_quantile25_error": np.nan,
        "IMC_to_preIMC_quantile50_error": np.nan,
        "IMC_to_preIMC_quantile75_error": np.nan,
        "IMC_to_preIMC_quantile95_error": np.nan,
        "IMC_to_preIMC_n_points": np.nan,
        "proportion_area_no_tissue_preIMC_IMC_beforereg": pnpw,
        "proportion_area_no_tissue_preIMC_IMC_afterreg": np.nan,
        "preIMC_proportion_area_no_tissue": prop_area_non_tissue,
        "IMC_proportion_area_no_tissue": np.sum(IMC_image==0)/np.prod(IMC_image.shape),
        "IMCmask_distance_tissue_to_nearest_cell_quantile05": IMCmask_distqs[0],
        "IMCmask_distance_tissue_to_nearest_cell_quantile25": IMCmask_distqs[1],
        "IMCmask_distance_tissue_to_nearest_cell_quantile50": IMCmask_distqs[2],
        "IMCmask_distance_tissue_to_nearest_cell_quantile75": IMCmask_distqs[3],
        "IMCmask_distance_tissue_to_nearest_cell_quantile95": IMCmask_distqs[4],
        "IMCmask_distance_tissue_to_nearest_cell_quantile100": IMCmask_distqs[5]
    }
    json.dump(metric_dict, open(IMC_to_preIMC_error_output,"w"))
    from pathlib import Path
    Path(IMC_to_preIMC_error_plot).touch()
    sys.exit(0)



def resample_image(transform, fixed, moving, default_value=0.0):
    resampler = sitk.ResampleImageFilter()
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(transform)
    resampler.SetReferenceImage(fixed)
    resampler.SetOutputSpacing(fixed.GetSpacing())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return sitk.GetArrayFromImage(resampler.Execute(moving))

IMCimgmov = resample_image(transform, preIMCimg, IMCimg, 1)
npwa = np.sum(np.logical_and(preIMC_image==0,IMCimgmov==0))
pnpwa = npwa/np.sum(preIMC_image==0)
logging.info(f"Proportion of area with no tissue in preIMC and cell mask in IMC before registration: {pnpw}")
logging.info(f"Proportion of area with no tissue in preIMC and cell mask in IMC after registration: {pnpwa}")

logging.info(f"Transform parameters: {transform.GetParameters()}")


logging.info("evaluate registration")
logging.info("distance between grid points before and after registration")
# create coordinate grid
def create_grid(shape):
    x = np.linspace(0, shape[0]-1, shape[0]//20)
    y = np.linspace(0, shape[1]-1, shape[1]//20)
    grid = np.meshgrid(x,y)
    grid = np.array(grid)
    grid = np.moveaxis(grid, [0,1,2], [2,0,1])
    grid = grid.reshape(-1,2)
    return grid

grid = create_grid(IMC_image.shape)
grid_trans = np.stack([transform.GetInverse().TransformPoint(pt) for pt in grid])

# distance between points
distances = np.linalg.norm(grid-grid_trans, axis=1)
logging.info(f"Distance quantiles: \n\tQ05: {np.quantile(distances, 0.05)}\n\tQ25: {np.quantile(distances, 0.25)}\n\tQ50: {np.quantile(distances, 0.5)}\n\tQ75: {np.quantile(distances, 0.75)}\n\tQ95: {np.quantile(distances, 0.95)}")

logging.info("write metrics")
metric_dict = {
    "IMC_to_preIMC_mean_error": np.mean(distances),
    "IMC_to_preIMC_quantile05_error": np.quantile(distances, 0.05),
    "IMC_to_preIMC_quantile25_error": np.quantile(distances, 0.25),
    "IMC_to_preIMC_quantile50_error": np.quantile(distances, 0.5),
    "IMC_to_preIMC_quantile75_error": np.quantile(distances, 0.75),
    "IMC_to_preIMC_quantile95_error": np.quantile(distances, 0.95),
    "IMC_to_preIMC_n_points": len(distances),
    "proportion_area_no_tissue_preIMC_IMC_beforereg": pnpw,
    "proportion_area_no_tissue_preIMC_IMC_afterreg": pnpwa,
    "preIMC_proportion_area_no_tissue": prop_area_non_tissue,
    "IMC_proportion_area_no_tissue": np.sum(IMC_image==0)/np.prod(IMC_image.shape),
    "IMCmask_distance_tissue_to_nearest_cell_quantile05": IMCmask_distqs[0],
    "IMCmask_distance_tissue_to_nearest_cell_quantile25": IMCmask_distqs[1],
    "IMCmask_distance_tissue_to_nearest_cell_quantile50": IMCmask_distqs[2],
    "IMCmask_distance_tissue_to_nearest_cell_quantile75": IMCmask_distqs[3],
    "IMCmask_distance_tissue_to_nearest_cell_quantile95": IMCmask_distqs[4],
    "IMCmask_distance_tissue_to_nearest_cell_quantile100": IMCmask_distqs[5]
}
json.dump(metric_dict, open(IMC_to_preIMC_error_output,"w"))

logging.info("plotting")
fig, ax = plt.subplots(2,3, figsize=(15,10))
ax[0,0].imshow(preIMC_image)
ax[0,0].set_title("preIMC tissue mask")
ax[0,1].imshow(IMC_image>0)
ax[0,1].set_title("IMC cell mask")
ax[0,2].imshow(((IMC_image>0)*127+preIMC_image/2)/4+np.logical_and(IMC_image>0,preIMC_image==0)*192)
ax[0,2].set_title(f"overlay \nProportion of area with no tissue \nin preIMC and cell mask in IMC: {pnpw:.4f}")
ax[1,0].hist(distances, bins=100)
ax[1,0].set_title("distances of grid points before and after registration")
ax[1,0].set_xlabel("distance in microns")
ax[1,1].imshow(IMCimgmov==0)
ax[1,1].set_title("IMC cell mask after registration")
ax[1,2].imshow(((IMCimgmov==0)*127+preIMC_image/2)/4+np.logical_and(IMCimgmov==0,preIMC_image==0)*192)
ax[1,2].set_title(f"overlay after registration\nProportion of area with no tissue \nin preIMC and cell mask in IMC: {pnpwa:.4f}")
plt.savefig(IMC_to_preIMC_error_plot)


logging.info("Finished")
