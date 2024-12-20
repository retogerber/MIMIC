import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
import numpy as np
import json
import skimage
import skimage.exposure
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
    snakemake.input["preIMC"] = "results/cirrhosis_TMA/data/preIMC/cirrhosis_TMA_preIMC.ome.tiff"
    snakemake.input["IMC_mask"] = "results/cirrhosis_TMA/data/IMC_mask/cirrhosis_TMA_IMC_transformed_on_preIMC.ome.tiff"
    snakemake.input["IMC_location_on_preIMC"] = "results/cirrhosis_TMA/data/IMC_location/cirrhosis_TMA_registered_IMC_mask_on_preIMC_A5.geojson"

    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
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

# inputs
microscopy_file = snakemake.input['preIMC']
IMC_file=snakemake.input["IMC_mask"]
IMC_location=snakemake.input["IMC_location_on_preIMC"]
if isinstance(IMC_location, list):
    IMC_location = IMC_location[0]

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
preIMC_image = cv2.cvtColor(preIMC_image, cv2.COLOR_RGB2GRAY) 

logging.info("Get Threshold for tissue")
collapse_microns = expand_microns + 5
reverse_expansion_bbox = [int(collapse_microns/input_spacing_preIMC), int(collapse_microns/input_spacing_preIMC), int(preIMC_image.shape[0]-collapse_microns/input_spacing_preIMC), int(preIMC_image.shape[1]-collapse_microns/input_spacing_preIMC)]
preIMC_image_sub = preIMC_image[reverse_expansion_bbox[0]:reverse_expansion_bbox[2],reverse_expansion_bbox[1]:reverse_expansion_bbox[3]]
logging.info(f"\tpreIMC image shape: {preIMC_image.shape}")
logging.info(f"\tpreIMC image shape for threshold: {preIMC_image_sub.shape}")
try:
    th = skimage.filters.threshold_minimum(preIMC_image_sub, nbins=256)
except:
    th = skimage.filters.threshold_otsu(preIMC_image_sub, nbins=256)
logging.info(f"Threshold: {th}")

logging.info("Thresholding and filtering")
preIMC_image = (preIMC_image>th).astype(np.uint8)
preIMC_image = cv2.medianBlur(preIMC_image, int(3/input_spacing_preIMC), preIMC_image)
preIMC_image = preIMC_image.astype(bool)
preIMC_image = skimage.morphology.remove_small_objects(preIMC_image,(5/input_spacing_preIMC)**2*np.pi)
preIMC_image = preIMC_image.astype(np.uint8)*255

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
        "IMC_proportion_area_no_tissue": np.nan 
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
    R.SetMetricAsMeanSquares()
    R.SetMetricSamplingStrategy(R.REGULAR)
    R.SetMetricSamplingPercentage(0.5, seed=1234)
    R.SetInterpolator(sitk.sitkLinear)
    # R.SetOptimizerAsGradientDescent(
    #     learningRate=1, numberOfIterations=1000, 
    #     convergenceMinimumValue=1e-9, convergenceWindowSize=50
    # )
    R.SetOptimizerAsGradientDescentLineSearch(learningRate=1e-3, numberOfIterations=1000, convergenceMinimumValue=1e-9, convergenceWindowSize=20, estimateLearningRate = sitk.ImageRegistrationMethod.Never)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(init_transform_inp)
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    return R

logging.info("register IMC to preIMC")
IMCimg = (IMC_image==0).astype(np.float32)
IMCimg = sitk.GetImageFromArray(IMCimg)
preIMCimg = (preIMC_image==0).astype(np.float32)
preIMCimg = sitk.GetImageFromArray(preIMCimg)
init_transform_euler1 = sitk.Euler2DTransform()
init_transform_euler1.SetCenter([IMC_image.shape[0]//2,IMC_image.shape[1]//2])
R = prepare_register(init_transform_euler1)
transform = R.Execute(preIMCimg, IMCimg)

def resample_image(transform, fixed, moving, default_value=0.0):
    resampler = sitk.ResampleImageFilter()
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(transform)
    resampler.SetReferenceImage(fixed)
    resampler.SetOutputSpacing(fixed.GetSpacing())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return sitk.GetArrayFromImage(resampler.Execute(moving))

IMCimgmov = resample_image(transform, preIMCimg, IMCimg)

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
grid_trans = np.stack([transform.TransformPoint(pt) for pt in grid])

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
    "IMC_proportion_area_no_tissue": np.sum(IMC_image==0)/np.prod(IMC_image.shape)
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
