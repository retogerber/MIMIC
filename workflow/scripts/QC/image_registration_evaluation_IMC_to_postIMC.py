import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
import numpy as np
import json
import tifffile
import wsireg
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

    snakemake.params["input_spacing_postIMC"] = 0.22537
    snakemake.params["input_spacing_IMC"] = 1
    snakemake.params["input_spacing_IMC_location"] = 0.22537
    snakemake.params["output_spacing"] = 1
    snakemake.params["non_tissue_proportion_lower"] = 0.05
    snakemake.params["non_tissue_proportion_upper"] = 0.5
    snakemake.input["postIMC"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/postIMC/test_combined_postIMC.ome.tiff"
    snakemake.input["IMC_mask"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_002.tiff"
    snakemake.input["IMC_location_on_postIMC"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMC_location/test_combined_IMC_mask_on_postIMC_B1.geojson"
    snakemake.input["IMC_to_postIMC_transform"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/registrations/IMC_to_postIMC/test_combined_B1/test_combined_B1-IMC_to_postIMC_transformations.json"

    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
input_spacing_postIMC = snakemake.params["input_spacing_postIMC"]
input_spacing_IMC = snakemake.params["input_spacing_IMC"]
input_spacing_IMC_location = snakemake.params["input_spacing_IMC_location"]
output_spacing = snakemake.params["output_spacing"]
non_tissue_proportion_lower = snakemake.params["non_tissue_proportion_lower"]
non_tissue_proportion_upper = snakemake.params["non_tissue_proportion_upper"]

# inputs
microscopy_file = snakemake.input['postIMC']
IMC_file=snakemake.input["IMC_mask"]
IMC_location=snakemake.input["IMC_location_on_postIMC"]
if isinstance(IMC_location, list):
    IMC_location = IMC_location[0]
transform_file_IMC_to_postIMC = snakemake.input["IMC_to_postIMC_transform"]

# outputs
IMC_to_postIMC_error_output = snakemake.output["IMC_to_postIMC_error"]
IMC_to_postIMC_error_plot = snakemake.output["IMC_to_postIMC_error_plot"]

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
s1f = input_spacing_postIMC/input_spacing_IMC_location
bb1 = [int(xmin/s1f-expand_microns/input_spacing_postIMC),int(ymin/s1f-expand_microns/input_spacing_postIMC),int(xmax/s1f+expand_microns/input_spacing_postIMC),int(ymax/s1f+expand_microns/input_spacing_postIMC)]
logging.info(f"bounding box postIMC: {bb1}")

logging.info("postIMC image")
postIMC_shape = get_image_shape(microscopy_file)
postIMC_image = readimage_crop(microscopy_file, bb1)
postIMC_image = cv2.cvtColor(postIMC_image, cv2.COLOR_RGB2GRAY) 

logging.info("Get Threshold for tissue")
collapse_microns = expand_microns + 5
reverse_expansion_bbox = [int(collapse_microns/input_spacing_postIMC), int(collapse_microns/input_spacing_postIMC), int(postIMC_image.shape[0]-collapse_microns/input_spacing_postIMC), int(postIMC_image.shape[1]-collapse_microns/input_spacing_postIMC)]
postIMC_image_sub = postIMC_image[reverse_expansion_bbox[0]:reverse_expansion_bbox[2],reverse_expansion_bbox[1]:reverse_expansion_bbox[3]]
logging.info(f"\tpostIMC image shape: {postIMC_image.shape}")
logging.info(f"\tpostIMC image shape for threshold: {postIMC_image_sub.shape}")
try:
    th = skimage.filters.threshold_minimum(postIMC_image_sub, nbins=256)
except:
    th = skimage.filters.threshold_otsu(postIMC_image_sub, nbins=256)
logging.info(f"Threshold: {th}")

logging.info("Thresholding and filtering")
postIMC_image = (postIMC_image>th).astype(np.uint8)
postIMC_image = cv2.medianBlur(postIMC_image, int(3/input_spacing_postIMC), postIMC_image)
postIMC_image = postIMC_image.astype(bool)
postIMC_image = skimage.morphology.remove_small_objects(postIMC_image,(5/input_spacing_postIMC)**2*np.pi)
postIMC_image = postIMC_image.astype(np.uint8)*255

prop_area_non_tissue = np.sum(postIMC_image>0)/np.prod(postIMC_image.shape)
logging.info(f"Non tissue area proportion: {prop_area_non_tissue}")
if prop_area_non_tissue<non_tissue_proportion_lower or prop_area_non_tissue>non_tissue_proportion_upper:
    logging.info(f"Non tissue area proportion is not within bounds: {non_tissue_proportion_lower} < {prop_area_non_tissue} < {non_tissue_proportion_upper}")
    logging.info("write metrics")
    metric_dict = {
        "IMC_to_postIMC_mean_error": np.nan,
        "IMC_to_postIMC_quantile05_error": np.nan,
        "IMC_to_postIMC_quantile25_error": np.nan,
        "IMC_to_postIMC_quantile50_error": np.nan,
        "IMC_to_postIMC_quantile75_error": np.nan,
        "IMC_to_postIMC_quantile95_error": np.nan,
        "IMC_to_postIMC_n_points": np.nan,
        "proportion_area_no_tissue_postIMC_IMC_beforereg": np.nan,
        "proportion_area_no_tissue_postIMC_IMC_afterreg": np.nan,
        "postIMC_proportion_area_no_tissue": prop_area_non_tissue,
        "IMC_proportion_area_no_tissue": np.nan 
    }
    json.dump(metric_dict, open(IMC_to_postIMC_error_output,"w"))
    sys.exit(0)

logging.info("invert postIMC image")
postIMC_image = cv2.bitwise_not(postIMC_image)

logging.info("IMC image")
IMC_mask = tifffile.imread(IMC_file)
imcmask = sitk.GetImageFromArray(IMC_mask.astype(float))


transform_dic = json.load(open(transform_file_IMC_to_postIMC, "r"))
rt = wsireg.reg_transforms.reg_transform_seq.RegTransform(transform_dic)
transform = rt.itk_transform

def get_resampler(transform, fixed, default_value=0.0):
    resampler = sitk.ResampleImageFilter()
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(transform)
    resampler.SetReferenceImage(fixed)
    resampler.SetOutputSpacing(fixed.GetSpacing())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return resampler

def resample_image(resampler, moving):
    return sitk.GetArrayFromImage(resampler.Execute(moving))


# postimc = np.zeros(postIMC_shape[:2], dtype=float)
# postimc = sitk.GetImageFromArray(postimc)

# composite transform to directly crop for increased speed, essentially a translation transform plus a smaller reference image
composite = sitk.CompositeTransform(2)
composite.AddTransform(transform)
composite.AddTransform(sitk.TranslationTransform(2, [bb1[1], bb1[0]]))


postimc_crop = np.zeros([bb1[2]-bb1[0], bb1[3]-bb1[1]], dtype=float)
postimc_crop = sitk.GetImageFromArray(postimc_crop)
resampler = get_resampler(composite, postimc_crop)
IMC_image = resample_image(resampler, imcmask)

logging.info("rescale images to micron resolution for registration")
wn = int(IMC_image.shape[0]*input_spacing_postIMC)
hn = int(IMC_image.shape[1]*input_spacing_postIMC)
IMC_image = cv2.resize(IMC_image, (hn,wn), interpolation=cv2.INTER_NEAREST)
wn = int(postIMC_image.shape[0]*input_spacing_postIMC)
hn = int(postIMC_image.shape[1]*input_spacing_postIMC)
postIMC_image = cv2.resize(postIMC_image, (hn,wn), interpolation=cv2.INTER_NEAREST)


npw = np.sum(np.logical_and(postIMC_image==0,IMC_image>0))
pnpw = npw/np.sum(postIMC_image==0)
logging.info(f"Proportion of area with no tissue in postIMC and cell mask in IMC: {pnpw}")



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

logging.info("register IMC to postIMC")
IMCimg = (IMC_image==0).astype(np.float32)
IMCimg = sitk.GetImageFromArray(IMCimg)
postIMCimg = (postIMC_image==0).astype(np.float32)
postIMCimg = sitk.GetImageFromArray(postIMCimg)
init_transform_euler1 = sitk.Euler2DTransform()
init_transform_euler1.SetCenter([IMC_image.shape[0]//2,IMC_image.shape[1]//2])
R = prepare_register(init_transform_euler1)
transform = R.Execute(postIMCimg, IMCimg)

def resample_image(transform, fixed, moving, default_value=0.0):
    resampler = sitk.ResampleImageFilter()
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(transform)
    resampler.SetReferenceImage(fixed)
    resampler.SetOutputSpacing(fixed.GetSpacing())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return sitk.GetArrayFromImage(resampler.Execute(moving))

IMCimgmov = resample_image(transform, postIMCimg, IMCimg)

npwa = np.sum(np.logical_and(postIMC_image==0,IMCimgmov==0))
pnpwa = npwa/np.sum(postIMC_image==0)
logging.info(f"Proportion of area with no tissue in postIMC and cell mask in IMC before registration: {pnpw}")
logging.info(f"Proportion of area with no tissue in postIMC and cell mask in IMC after registration: {pnpwa}")

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
    "IMC_to_postIMC_mean_error": np.mean(distances),
    "IMC_to_postIMC_quantile05_error": np.quantile(distances, 0.05),
    "IMC_to_postIMC_quantile25_error": np.quantile(distances, 0.25),
    "IMC_to_postIMC_quantile50_error": np.quantile(distances, 0.5),
    "IMC_to_postIMC_quantile75_error": np.quantile(distances, 0.75),
    "IMC_to_postIMC_quantile95_error": np.quantile(distances, 0.95),
    "IMC_to_postIMC_n_points": len(distances),
    "proportion_area_no_tissue_postIMC_IMC_beforereg": pnpw,
    "proportion_area_no_tissue_postIMC_IMC_afterreg": pnpwa,
    "postIMC_proportion_area_no_tissue": prop_area_non_tissue,
    "IMC_proportion_area_no_tissue": np.sum(IMC_image==0)/np.prod(IMC_image.shape)
}
json.dump(metric_dict, open(IMC_to_postIMC_error_output,"w"))

logging.info("plotting")
fig, ax = plt.subplots(2,3, figsize=(15,10))
ax[0,0].imshow(postIMC_image)
ax[0,0].set_title("postIMC tissue mask")
ax[0,1].imshow(IMC_image>0)
ax[0,1].set_title("IMC cell mask")
ax[0,2].imshow(((IMC_image>0)*127+postIMC_image/2)/4+np.logical_and(IMC_image>0,postIMC_image==0)*192)
ax[0,2].set_title(f"overlay \nProportion of area with no tissue \nin postIMC and cell mask in IMC: {pnpw:.4f}")
ax[1,0].hist(distances, bins=100)
ax[1,0].set_title("distances of grid points before and after registration")
ax[1,0].set_xlabel("distance in microns")
ax[1,1].imshow(IMCimgmov==0)
ax[1,1].set_title("IMC cell mask after registration")
ax[1,2].imshow(((IMCimgmov==0)*127+postIMC_image/2)/4+np.logical_and(IMCimgmov==0,postIMC_image==0)*192)
ax[1,2].set_title(f"overlay after registration\nProportion of area with no tissue \nin postIMC and cell mask in IMC: {pnpwa:.4f}")
plt.savefig(IMC_to_postIMC_error_plot)


logging.info("Finished")