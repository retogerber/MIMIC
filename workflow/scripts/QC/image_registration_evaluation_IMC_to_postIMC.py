import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
import numpy as np
import json
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
    snakemake.params["input_spacing_IMC_location"] = 0.22537
    snakemake.params["output_spacing"] = 1
    snakemake.params["pixel_expansion"] = 501
    snakemake.input["postIMC"] = "results/NASH_HCC_TMA/data/postIMC/NASH_HCC_TMA_postIMC.ome.tiff"
    snakemake.input["IMC_location_on_postIMC"] = "results/NASH_HCC_TMA/data/IMC_location/NASH_HCC_TMA_IMC_mask_on_postIMC_B5.geojson"
    snakemake.input['sam_weights'] = "results/Misc/sam_vit_h_4b8939.pth"

    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
input_spacing_postIMC = snakemake.params["input_spacing_postIMC"]
input_spacing_IMC_location = snakemake.params["input_spacing_IMC_location"]
output_spacing = snakemake.params["output_spacing"]
pixel_expansion = snakemake.params["pixel_expansion"]

# inputs
microscopy_file = snakemake.input['postIMC']
IMC_location=snakemake.input["IMC_location_on_postIMC"]
if isinstance(IMC_location, list):
    IMC_location = IMC_location[0]
CHECKPOINT_PATH = snakemake.input["sam_weights"]

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

logging.info("postIMC image processing")
def gamma_correction(img,gamma):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(img, lookUpTable)

# from: https://stackoverflow.com/questions/65791233/auto-selection-of-gamma-value-for-image-brightness
logging.info(f"Adjust Gamma")
val = cv2.cvtColor(postIMC_image, cv2.COLOR_BGR2HSV)[:,:,2]
gamma = 1/ (np.log(0.5*255)/np.log(np.mean(val)))
logging.info(f"Gamma: {gamma}")
postIMCcutg = gamma_correction(postIMC_image, gamma)

logging.info(f"To HSV, extract saturation")
blur = cv2.cvtColor(postIMCcutg, cv2.COLOR_BGR2HSV)[:,:,1]

logging.info(f"Bilateral filter")
kernel = int(2/input_spacing_postIMC)
blur = cv2.bilateralFilter(blur,kernel,10,10)

# logging.info(f"Threshold with Otsu")
# th,blur = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

logging.info("Optimal thresholding")
tmpshift = int((expand_microns/input_spacing_postIMC)*0.5)
edgepx = np.concatenate([blur[:,:tmpshift].flatten(), blur[:,-tmpshift:].flatten(), blur[:tmpshift,:].flatten(), blur[-tmpshift:,:].flatten()])

tmpshift = int((expand_microns/input_spacing_postIMC)*2)
centerpx = blur[tmpshift:(blur.shape[0]-tmpshift),tmpshift:(blur.shape[1]-tmpshift)].flatten()

edgehist = np.histogram(edgepx, bins=256, range=(0, 256))[0]
centerhist = np.histogram(centerpx, bins=256, range=(0, 256))[0]

minval = 1
minind = 0
for testhr in range(255):
    tmpval = np.sum(edgehist[(testhr+1):])/edgepx.shape + np.sum(centerhist[:testhr])/centerpx.shape
    if tmpval<minval:
        minval = tmpval
        minind = testhr
logging.info(f"Threshold: {minind}")
logging.info(f"proportion of wrong edge pixels: {(np.sum(edgehist[(minind+1):])/edgepx.shape)[0]:.4f}")
logging.info(f"proportion of wrong center pixels: {(np.sum(centerhist[:minind])/centerpx.shape)[0]:.4f}")

logging.info("Thresholding")
blurbin = (blur>minind).astype(np.uint8)*255

logging.info("Median filter")
blurbin = cv2.medianBlur(blurbin, int(1/input_spacing_postIMC)//2*2+1)

logging.info("Create IMC mask")
IMC_image = np.zeros((blurbin.shape), dtype=np.uint8)
tmpshift = int((expand_microns/input_spacing_postIMC))
IMC_image[tmpshift:(IMC_image.shape[0]-tmpshift),tmpshift:(IMC_image.shape[1]-tmpshift)] = 255


logging.info("Calculate Dice score before registration")
union = np.logical_and(IMC_image>0,blurbin>0)
dice_score_before = (2*np.sum(union))/(np.sum(IMC_image>0)+np.sum(blurbin>0))
logging.info(f"Dice score before registration: {dice_score_before}")

def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.8f} "
        + f": {method.GetOptimizerPosition()}"
        + f"; {method.GetOptimizerLearningRate():10.8f} "
    )

def prepare_register(init_transform_inp):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation()
    R.SetMetricSamplingStrategy(R.REGULAR)
    R.SetMetricSamplingPercentage(0.2, seed=1234)    
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsGradientDescentLineSearch(learningRate=1, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10, estimateLearningRate = sitk.ImageRegistrationMethod.Once)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(init_transform_inp)
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    return R

logging.info("register IMC to preIMC")
IMCimg = (IMC_image>0).astype(np.uint8)
# distance to zero
IMCimg = cv2.distanceTransform(IMCimg, cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
IMCimg = IMCimg.astype(np.float32)

# invert
IMCimg = 1/(IMCimg+1)
IMCimg[IMCimg==1]=0
tmpshift = expand_microns+20
IMCmask = np.ones(IMCimg.shape, dtype=np.uint8)*255
IMCmask[tmpshift:(IMCmask.shape[0]-tmpshift),tmpshift:(IMCmask.shape[1]-tmpshift)] = 0

postIMCimg = (blurbin>0).astype(np.uint8)
cv2.medianBlur(postIMCimg, 3, postIMCimg)
postIMCimg = cv2.distanceTransform(postIMCimg, cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE) 
postIMCimg = 1/(postIMCimg+1)
postIMCimg[postIMCimg==1]=0
postIMCimg = postIMCimg.astype(np.float32)

IMCimg = sitk.GetImageFromArray(IMCimg)
IMCimg.SetSpacing([input_spacing_postIMC,input_spacing_postIMC])
postIMCimg = sitk.GetImageFromArray(postIMCimg)
postIMCimg.SetSpacing([input_spacing_postIMC,input_spacing_postIMC])
init_transform_euler1 = sitk.Euler2DTransform()
init_transform_euler1.SetCenter([IMC_image.shape[0]//2,IMC_image.shape[1]//2])
init_transform_euler1.SetParameters([0,0,0])
R = prepare_register(init_transform_euler1)
R.SetMetricMovingMask(sitk.GetImageFromArray(IMCmask))
R.SetOptimizerAsExhaustive([0, 15, 15])
R.SetOptimizerScales([0, 0.2, 0.2])
R.SetMetricSamplingPercentage(0.01, seed=1234)    
logging.info("initial registration")
try:
    init_transform = R.Execute(postIMCimg, IMCimg)
    logging.info(f"Transform parameters: {init_transform.GetParameters()}")
except Exception as e:
    logging.info(f"Registration failed: {e}")
    logging.info("write metrics")
    metric_dict = {
        "IMC_to_postIMC_mean_error": np.nan,
        "IMC_to_postIMC_quantile05_error": np.nan,
        "IMC_to_postIMC_quantile25_error": np.nan,
        "IMC_to_postIMC_quantile50_error": np.nan,
        "IMC_to_postIMC_quantile75_error": np.nan,
        "IMC_to_postIMC_quantile95_error": np.nan,
        "IMC_to_postIMC_n_points": np.nan,
        "IMC_to_postIMC_DICE_score_before_registration": dice_score_before,
        "IMC_to_postIMC_DICE_score_after_registration": np.nan,
    }
    json.dump(metric_dict, open(IMC_to_postIMC_error_output,"w"))
    from pathlib import Path
    Path(IMC_to_postIMC_error_plot).touch()
    sys.exit(0)

logging.info("fine registration")
R = prepare_register(init_transform)
R.SetMetricMovingMask(sitk.GetImageFromArray(IMCmask))
try:
    transform = R.Execute(postIMCimg, IMCimg)
except Exception as e:
    logging.info(f"Registration failed: {e}")
    logging.info("write metrics")
    metric_dict = {
        "IMC_to_postIMC_mean_error": np.nan,
        "IMC_to_postIMC_quantile05_error": np.nan,
        "IMC_to_postIMC_quantile25_error": np.nan,
        "IMC_to_postIMC_quantile50_error": np.nan,
        "IMC_to_postIMC_quantile75_error": np.nan,
        "IMC_to_postIMC_quantile95_error": np.nan,
        "IMC_to_postIMC_n_points": np.nan,
        "IMC_to_postIMC_DICE_score_before_registration": dice_score_before,
        "IMC_to_postIMC_DICE_score_after_registration": np.nan,
    }
    json.dump(metric_dict, open(IMC_to_postIMC_error_output,"w"))
    from pathlib import Path
    Path(IMC_to_postIMC_error_plot).touch()
    sys.exit(0)

logging.info(f"Transform parameters: {transform.GetParameters()}")

def resample_image(transform, fixed, moving, default_value=0.0):
    resampler = sitk.ResampleImageFilter()
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(transform)
    resampler.SetReferenceImage(fixed)
    resampler.SetOutputSpacing(fixed.GetSpacing())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return sitk.GetArrayFromImage(resampler.Execute(moving))


logging.info("resample IMC mask")
IMCimgmov = resample_image(transform, postIMCimg, IMCimg, 0)

logging.info("Calculate Dice score after registration")
union = np.logical_and(IMCimgmov>0,blurbin>0)
dice_score_after = (2*np.sum(union))/(np.sum(IMCimgmov>0)+np.sum(blurbin>0))
logging.info(f"Dice score after registration: {dice_score_after}")

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

grid = create_grid(IMC_image.shape)*input_spacing_postIMC
grid_trans = np.stack([transform.GetInverse().TransformPoint(pt) for pt in grid])

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
    "IMC_to_postIMC_DICE_score_before_registration": dice_score_before,
    "IMC_to_postIMC_DICE_score_after_registration": dice_score_after,
}
json.dump(metric_dict, open(IMC_to_postIMC_error_output,"w"))

logging.info("plotting")
fig, ax = plt.subplots(2,3, figsize=(15,10))
ax[0,0].imshow(blurbin>0)
ax[0,0].set_title("postIMC tissue mask")
ax[0,1].imshow(IMC_image>0)
ax[0,1].set_title("IMC mask")
ax[0,2].imshow(((IMC_image>0)*127+(blurbin>0)*127)/4+np.logical_and(IMC_image>0,blurbin>0)*192)
ax[0,2].set_title(f"Dice score: {dice_score_before:.4f}")
ax[1,0].hist(distances, bins=100)
ax[1,0].set_title("distances of grid points before and after registration")
ax[1,0].set_xlabel("distance in microns")
ax[1,1].imshow(IMCimgmov>0)
ax[1,1].set_title("IMC mask after registration")
ax[1,2].imshow(((IMCimgmov>0)*127+(blurbin>0)*127)/4+np.logical_and(IMCimgmov>0,blurbin>0)*192)
ax[1,2].set_title(f"Dice score: {dice_score_after:.4f}")
plt.savefig(IMC_to_postIMC_error_plot)


logging.info("Finished")
