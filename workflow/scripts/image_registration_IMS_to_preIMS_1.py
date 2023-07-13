import pandas as pd
import SimpleITK as sitk
import napari_imsmicrolink
import skimage
import numpy as np
from wsireg.utils.im_utils import grayscale
# from imc_to_ims_workflow.workflow.scripts.image_registration_IMS_to_preIMS_utils import *
from image_registration_IMS_to_preIMS_utils import *
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

# parameters
# stepsize = 30
stepsize = float(snakemake.params["IMS_pixelsize"])
# pixelsize = 24
pixelsize = stepsize*float(snakemake.params["IMS_shrink_factor"])
# resolution = 1
resolution = float(snakemake.params["IMC_pixelsize"])
rotation_imz = 180
rotation_imz = float(snakemake.params["IMS_rotation_angle"])
assert(rotation_imz in [-270,-180,-90,0,90,180,270])

# postIMSr_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/postIMS/test_split_ims_postIMS_reduced_mask.ome.tiff"
postIMSr_file = snakemake.input["postIMSmask_downscaled"]
# imzmlfile = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMS/IMS_test_split_ims_2.imzML"
imzmlfile = snakemake.input["imzml"]
# imc_mask_files = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_002_transformed.ome.tiff"
imc_mask_files = snakemake.input["IMCmask"]
if isinstance(imc_mask_files, str):
    imc_mask_files = [imc_mask_files]
# sample_metadata = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/config/sample_metadata.csv"
sample_metadata = snakemake.input["sample_metadata"]

# output_table = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMS/IMS_to_postIMS_matches.csv"
output_table = snakemake.output["IMS_to_postIMS_matches"]



logging.info("IMC location")
# get imc location info
imcbboxls = list()
for imcmaskfile in imc_mask_files:
    imc=skimage.io.imread(imcmaskfile)
    imc[imc>0]=255
    imc = imc.astype(np.uint8)
    imcbboxls.append(skimage.measure.regionprops(imc)[0].bbox)

imc_samplenames = [ os.path.splitext(os.path.splitext(os.path.split(f)[1])[0])[0].replace("_transformed","") for f in imc_mask_files]
imc_projects = [ os.path.split(os.path.split(os.path.split(os.path.split(f)[0])[0])[0])[1] for f in imc_mask_files]


with open(sample_metadata, 'r') as fil:
    sample_metadata_df = pd.read_csv(fil)

core_names = list()
for i in range(len(imc_samplenames)):
    inds_arr = np.logical_and(sample_metadata_df["project_name"] == imc_projects[i], sample_metadata_df["sample_name"] == imc_samplenames[i])
    df_sub = sample_metadata_df.loc[inds_arr]
    core_names.append(df_sub["core_name"].tolist()[0])


logging.info("Read postIMS mask")
postIMSr = skimage.io.imread(postIMSr_file)

# expand mask
outermask = skimage.morphology.isotropic_dilation(postIMSr, (1/resolution)*stepsize*2)
# measure regions
postIMSregin = skimage.measure.label(outermask.astype(np.uint8))

logging.info("Find IMC to postIMS overlap")
postIMSregions = list()
for bb in imcbboxls:
    tmpuqs = np.unique([postIMSregin[bb[0],bb[1]], postIMSregin[bb[0],bb[3]], postIMSregin[bb[2],bb[1]], postIMSregin[bb[2],bb[3]]])
    tmpuqs = tmpuqs[tmpuqs>0]
    assert(len(tmpuqs)==1)
    postIMSregions.append(tmpuqs[0])

regpops = skimage.measure.regionprops(postIMSregin)
postIMS_pre_bbox = [r.bbox for r in regpops]
postIMS_pre_labels = [r.label for r in regpops]

logging.info("Find global bounding box of cores")
tmpbool = np.array([p in np.array(postIMSregions) for p in postIMS_pre_labels])
postIMS_pre_bbox = np.array(postIMS_pre_bbox)[tmpbool]
postIMS_pre_labels = np.array(postIMS_pre_labels)[tmpbool]
global_bbox = [
    np.min([p[0] for p in postIMS_pre_bbox]),
    np.min([p[1] for p in postIMS_pre_bbox]),
    np.max([p[2] for p in postIMS_pre_bbox]),
    np.max([p[3] for p in postIMS_pre_bbox]),
]


logging.info("Read imzML file")
# read imzml file
imz = napari_imsmicrolink.data.ims_pixel_map.PixelMapIMS(imzmlfile)
# stepsize (not actually used)
imz.ims_res = stepsize
# create image mask
imzimg = imz._make_pixel_map_at_ims(randomize=False, map_type="minimized")
# rescale to postIMS resolution
imzimgres = skimage.transform.rescale(imzimg, stepsize*resolution, preserve_range = True)   
imzimgres[imzimgres>0] = 255 

logging.info("Apply rotation")
# rotate 180 degrees
imzimg = skimage.transform.rotate(imzimg,rotation_imz, preserve_range=True)
imzimgres = skimage.transform.rotate(imzimgres,rotation_imz, preserve_range=True)


logging.info("Cut postIMS mask to global bounding box")
postIMSrcut = postIMSr[global_bbox[0]:global_bbox[2],global_bbox[1]:global_bbox[3]]
postIMSregincut = postIMSregin[global_bbox[0]:global_bbox[2],global_bbox[1]:global_bbox[3]]

logging.info("Remove cores in postIMS that are missing in imz")
boolimgls = list()
for region in postIMSregions:
    boolimgls.append(postIMSregincut == region)
boolimg = np.sum(np.stack(boolimgls),axis=0).astype(bool)
postIMSrcut[np.logical_not(boolimg)] = 0

logging.info("Register postIMS to imz")
# function used in registration
def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.5f} "
        + f": {method.GetOptimizerPosition()}"
    )
# setup sitk images
fixed_np = imzimgres.astype(np.float32)
moving_np = postIMSrcut.astype(np.float32)*255
fixed = sitk.GetImageFromArray(fixed_np)
moving = sitk.GetImageFromArray(moving_np)

# initial transformation
init_transform = sitk.CenteredTransformInitializer(
        fixed, 
        moving, 
        sitk.AffineTransform(2), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

# setup registration
R = sitk.ImageRegistrationMethod()
R.SetMetricAsCorrelation()
R.SetMetricSamplingStrategy(R.RANDOM)
R.SetMetricSamplingPercentage(0.01)
R.SetInterpolator(sitk.sitkLinear)
R.SetOptimizerAsGradientDescent(
    learningRate=1.0, numberOfIterations=1000
)
R.SetOptimizerScalesFromPhysicalShift()
R.SetInitialTransform(init_transform)
R.SetInterpolator(sitk.sitkLinear)

R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

# run registration
transform = R.Execute(fixed, moving)

# transform expanded, labeled mask
resampler = sitk.ResampleImageFilter()
resampler.SetTransform(transform)
resampler.SetReferenceImage(fixed)
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
tmp1 = resampler.Execute(sitk.GetImageFromArray(postIMSregincut))
postIMSro_trans = sitk.GetArrayFromImage(tmp1)


# create imz region image
y_extent, x_extent, y_coords, x_coords = imz._get_xy_extents_coords(map_type="minimized")
imzregions = np.zeros((y_extent, x_extent), dtype=np.uint8)
imzregions[y_coords, x_coords] = imz.regions
imzregions = skimage.transform.rotate(imzregions,rotation_imz, preserve_range=True)
imzregions = np.round(imzregions)
imzuqregs = np.unique(imz.regions)


logging.info("Find matching regions between postIMS and imz")
observedpostIMSregion_match = list()
# for imz region 1 get matching postIMS core
for regionimz in imzuqregs:
    t1=imzregions==regionimz
    t1 = skimage.transform.rescale(t1, stepsize*resolution, preserve_range = True)   
    overlaps = np.unique(postIMSro_trans.astype(np.uint8)[t1], return_counts=True)
    overlaps = (overlaps[0][overlaps[0]!=0],overlaps[1][overlaps[0]!=0])
    observedpostIMSregion_match.append(overlaps[0][overlaps[1] == np.max(overlaps[1])][0])
observedpostIMSregion_match = np.array(observedpostIMSregion_match)


logging.info("Create bounding boxes for all cores")
postIMS_bboxls = list()
for i in range(len(observedpostIMSregion_match)):
    # select subset region for postIMS
    postIMSregpops = skimage.measure.regionprops(postIMSregin)
    regpopslabs = np.asarray([t.label for t in postIMSregpops])
    tmpind = np.asarray(list(range(len(regpopslabs))))[regpopslabs==observedpostIMSregion_match[i]][0]
    postIMS_bboxls.append(postIMSregpops[tmpind].bbox)

postIMSxmins=[b[0] for b in postIMS_bboxls]
postIMSymins=[b[1] for b in postIMS_bboxls]
postIMSxmaxs=[b[2] for b in postIMS_bboxls]
postIMSymaxs=[b[3] for b in postIMS_bboxls]

logging.info("Save data")
df1 = pd.DataFrame({
    "imzregion": imzuqregs,
    "postIMSregion": observedpostIMSregion_match,
    "postIMS_xmin": postIMSxmins,
    "postIMS_ymin": postIMSymins,
    "postIMS_xmax": postIMSxmaxs,
    "postIMS_ymax": postIMSymaxs
}).set_index("postIMSregion")

df2 = pd.DataFrame({
    "postIMSregion": postIMSregions,
    "core_name": core_names,
    "project_name": imc_projects,
    "sample_name": imc_samplenames
}).set_index("postIMSregion")

dfout = df2.join(df1, on=["postIMSregion"])
dfout.to_csv(output_table)

logging.info("Finished")