import sys,os
import wsireg
import SimpleITK as sitk
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
import numpy as np
import json
from image_utils import get_image_shape
from utils import setNThreads, snakeMakeMock
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["input_spacing_postIMC"] = 0.22537
    snakemake.params["input_spacing_IMC"] = 1
    snakemake.params["IMC_rotation_angle"] = 180
    snakemake.input["IMC"] = "results/NASH_HCC_TMA/data/IMC/NASH_HCC_TMA-2_014.tiff"
    snakemake.input["IMC_location_on_postIMC"] = "results/NASH_HCC_TMA/data/IMC_location/NASH_HCC_TMA_registered_IMC_mask_on_postIMC_B9.geojson"
    snakemake.input["postIMC"] = "results/NASH_HCC_TMA/data/postIMC/NASH_HCC_TMA_postIMC.ome.tiff"
    snakemake.output["IMC_to_postIMC_transform"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/registrations/IMC_to_postIMC/test_split_pre_B1/test_split_pre_B1-IMC_to_postIMC_transformations.json"
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")

# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
input_spacing_postIMC = snakemake.params["input_spacing_postIMC"]
input_spacing_IMC = snakemake.params["input_spacing_IMC"]
IMC_rotation_angle = snakemake.params["IMC_rotation_angle"]

assert IMC_rotation_angle in [0,90,180,270], "IMC rotation angle must be 0, 90, 180 or 270"

# inputs
imc_file = snakemake.input["IMC"]
IMC_location = snakemake.input["IMC_location_on_postIMC"]
postIMC_file = snakemake.input["postIMC"]

# outputs
IMC_to_postIMC_transform = snakemake.output["IMC_to_postIMC_transform"]

imc_shape = get_image_shape(imc_file)

if isinstance(IMC_location, list):
    IMC_location = IMC_location[0]

logging.info("core bounding box extraction")
IMC_geojson = json.load(open(IMC_location, "r"))
if isinstance(IMC_geojson,list):
    IMC_geojson=IMC_geojson[0]
boundary_points = np.array(IMC_geojson['geometry']['coordinates'])[0,:,:]
boundary_points = np.unique(boundary_points, axis=0)

assert boundary_points.shape[0] == 4

xmin=np.min(boundary_points[:,1])
xmax=np.max(boundary_points[:,1])
ymin=np.min(boundary_points[:,0])
ymax=np.max(boundary_points[:,0])


is_left = boundary_points[:,1]<(xmin+xmax)/2
is_right = boundary_points[:,1]>=(xmin+xmax)/2
is_top = boundary_points[:,0]<(ymin+ymax)/2
is_bottom = boundary_points[:,0]>=(ymin+ymax)/2

imc_corners = np.array([
    [0.5,0.5],
    [imc_shape[2]-0.5,0.5],
    [imc_shape[2]-0.5,imc_shape[1]-0.5],
    [0.5,imc_shape[1]-0.5]
])

is_left_imc = imc_corners[:,1]<(imc_corners[:,1].max()+imc_corners[:,1].min())/2
is_right_imc = imc_corners[:,1]>=(imc_corners[:,1].max()+imc_corners[:,1].min())/2
is_top_imc = imc_corners[:,0]<(imc_corners[:,0].max()+imc_corners[:,0].min())/2
is_bottom_imc = imc_corners[:,0]>=(imc_corners[:,0].max()+imc_corners[:,0].min())/2

imc_location_pts_ind = [
    np.arange(4)[np.logical_and(is_left, is_top)][0],
    np.arange(4)[np.logical_and(is_right, is_top)][0],
    np.arange(4)[np.logical_and(is_right, is_bottom)][0],
    np.arange(4)[np.logical_and(is_left, is_bottom)][0]
]

imc_pts_ind = [
    np.arange(4)[np.logical_and(is_left_imc, is_top_imc)][0],
    np.arange(4)[np.logical_and(is_right_imc, is_top_imc)][0],
    np.arange(4)[np.logical_and(is_right_imc, is_bottom_imc)][0],
    np.arange(4)[np.logical_and(is_left_imc, is_bottom_imc)][0]
]

bpts = np.stack([boundary_points[imc_location_pts_ind[i],:] for i in range(4)])*input_spacing_postIMC
ipts = np.stack([imc_corners[imc_pts_ind[i],:] for i in range(4)])*input_spacing_IMC

rot_transform = sitk.Euler2DTransform()
rot_transform.SetCenter([imc_shape[2]/2,imc_shape[1]/2])
rot_transform.SetAngle(IMC_rotation_angle/180*np.pi)

logging.info(f"IMC points before rotation: \n{ipts}")
ipts = np.stack([rot_transform.GetInverse().TransformPoint(pt) for pt in ipts.astype(float)])
logging.info(f"IMC points after rotation: \n{ipts}")
logging.info(f"PostIMC points: \n{bpts}")
assert np.all(ipts>=0)

# bptsf = [float(c) for p in bpts[:, [1, 0]] for c in p]
# iptsf = [float(c) for p in ipts[:, [1, 0]] for c in p]
bptsf = [float(c) for p in bpts[:, [0, 1]] for c in p]
iptsf = [float(c) for p in ipts[:, [0, 1]] for c in p]

logging.info("create resampler")
postIMC_shape = get_image_shape(postIMC_file)
postimc = np.zeros(np.array(postIMC_shape[:2])[::-1], dtype=np.uint8)
postimc = sitk.GetImageFromArray(postimc)
postimc.SetSpacing([input_spacing_postIMC,input_spacing_postIMC])


landmark_initializer = sitk.LandmarkBasedTransformInitializerFilter()
landmark_initializer.SetFixedLandmarks(bptsf)
landmark_initializer.SetMovingLandmarks(iptsf)
landmark_initializer.SetReferenceImage(postimc)
transform = sitk.AffineTransform(2)
# transform = sitk.Euler2DTransform()
transform = landmark_initializer.Execute(transform)
logging.info(f"Transform parameters: {transform.GetParameters()}")

iptsf_trans = [np.array(transform.GetInverse().TransformPoint(pt[[1,0]]))[[1,0]] for pt in ipts.astype(float)]
iptsf_trans =np.stack(iptsf_trans)
logging.info(f"PostIMC points: \n{bpts}")
logging.info(f"Transformed IMC points: \n{iptsf_trans}")

# linalp
distances = np.linalg.norm(bpts-iptsf_trans, axis=1)
logging.info(f"Distances: {distances}")

# from skimage.transform import warp, AffineTransform
# model = AffineTransform()
# model.estimate(bpts, ipts)
# model.params

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

resampler = get_resampler(transform, postimc)

logging.info("ITK transformation to parameter map")
tform=wsireg.parameter_maps.transformations.BASE_AFF_TFORM.copy()
# tform=wsireg.parameter_maps.transformations.BASE_RIG_TFORM.copy()
tform["CenterOfRotationPoint"] = [str(p) for p in transform.GetCenter()]
tform["TransformParameters"] = [str(p) for p in transform.GetParameters()]

tform["Spacing"] = [str(p) for p in resampler.GetOutputSpacing()]
tform["Direction"] = [str(p) for p in resampler.GetOutputDirection()]
tform["Origin"] = [str(p) for p in resampler.GetOutputOrigin()]
tform["Size"] = [str(p) for p in resampler.GetSize()]
tform["ResampleInterpolator"] = [k for k,v in wsireg.utils.tform_utils.ELX_TO_ITK_INTERPOLATORS.items() if v == resampler.GetInterpolator()]
logging.info(f"Transform: \n{tform}")

logging.info("check conversion")
test_transform = wsireg.utils.tform_conversion.convert_to_itk(tform)
logging.info(f"parameter: {test_transform.GetParameters() == transform.GetParameters()}")
assert test_transform.GetParameters() == transform.GetParameters()
logging.info(f"center: {test_transform.GetCenter() == transform.GetCenter()}")
assert test_transform.GetCenter() == transform.GetCenter()
logging.info(f"size: {test_transform.OutputSize == np.array(postIMC_shape[:2])}")
assert np.all(test_transform.OutputSize == np.array(postIMC_shape[:2]))

logging.info("save transformation")
if not os.path.exists(os.path.dirname(IMC_to_postIMC_transform)):
    os.makedirs(os.path.dirname(IMC_to_postIMC_transform))
json.dump(tform, open(IMC_to_postIMC_transform,"w"))

logging.info("Finished")
