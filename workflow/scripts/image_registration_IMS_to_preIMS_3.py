import cv2
import napari
import pycpd 
import SimpleITK as sitk
import napari_imsmicrolink
from napari_imsmicrolink.utils.json import NpEncoder
import skimage
import numpy as np
import json
import matplotlib.pyplot as plt
from image_registration_IMS_to_preIMS_utils import readimage_crop,  create_ring_mask, composite2affine, saveimage_tile,  create_imz_coords,get_rotmat_from_angle
from sklearn.neighbors import KDTree
import shapely
import shapely.affinity
from pathlib import Path
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

threads = int(snakemake.threads)
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)

# stepsize = 30
# stepsize = 10
stepsize = float(snakemake.params["IMS_pixelsize"])
# pixelsize = 24
# pixelsize = 8 
pixelsize = stepsize*float(snakemake.params["IMS_shrink_factor"])
# resolution = 1
# resolution = 0.22537
resolution = float(snakemake.params["IMC_pixelsize"])
# rotation_imz = 180
# rotation_imz = 0
rotation_imz = float(snakemake.params["IMS_rotation_angle"])
assert(rotation_imz in [-270,-180,-90,0,90,180,270])
rotmat = get_rotmat_from_angle(rotation_imz)

logging.info("Rotation angle: "+str(rotation_imz))
logging.info("IMS stepsize: "+str(stepsize))
logging.info("IMS pixelsize: "+str(pixelsize))
logging.info("Microscopy pixelsize: "+str(resolution))

# imzmlfile = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMS/IMS_test_split_pre.imzML"
# imzmlfile = "/home/retger/Downloads/pos_mode_lipids_tma_02022023_imzml.imzML"
imzmlfile = snakemake.input["imzml"]

# imc_mask_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_002_transformed.ome.tiff"
# imc_mask_file = "/home/retger/Downloads/Lipid_TMA_37819_025_transformed.ome.tiff"
imc_mask_file = snakemake.input["IMCmask"]

imc_samplename = os.path.splitext(os.path.splitext(os.path.split(imc_mask_file)[1])[0])[0].replace("_transformed","")
# imc_project = "Lipid_TMA"
imc_project = os.path.split(os.path.split(os.path.split(os.path.split(imc_mask_file)[0])[0])[0])[1]
project_name = "postIMS_to_IMS_"+imc_project+"-"+imc_samplename

# postIMS_file = "/home/retger/Downloads/Lipid_TMA_3781_postIMS.ome.tiff"
postIMS_file = snakemake.input["postIMS_downscaled"]

# masks_transform_filename = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_002_masks_transform.txt"
# masks_transform_filename = "/home/retger/Downloads/test_images_ims_to_imc_workflow/Lipid_TMA_37819_025_masks_transform.txt"
masks_transform_filename = snakemake.input["masks_transform"]
# gridsearch_transform_filename = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_002_gridsearch_transform.txt"
# gridsearch_transform_filename = "/home/retger/Downloads/test_images_ims_to_imc_workflow/Lipid_TMA_37819_025_gridsearch_transform.txt"
gridsearch_transform_filename = snakemake.input["gridsearch_transform"]

# postIMS_ablation_centroids_filename = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_002_postIMS_ablation_centroids.csv"
# postIMS_ablation_centroids_filename = "/home/retger/Downloads/test_images_ims_to_imc_workflow/Lipid_TMA_37819_025_postIMS_ablation_centroids.csv"
postIMS_ablation_centroids_filename = snakemake.input["postIMS_ablation_centroids"]
# metadata_to_save_filename = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_002_step1_metadata.json"
# metadata_to_save_filename = "/home/retger/Downloads/test_images_ims_to_imc_workflow/Lipid_TMA_37819_025_step1_metadata.json"
metadata_to_save_filename = snakemake.input["metadata"]


# ims_to_postIMS_regerror = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_001_IMS_to_postIMS_reg_auto_metrics.json"
ims_to_postIMS_regerror = snakemake.output["IMS_to_postIMS_error"]
ims_to_postIMS_regerror_image = snakemake.output["IMS_to_postIMS_error_image"]

# coordsfile_out = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMS/postIMS_to_IMS_test_split_pre-IMSML-coords.h5"
coordsfile_out = snakemake.output["imsml_coords_fp"]
output_dir = os.path.dirname(coordsfile_out)


logging.info("Read data")

with open(metadata_to_save_filename, 'r') as fp:
    metadata = json.load(fp)
imz_bbox = metadata['IMS_bbox']
postIMS_bbox = metadata['postIMS_bbox']
xmin, ymin, xmax, ymax = postIMS_bbox
imzuqregs = np.array(metadata['IMS_regions'])
regionimz = metadata['Matching_IMS_region']
postIMS_shape = (int(postIMS_bbox[2]/resolution)-int(postIMS_bbox[0]/resolution), int(postIMS_bbox[3]/resolution)-int(postIMS_bbox[1]/resolution))

tinv = sitk.ReadTransform(masks_transform_filename )
tmp_transform = sitk.ReadTransform(gridsearch_transform_filename )

imspixel_inscale = 4
imspixel_outscale = 2
imz = napari_imsmicrolink.data.ims_pixel_map.PixelMapIMS(imzmlfile)
imz.ims_res = stepsize
imzimg = imz._make_pixel_map_at_ims(randomize=False, map_type="minimized")
imzimg = skimage.transform.rotate(imzimg,rotation_imz, preserve_range=True)
xminimz, yminimz, xmaximz, ymaximz = imz_bbox
imzrefcoords = np.stack([imz.y_coords_min,imz.x_coords_min],axis=1)

centsred = np.loadtxt(postIMS_ablation_centroids_filename, delimiter=',')

def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.8f} "
        + f": {method.GetOptimizerPosition()}"
    )


def get_sigma(threshold:float = (stepsize-pixelsize)/resolution, 
              p:float = 0.99):
    return 1/(5.5556*(1-((1-p)/p)**0.1186)/threshold)


########### sitk registration of points
### Get points
logging.info("Prepare first registration")
imzringmask = create_ring_mask(imzimg[xminimz:xmaximz,yminimz:ymaximz], imspixel_outscale, imspixel_inscale+1)
imzcoords = create_imz_coords(imzimg, imzringmask, imzrefcoords, imz_bbox, rotmat)
init_translation = -np.array([xminimz,yminimz]).astype(int)
imzcoords_in = imzcoords + init_translation
imzcoordsfilttrans = np.array([tinv.TransformPoint(imzcoords_in[i,:].astype(float)) for i in range(imzcoords_in.shape[0])])
tmpimzrot = np.array([tmp_transform.TransformPoint(imzcoordsfilttrans[i,:]) for i in range(imzcoordsfilttrans.shape[0])])
kdt = KDTree(tmpimzrot, leaf_size=30, metric='euclidean')
centsred_distances, indices = kdt.query(centsred, k=1, return_distance=True)
centsred_has_match = centsred_distances.flatten()<1.5
centsredfilt = centsred[centsred_has_match,:]
imzringmask = create_ring_mask(imzimg[xminimz:xmaximz,yminimz:ymaximz], imspixel_outscale, imspixel_inscale+3)
imzcoords = create_imz_coords(imzimg, imzringmask, imzrefcoords, imz_bbox, rotmat)
init_translation = -np.array([xminimz,yminimz]).astype(int)
imzcoords_in = imzcoords + init_translation
imzcoordsfilttrans = np.array([tinv.TransformPoint(imzcoords_in[i,:].astype(float)) for i in range(imzcoords_in.shape[0])])
tmpimzrot = np.array([tmp_transform.TransformPoint(imzcoordsfilttrans[i,:]) for i in range(imzcoordsfilttrans.shape[0])])
kdt = KDTree(centsredfilt, leaf_size=30, metric='euclidean')
imz_distances, indices = kdt.query(tmpimzrot, k=1, return_distance=True)
imz_has_match = imz_distances.flatten()<3
imzcoordsfilt = imzcoordsfilttrans[imz_has_match,:]

def image_from_points(shape, points, sigma=0, half_pixel_size = 1):
    img = np.zeros(shape, dtype=bool)
    for i in range(points.shape[0]):
        xr = int(points[i,0])
        if xr<0:
            xr=0
        if (xr)>img.shape[0]:
            xr=img.shape[0]-1
        yr = int(points[i,1])
        if yr<0:
            yr=0
        if (yr)>img.shape[1]:
            yr=img.shape[1]-1
        img[xr,yr] = True
    img = cv2.morphologyEx(src=img.astype(np.uint8), op = cv2.MORPH_DILATE, kernel = skimage.morphology.disk(half_pixel_size)).astype(bool)
    img = cv2.GaussianBlur(img.astype(np.uint8)*255,ksize=[0,0],sigmaX=sigma)
    return img/np.max(img)*255


postIMSpimg1 = image_from_points(postIMS_shape, centsredfilt/resolution*stepsize, get_sigma((stepsize-pixelsize)/resolution/2, 0.99), int(stepsize/3/resolution))

IMSpimg1 = image_from_points(postIMS_shape, imzcoordsfilt/resolution*stepsize, get_sigma((stepsize-pixelsize)/resolution/2,0.99), int(stepsize/3/resolution))

# plt.imshow(IMSpimg1.astype(float)-postIMSpimg1.astype(float))
# plt.show()

## First registration, Euler2D
## only boundary points
logging.info("Run first registration")
fixed = sitk.GetImageFromArray(IMSpimg1.astype(float))
fixedmask = create_ring_mask(imzimg[xminimz:xmaximz,yminimz:ymaximz], imspixel_outscale+5, imspixel_inscale+8)
fixedmask = skimage.transform.resize(fixedmask,postIMS_shape)
fixedmask = sitk.GetImageFromArray(fixedmask.astype(np.uint8)*200)
fixedmask = sitk.BinaryThreshold(fixedmask,lowerThreshold=127, upperThreshold=255)
moving = sitk.GetImageFromArray(postIMSpimg1.astype(float))
movingmask = cv2.morphologyEx(src=(postIMSpimg1>0).astype(np.uint8), op = cv2.MORPH_DILATE, kernel = skimage.morphology.square(int(stepsize/resolution)*8)).astype(bool)
movingmask = sitk.GetImageFromArray(movingmask.astype(np.uint8)*200)
movingmask = sitk.BinaryThreshold(movingmask,lowerThreshold=127, upperThreshold=255)
# plt.imshow(movingmask*255-postIMSpimg1)
# plt.show()


init_transform = sitk.Euler2DTransform()
init_transform.SetTranslation(np.array(tmp_transform.GetTranslation())/resolution*stepsize)
init_transform.SetAngle(tmp_transform.GetAngle())
# postIMSro_trans = resample_image(init_transform, fixed, postIMSpimg1)
# plt.imshow(IMSpimg1.astype(float)-postIMSro_trans.astype(float))
# plt.show()


R = sitk.ImageRegistrationMethod()
R.SetMetricAsMeanSquares()
R.SetMetricSamplingStrategy(R.REGULAR)
R.SetMetricSamplingPercentage(0.05)
R.SetMetricFixedMask(fixedmask)
R.SetMetricMovingMask(movingmask)
R.SetInterpolator(sitk.sitkNearestNeighbor)
# es_stepsize = np.ceil(1/resolution)
es_stepsize = 1/resolution*(stepsize/2)
# R.SetOptimizerAsExhaustive([7, round(stepsize/resolution*1.25/es_stepsize), round(stepsize/resolution*1.25/es_stepsize)])
n_halfsteps = [3, round(stepsize/resolution*1.5/es_stepsize), round(stepsize/resolution*1.5/es_stepsize)]
R.SetOptimizerAsExhaustive(n_halfsteps)
R.SetOptimizerScales([np.pi / 720, es_stepsize, es_stepsize])
R.SetInitialTransform(init_transform)
def progress_bar_simple(method, maxiter=1000):
    if int(method.GetOptimizerIteration())%int(maxiter/80) == 0:
        print(f"{method.GetOptimizerIteration():5} / {maxiter:5}")
R.AddCommand(sitk.sitkIterationEvent, lambda: progress_bar_simple(R, maxiter=np.prod(np.array(n_halfsteps)*2+1)))
# R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
transform = R.Execute(fixed, moving)
transform.GetParameters()
logging.info(f"Gridsearch parameters: {transform.GetParameters()}")


def resample_image(transform, fixed, moving_np):
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return sitk.GetArrayFromImage(resampler.Execute(sitk.GetImageFromArray(moving_np)))

postIMSro_trans = resample_image(init_transform, fixed, postIMSpimg1)
# plt.imshow(IMSpimg1.astype(float)-postIMSro_trans.astype(float))
# plt.show()

tmpfilename = f"{os.path.dirname(snakemake.log['stdout'])}/{os.path.basename(snakemake.log['stdout']).split('.')[0]}_gridsearch_registration.ome.tiff"
logging.info(f"Save Image difference as: {tmpfilename}")
saveimage_tile(postIMSro_trans.astype(float)-IMSpimg1.astype(float), tmpfilename, resolution)


## Prepare second registration
logging.info("Prepare second registration")
init_transform_scaled = sitk.Euler2DTransform()
init_transform_scaled.SetTranslation(np.array(init_transform.GetTranslation())*resolution/stepsize)
init_transform_scaled.SetMatrix(init_transform.GetMatrix())
tmpimzrot2 = np.array([init_transform_scaled.TransformPoint(imzcoordsfilttrans[i,:]) for i in range(imzcoordsfilttrans.shape[0])])
# plt.scatter(tmpimzrot2[:,1],tmpimzrot2[:,0])
# plt.scatter(centsredfilt[:,1],centsredfilt[:,0])
# plt.show()


kdt = KDTree(centsredfilt, leaf_size=30, metric='euclidean')
imz_distances, indices = kdt.query(tmpimzrot2, k=1, return_distance=True)
imz_has_match = imz_distances.flatten()<1.5
imzcoordsfilt = imzcoordsfilttrans[imz_has_match,:]

postIMSpimg2 = image_from_points(postIMS_shape, centsredfilt/resolution*stepsize, get_sigma((stepsize-pixelsize)/resolution*2), int(1/resolution))
IMSpimg2 = image_from_points(postIMS_shape, imzcoordsfilt/resolution*stepsize, get_sigma((stepsize-pixelsize)/resolution*2), int(1/resolution))
postIMSro_trans = resample_image(transform, fixed, postIMSpimg2)

# plt.imshow(IMSpimg2.astype(float)-postIMSro_trans.astype(float))
# plt.show()



## Second registration, Affine
## only boundary
logging.info("Run second registration")
fixed = sitk.GetImageFromArray(IMSpimg2.astype(float))
moving = sitk.GetImageFromArray(postIMSpimg2.astype(float))

init_transform2 = sitk.AffineTransform(2)
init_transform2.SetMatrix(transform.GetMatrix())
init_transform2.SetTranslation(transform.GetTranslation())
init_transform2.SetCenter((np.max(centsred,axis=0)-np.min(centsred,axis=0))/2/resolution*stepsize+np.min(centsred,axis=0)/resolution*stepsize)
R = sitk.ImageRegistrationMethod()
R.SetMetricAsMeanSquares()
# R.SetMetricAsCorrelation()
R.SetMetricSamplingStrategy(R.REGULAR)
R.SetMetricSamplingPercentage(0.25)
R.SetInterpolator(sitk.sitkNearestNeighbor)
R.SetOptimizerAsGradientDescent(
    learningRate=1, numberOfIterations=1000, 
    convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    estimateLearningRate=R.EachIteration
)
R.SetOptimizerScalesFromIndexShift()
R.SetInitialTransform(init_transform2)
R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

transform2 = R.Execute(fixed, moving)
logging.info(f"Full registration parameters: {transform2.GetParameters()}")

postIMSro_trans = resample_image(transform2, fixed, postIMSpimg2)
# plt.imshow(IMSpimg2.astype(float)-postIMSro_trans.astype(float))
# plt.show()

tmpfilename = f"{os.path.dirname(snakemake.log['stdout'])}/{os.path.basename(snakemake.log['stdout']).split('.')[0]}_full_registration.ome.tiff"
logging.info(f"Save Image difference as: {tmpfilename}")
saveimage_tile(postIMSro_trans.astype(float)-IMSpimg2.astype(float), tmpfilename, resolution)

## Third registration, Affine
## only points in IMC region
logging.info("Prepare third registration")
imcmask = readimage_crop(imc_mask_file, [int(xmin), int(ymin), int(xmax), int(ymax)])
imcmaskch = skimage.morphology.convex_hull_image(imcmask>0)
imcmaskch = skimage.transform.resize(imcmaskch,postIMS_shape)
imcmaskch = cv2.morphologyEx(src=imcmaskch.astype(np.uint8), op = cv2.MORPH_DILATE, kernel = skimage.morphology.square(2*int((1/resolution)*stepsize*2))).astype(bool)
imcmaskch = resample_image(transform2, fixed, imcmaskch.astype(np.uint8)).astype(bool)

IMSoutermask = skimage.morphology.isotropic_dilation(imzimg[xminimz:xmaximz,yminimz:ymaximz], (1/resolution)*stepsize*imspixel_outscale)
imzcoords = create_imz_coords(imzimg, IMSoutermask, imzrefcoords, imz_bbox, rotmat)
init_translation = -np.array([xminimz,yminimz]).astype(int)
imzcoords_in = imzcoords + init_translation
imzcoordsfilttrans = np.array([tinv.TransformPoint(imzcoords_in[i,:].astype(float)) for i in range(imzcoords_in.shape[0])])
# tmpimzrot = np.array([tmp_transform.TransformPoint(imzcoordsfilttrans[i,:]) for i in range(imzcoordsfilttrans.shape[0])])

# init_transform_scaled = sitk.Euler2DTransform()
# init_transform_scaled.SetTranslation(np.array(init_transform.GetTranslation())*resolution/stepsize)
# init_transform_scaled.SetMatrix(init_transform.GetMatrix())
# tmpimzrot2 = np.array([init_transform_scaled.GetInverse().TransformPoint(tmpimzrot[i,:]) for i in range(tmpimzrot.shape[0])])

# init_transform_scaled2 = sitk.AffineTransform(2)
# init_transform_scaled2.SetTranslation(np.array(init_transform2.GetTranslation())*resolution/stepsize)
# init_transform_scaled2.SetMatrix(init_transform2.GetMatrix())
# init_transform_scaled2.SetCenter(np.array(init_transform2.GetCenter())*resolution/stepsize)
# tmpimzrot3 = np.array([init_transform_scaled2.TransformPoint(tmpimzrot2[i,:]) for i in range(tmpimzrot2.shape[0])])


postIMSpimg3 = image_from_points(postIMS_shape, centsred/resolution*stepsize,get_sigma((stepsize-pixelsize)/resolution), int(1/resolution))
postIMSpimg3compl = postIMSpimg3.copy()
postIMSpimg3[~imcmaskch] = postIMSpimg3[~imcmaskch]/2

# IMSpimg3 = image_from_points(postIMSmpre.shape, tmpimzrot3/resolution*stepsize, 6, int(1/resolution))
IMSpimg3 = image_from_points(postIMS_shape, imzcoordsfilttrans/resolution*stepsize, get_sigma((stepsize-pixelsize)/resolution), int(1/resolution))
IMSpimg3compl = IMSpimg3.copy()
IMSpimg3[~imcmaskch] = IMSpimg3[~imcmaskch]/2

# postIMSro_trans = resample_image(transform2, fixed, postIMSpimg3compl)
# plt.imshow(IMSpimg3compl.astype(float)-postIMSro_trans.astype(float))
# plt.show()

# postIMSro_trans = resample_image(transform2, fixed, postIMSpimg3)
# plt.imshow(IMSpimg3.astype(float)-postIMSro_trans.astype(float))
# plt.show()


logging.info("Run third registration")
fixed = sitk.GetImageFromArray(IMSpimg3.astype(float))
moving = sitk.GetImageFromArray(postIMSpimg3.astype(float))

init_transform3 = sitk.AffineTransform(2)
init_transform3.SetParameters(transform2.GetParameters())
init_transform3.SetCenter((np.max(centsred,axis=0)-np.min(centsred,axis=0))/2/resolution*stepsize+np.min(centsred,axis=0)/resolution*stepsize)
R = sitk.ImageRegistrationMethod()
R.SetMetricAsMeanSquares()
R.SetMetricSamplingStrategy(R.REGULAR)
R.SetMetricSamplingPercentage(0.25)
R.SetInterpolator(sitk.sitkNearestNeighbor)
R.SetOptimizerAsGradientDescent(
    learningRate=1, numberOfIterations=1000, 
    convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    estimateLearningRate=R.EachIteration
)
R.SetOptimizerScalesFromIndexShift()
R.SetInitialTransform(init_transform3)
R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

transform3 = R.Execute(fixed, moving)

postIMSro_trans = resample_image(transform3, fixed, postIMSpimg3compl)
postIMSro_trans[~imcmaskch] = postIMSro_trans[~imcmaskch]/2
IMSpimg3compl[~imcmaskch] = IMSpimg3compl[~imcmaskch]/2
# plt.imshow(IMSpimg3compl.astype(float)-postIMSro_trans.astype(float))
# plt.show()


logging.info(f"Partial registration parameters: {transform3.GetParameters()}")
tmpfilename = f"{os.path.dirname(snakemake.log['stdout'])}/{os.path.basename(snakemake.log['stdout']).split('.')[0]}_partial_registration.ome.tiff"
logging.info(f"Save Image difference as: {tmpfilename}")
saveimage_tile(postIMSro_trans.astype(float)-IMSpimg3compl.astype(float), tmpfilename, resolution)


transform3_inv = sitk.AffineTransform(2)
transform3_inv.SetCenter(np.array(transform3.GetCenter())*resolution/stepsize)
transform3_inv.SetTranslation(np.array(transform3.GetTranslation())*resolution/stepsize)
transform3_inv.SetMatrix(transform3.GetMatrix())
transform3_inv.GetParameters()

# tm = sitk.CompositeTransform(2)
# tm.AddTransform(transform3_inv)
# tm.AddTransform(tmp_transform)
# pycpd_transform = composite2affine(tm, [0,0])
# tmp_transform.GetParameters()
# transform3_inv.GetParameters()
# pycpd_transform.GetParameters()

imzcoordsfilttrans2 = np.array([transform3_inv.TransformPoint(imzcoordsfilttrans[i,:].astype(float)) for i in range(imzcoordsfilttrans.shape[0])])
# plt.scatter(imzcoordsfilttrans2[:,1]*stepsize/resolution, imzcoordsfilttrans2[:,0]*stepsize/resolution,color="red",alpha=0.5)
# plt.scatter(centsred[:,1]*stepsize/resolution, centsred[:,0]*stepsize/resolution,color="blue",alpha=0.5)
# plt.show()





#####################################
logging.info("Find matching IMS and postIMS points")
contours,_ = cv2.findContours(imcmaskch.astype(np.uint8),cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours = np.squeeze(contours[0])
poly = shapely.geometry.Polygon(contours)
poly = poly.buffer(2)
tpls = [shapely.geometry.Point(centsred[i,:]/resolution*stepsize) for i in range(centsred.shape[0])]
pconts = np.array([poly.contains(tpls[i]) for i in range(len(tpls))])
centsredinmask = centsred[pconts]

kdt = KDTree(imzcoordsfilttrans2, leaf_size=30, metric='euclidean')
centsred_distances, indices = kdt.query(centsredinmask, k=1, return_distance=True)
centsred_has_match = centsred_distances.flatten()<0.5
kdt = KDTree(centsredinmask, leaf_size=30, metric='euclidean')
imz_distances, indices = kdt.query(imzcoordsfilttrans2, k=1, return_distance=True)
imz_has_match = imz_distances.flatten()<0.5

centsredfilt = centsredinmask[centsred_has_match,:]
imzcoordsfilt = imzcoordsfilttrans2[imz_has_match,:]

# matches = skimage.feature.match_descriptors(centsredfilt, tmpimzrot[imz_has_match,:], max_distance=1)
# dst = centsredfilt[matches[:,0],:]
# src = imzcoordsfilt[matches[:,1],:]
# import random
# random.seed(45)
# # model_robust, inliers = skimage.measure.ransac((src, dst), skimage.transform.EuclideanTransform, min_samples=21, residual_threshold=0.02, max_trials=10000)
# model_robust, inliers = skimage.measure.ransac((src, dst), skimage.transform.AffineTransform, min_samples=21, residual_threshold=0.02, max_trials=1000)
# model_robust
# R_reg = model_robust.params[:2,:2]
# t_reg = model_robust.translation
# postIMScoordsout = np.matmul(centsredfilt,R_reg)+t_reg



logging.info("Run point cloud registration")
# reg = pycpd.RigidRegistration(Y=centsredfilt.astype(float), X=imzcoordsfilt.astype(float), w=0, s=1, scale=False)
# postIMScoordsout, (s_reg, R_reg, t_reg) = reg.register()
n_points_ims_total = np.sum(np.array([poly.contains(shapely.geometry.Point(imzcoordsfilttrans2[i,:]/resolution*stepsize)) for i in range(len(imzcoordsfilttrans2))]))
if (centsredfilt.shape[0] > 10) and (centsredfilt.shape[0] > (n_points_ims_total*0.1)):
    reg = pycpd.AffineRegistration(Y=centsredfilt.astype(float), X=imzcoordsfilt.astype(float), w=0, s=1, scale=False)
    postIMScoordsout, (R_reg, t_reg) = reg.register()
    tmpfilename = f"{os.path.dirname(snakemake.log['stdout'])}/{os.path.basename(snakemake.log['stdout']).split('.')[0]}_pycpd_registration.svg"
    plt.close()
    plt.scatter(imzcoordsfilt[:,0]*stepsize, imzcoordsfilt[:,1]*stepsize,color="red",alpha=0.5)
    plt.scatter(postIMScoordsout[:,0]*stepsize, postIMScoordsout[:,1]*stepsize,color="blue",alpha=0.5)
    plt.title("matching points")
    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches(20,20)
    fig.savefig(tmpfilename)
else:
    R_reg = np.array([[1,0],[0,1]])
    t_reg = np.array([0,0])


# Invert Transformation
if R_reg[1,0]!=R_reg[0,1]:
    # R_reg_inv = np.array([[1-(R_reg[1,1]-1),-R_reg[0,1]],[-R_reg[1,0],1-(R_reg[0,0]-1)]])
    R_reg_inv = np.array([[1-(R_reg[0,0]-1),-R_reg[1,0]],[-R_reg[0,1],1-(R_reg[1,1]-1)]])
else:
    R_reg_inv = np.array([[R_reg[0,0],-R_reg[0,1]],[-R_reg[1,0],R_reg[1,1]]])

# imzcoordsfilttrans2 = np.matmul(imzcoordsfilttrans,R_reg_inv) + -t_reg

pycpd_transform = sitk.AffineTransform(2)
pycpd_transform.SetCenter(np.array([0.0,0.0]).astype(np.double))
pycpd_transform.SetMatrix(R_reg_inv.flatten().astype(np.double))
pycpd_transform.SetTranslation(-t_reg)


tmpfilename = f"{os.path.dirname(snakemake.log['stdout'])}/{os.path.basename(snakemake.log['stdout']).split('.')[0]}_pycpd_registration_all.svg"
plt.close()
imzcoordsfilttrans3 = np.array([pycpd_transform.TransformPoint(imzcoordsfilttrans2[i,:].astype(float)) for i in range(imzcoordsfilttrans2.shape[0])])
plt.scatter(imzcoordsfilttrans3[:,1]*stepsize/resolution, imzcoordsfilttrans3[:,0]*stepsize/resolution,color="red",alpha=0.5)
plt.scatter(centsred[:,1]*stepsize/resolution, centsred[:,0]*stepsize/resolution,color="blue",alpha=0.5)
# plt.show()
fig = plt.gcf()
fig.set_size_inches(20,20)
fig.savefig(tmpfilename)

# combined transformation steps
tm = sitk.CompositeTransform(2)
tm.AddTransform(pycpd_transform)
tm.AddTransform(transform3_inv)
tm.AddTransform(tinv)
pycpd_transform_comb = composite2affine(tm, [0,0])
pycpd_transform_comb.GetParameters()


imzcoords_all = create_imz_coords(imzimg, None, imzrefcoords, imz_bbox, rotmat)
imzcoords_in = imzcoords_all + init_translation
imzcoordstransformed = np.array([pycpd_transform_comb.TransformPoint(imzcoords_in[i,:].astype(float)) for i in range(imzcoords_in.shape[0])])
tmpfilename = f"{os.path.dirname(snakemake.log['stdout'])}/{os.path.basename(snakemake.log['stdout']).split('.')[0]}_pycpd_registration_image.png"
plt.close()
plt.scatter(imzcoordstransformed[:,1]*stepsize/resolution, imzcoordstransformed[:,0]*stepsize/resolution,color="red")
plt.scatter(centsred[:,1]*stepsize/resolution, centsred[:,0]*stepsize/resolution,color="blue")
# plt.imshow(postIMSmpre)
plt.title("matching points")
fig = plt.gcf()
fig.set_size_inches(20,20)
fig.savefig(tmpfilename)
# plt.show()


logging.info("Init Transform (Euler2D):")
logging.info(f"Translation: {init_transform.GetTranslation()}")
logging.info(f"Rotation: {init_transform.GetMatrix()}")
logging.info(f"Center: {init_transform.GetCenter()}")
logging.info("SITK Transform (Euler2D):")
logging.info(f"Translation: {transform.GetTranslation()}")
logging.info(f"Rotation: {transform.GetMatrix()}")
logging.info(f"Center: {transform.GetCenter()}")
logging.info("SITK Transform Inverse (Euler2D):")
logging.info(f"Translation: {tinv.GetTranslation()}")
logging.info(f"Rotation: {tinv.GetMatrix()}")
logging.info(f"Center: {tinv.GetCenter()}")
logging.info("pycpd Transform (Euler2D):")
logging.info(f"Translation: {pycpd_transform.GetTranslation()}")
logging.info(f"Rotation: {pycpd_transform.GetMatrix()}")
logging.info(f"Center: {pycpd_transform.GetCenter()}")
del transform

logging.info("Final pycpd registration:")
logging.info(f"Number of points IMS: {imzcoordsfilt.shape[0]}")
logging.info(f"Number of points postIMS: {centsredfilt.shape[0]}")
logging.info(f"Translation: {pycpd_transform_comb.GetTranslation()}")
logging.info(f"Rotation: {pycpd_transform_comb.GetMatrix()}")

logging.info("Get mean error after registration")
kdt = KDTree(imzcoordstransformed, leaf_size=30, metric='euclidean')
distances, indices = kdt.query(centsred, k=1, return_distance=True)
mean_error = np.mean(distances)*stepsize
q95_error = np.quantile(distances,0.95)*stepsize
q75_error = np.quantile(distances,0.75)*stepsize
q50_error = np.quantile(distances,0.5)*stepsize
q25_error = np.quantile(distances,0.25)*stepsize
q05_error = np.quantile(distances,0.05)*stepsize

logging.info(f"Mean Error: {str(mean_error)}")
logging.info(f"Number of points: {str(len(distances))} / {str(len(imzcoordstransformed))}")
logging.info(f"Quantiles: \n\tQ05: {str(q05_error)}\n\tQ25: {str(q25_error)}\n\tQ50: {str(q50_error)}\n\tQ75: {str(q75_error)}\n\tQ95: {str(q95_error)}")

imzcoordstransformedinmask = imzcoordstransformed[np.array([poly.contains(shapely.geometry.Point(imzcoordstransformed[i,:]/resolution*stepsize)) for i in range(len(imzcoordstransformed))])]
distances_inmask, indices = kdt.query(centsredinmask, k=1, return_distance=True)
mean_error_inmask = np.mean(distances_inmask)*stepsize
q95_error_inmask = np.quantile(distances_inmask,0.95)*stepsize
q75_error_inmask = np.quantile(distances_inmask,0.75)*stepsize
q50_error_inmask = np.quantile(distances_inmask,0.5)*stepsize
q25_error_inmask = np.quantile(distances_inmask,0.25)*stepsize
q05_error_inmask = np.quantile(distances_inmask,0.05)*stepsize

logging.info(f"Mean Error in IMC mask: {str(mean_error_inmask)}")
logging.info(f"Number of points in IMC mask: {str(len(distances_inmask))} / {str(len(imzcoordstransformedinmask))}")
logging.info(f"Quantiles: \n\tQ05: {str(q05_error_inmask)}\n\tQ25: {str(q25_error_inmask)}\n\tQ50: {str(q50_error_inmask)}\n\tQ75: {str(q75_error_inmask)}\n\tQ95: {str(q95_error_inmask)}")

# mean error
reg_measure_dic = {
    "IMS_to_postIMS_part_mean_error": str(mean_error),
    "IMS_to_postIMS_part_quantile05_error": str(q05_error),
    "IMS_to_postIMS_part_quantile25_error": str(q25_error),
    "IMS_to_postIMS_part_quantile50_error": str(q50_error),
    "IMS_to_postIMS_part_quantile75_error": str(q75_error),
    "IMS_to_postIMS_part_quantile95_error": str(q95_error),
    "n_points_part": str(len(distances)),
    "n_points_part_total": str(len(imzcoordstransformed)),
    "IMS_to_postIMS_part_inmask_mean_error": str(mean_error_inmask),
    "IMS_to_postIMS_part_inmask_quantile05_error": str(q05_error_inmask),
    "IMS_to_postIMS_part_inmask_quantile25_error": str(q25_error_inmask),
    "IMS_to_postIMS_part_inmask_quantile50_error": str(q50_error_inmask),
    "IMS_to_postIMS_part_inmask_quantile75_error": str(q75_error_inmask),
    "IMS_to_postIMS_part_inmask_quantile95_error": str(q95_error_inmask),
    "n_points_part_inmask": str(len(distances_inmask)),
    "n_points_part_total_inmask": str(len(imzcoordstransformedinmask)),
    }

logging.info("Initiate napari_imsmicrolink widget to save data")
## data wrangling to get correct output
vie = napari_imsmicrolink._dock_widget.IMSMicroLink(napari.Viewer(show=False))
vie.ims_pixel_map = napari_imsmicrolink.data.ims_pixel_map.PixelMapIMS(imzmlfile)
vie.ims_pixel_map.ims_res = stepsize
vie._tform_c.tform_ctl.target_mode_combo.setItemText(int(0),'Microscopy')
vie._data.micro_d.res_info_input.setText(str(resolution))
for regi in imzuqregs[imzuqregs != regionimz]:
    vie.ims_pixel_map.delete_roi(roi_name = str(regi), remove_padding=False)

logging.info("Create Transformation matrix")
# initial rotation of imz
tm1 = sitk.Euler2DTransform()
tm1.SetTranslation([0,0])
transparamtm1 = ((np.asarray(imzimg.shape).astype(np.double))/2*stepsize-stepsize/2).astype(np.double)
tm1.SetCenter(transparamtm1)
tm1.SetMatrix(rotmat.flatten().astype(np.double))
logging.info(f"1. Transformation: {tm1.GetName()}")
logging.info(tm1.GetParameters())

# Translation because of IMS crop
tm2 = sitk.TranslationTransform(2)
yshift = init_translation[1]*stepsize
xshift = init_translation[0]*stepsize
tm2.SetParameters(np.array([xshift,yshift]).astype(np.double))
logging.info(f"2. Transformation: {tm2.GetName()}")
logging.info(tm2.GetParameters())

# Registration of points 
# tm3 = sitk.Euler2DTransform()
tm3 = sitk.AffineTransform(2)
tm3.SetCenter(np.array([0,0]).astype(np.double))
tm3_rotmat = pycpd_transform_comb.GetMatrix()
tm3.SetMatrix(tm3_rotmat)
tm3_translation = np.array(pycpd_transform_comb.GetTranslation())*stepsize
tm3.SetTranslation(tm3_translation)
logging.info(f"3. Transformation: {tm3.GetName()}")
logging.info(tm3.GetParameters())


# Translation because of postIMS crop
tm4 = sitk.TranslationTransform(2)
yshift = ymin
xshift = xmin
tm4.SetParameters(np.array([xshift,yshift]).astype(np.double))
logging.info(f"4. Transformation: {tm4.GetName()}")
logging.info(tm4.GetParameters())



# combine transforms to single affine transform
tm = sitk.CompositeTransform(2)
tm.AddTransform(tm4)
tm.AddTransform(tm3)
tm.AddTransform(tm2)
tm.AddTransform(tm1)
tmfl = composite2affine(tm, [0,0])
logging.info(f"Combined Transformation: {tmfl.GetName()}")
logging.info(tmfl.GetParameters())

# Since axis are flipped, rotate transformation
tmfl.SetTranslation(np.flip(np.array(tmfl.GetTranslation())))
tmpmat = tmfl.GetMatrix()
# tmfl.SetMatrix(np.array([tmpmat[:2],tmpmat[2:]]).T.flatten())
tmfl.SetMatrix(np.array([[tmpmat[3],tmpmat[2]],[tmpmat[1],tmpmat[0]]]).flatten())

# tmfl.GetTranslation()
# pmap_coord_data.keys()
# xs = np.array(pmap_coord_data['x_padded'].tolist())*stepsize
# ys = np.array(pmap_coord_data['y_padded'].tolist())*stepsize
# tmpcoords = np.stack([xs,ys],axis=1)

# tmpcoordstrans = np.array([tmfl.TransformPoint(tmpcoords[i,:].astype(float)) for i in range(tmpcoords.shape[0])])
# tmpcoordstrans

# np.min(tmpcoordstrans,axis=0)-np.array([ymin,xmin])
# np.max(tmpcoordstrans,axis=0)-np.array([ymax,xmax])

# # check match
# postIMScut = readimage_crop(postIMS_file, [int(xmin/resolution), int(ymin/resolution), int(xmax/resolution), int(ymax/resolution)])
# for i in range(-3,4,1):
#     for j in range(-3,4,1):
#         xinds = (tmpcoordstrans[:,1]/resolution+i-xmin/resolution).astype(int)
#         yinds = (tmpcoordstrans[:,0]/resolution+j-ymin/resolution).astype(int)
#         xb = np.logical_and(xinds >= 0, xinds <= (postIMScut.shape[0]-1))
#         yb = np.logical_and(yinds >= 0, yinds <= (postIMScut.shape[1]-1))
#         inds = np.logical_and(xb,yb)
#         xinds = xinds[inds]
#         yinds = yinds[inds]
#         postIMScut[xinds,yinds,:] = [0,0,255]


# import matplotlib.pyplot as plt
# plt.imshow(postIMScut)
# plt.show()



logging.info("Apply transformation")
vie.image_transformer.affine_transform = tmfl
vie.image_transformer.inverse_affine_transform = tmfl.GetInverse()
vie.image_transformer.output_spacing = [resolution, resolution]
vie.image_transformer._get_np_matrices()
vie.microscopy_image = napari_imsmicrolink.data.tifffile_reader.TiffFileRegImage(postIMS_file)
vie._add_ims_data()
vie._data.ims_d.res_info_input.setText(str(stepsize))

logging.info("Write data")
vie._write_data(
    project_name = project_name,
    output_dir = output_dir,
    output_filetype = ".h5"
)

project_metadata, pmap_coord_data = vie._generate_pmap_coords_and_meta(project_name)
(
    transformed_coords_ims,
    transformed_coords_micro,
    transformed_coords_micro_px,
) = vie._transform_ims_coords_to_microscopy()

print(pmap_coord_data)
print(transformed_coords_ims)
print(transformed_coords_micro)
pmap_coord_data["x_micro_ims_px"] = transformed_coords_ims[:, 0]
pmap_coord_data["y_micro_ims_px"] = transformed_coords_ims[:, 1]
pmap_coord_data["x_micro_physical"] = transformed_coords_micro[:, 0]
pmap_coord_data["y_micro_physical"] = transformed_coords_micro[:, 1]
pmap_coord_data["x_micro_px"] = transformed_coords_micro_px[:, 0]
pmap_coord_data["y_micro_px"] = transformed_coords_micro_px[:, 1]


logging.info("Get mean error after registration")
centsredfilttrans = centsredfilt*stepsize+np.array([xshift,yshift])
imzcoordstrans = np.stack([np.array(pmap_coord_data["y_micro_physical"].to_list()).T,np.array(pmap_coord_data["x_micro_physical"].to_list()).T], axis=1)

kdt = KDTree(centsredfilttrans, leaf_size=30, metric='euclidean')
distances, indices = kdt.query(imzcoordstrans, k=1, return_distance=True)
point_to_keep = distances[:,0]<0.5*stepsize
imzcoordstransfilt = imzcoordstrans[point_to_keep,:]
vie.image_transformer.target_pts = centsredfilttrans
vie.image_transformer.source_pts = imzcoordstransfilt
logging.info("Error: "+ str(np.mean(distances[point_to_keep,0])))
logging.info("Number of points: "+ str(np.sum(point_to_keep)))
reg_measure_dic['IMS_to_postIMS_error'] = str(np.mean(distances[point_to_keep,0]))
reg_measure_dic['n_points'] = str(np.sum(point_to_keep))
json.dump(reg_measure_dic, open(ims_to_postIMS_regerror,"w"))

pmeta_out_fp = Path(output_dir) / f"{project_name}-IMSML-meta.json"
logging.info("Save data")
with open(pmeta_out_fp, "w") as json_out:
    json.dump(project_metadata, json_out, indent=1, cls=NpEncoder)

coords_out_fp = Path(output_dir) / f"{project_name}-IMSML-coords.h5"
napari_imsmicrolink.utils.coords.pmap_coords_to_h5(pmap_coord_data, coords_out_fp)


# check match
postIMScut = readimage_crop(postIMS_file, [int(xmin/resolution), int(ymin/resolution), int(xmax/resolution), int(ymax/resolution)])
for i in [-2,-1,0,1,2]:
    for j in [-2,-1,0,1,2]:
        xindsc = (centsred[:,0]/resolution*stepsize).astype(int)+i
        yindsc = (centsred[:,1]/resolution*stepsize).astype(int)+j
        xb = np.logical_and(xindsc >= 0, xindsc <= (postIMScut.shape[0]-1))
        yb = np.logical_and(yindsc >= 0, yindsc <= (postIMScut.shape[1]-1))
        inds = np.logical_and(xb,yb)
        xindsc = xindsc[inds]
        yindsc = yindsc[inds]
        postIMScut[xindsc,yindsc,:] = [255,0,0]
        xinds = (np.array(pmap_coord_data["y_micro_physical"].to_list())/resolution+i-xmin/resolution).astype(int)
        yinds = (np.array(pmap_coord_data["x_micro_physical"].to_list())/resolution+j-ymin/resolution).astype(int)
        xb = np.logical_and(xinds >= 0, xinds <= (postIMScut.shape[0]-1))
        yb = np.logical_and(yinds >= 0, yinds <= (postIMScut.shape[1]-1))
        inds = np.logical_and(xb,yb)
        xinds = xinds[inds]
        yinds = yinds[inds]
        coled = postIMScut[xinds,yinds,0]==255
        postIMScut[xinds[np.logical_not(coled)],yinds[np.logical_not(coled)],:] = [0,0,255]
        postIMScut[xinds[coled],yinds[coled],:] = [255,0,255]

# tmpy = np.array(pmap_coord_data["y_micro_physical"].to_list())/resolution-xmin/resolution
# tmpx = np.array(pmap_coord_data["x_micro_physical"].to_list())/resolution-xmin/resolution
# [np.arange(int(tmpy[0]-pixelsize/2),int(tmpy[0]+pixelsize/2)),np.arange(int(tmpy[0]-pixelsize/2),int(tmpy[0]+pixelsize/2)),np.repeat(int(tmpy[0]-pixelsize/2),pixelsize),np.repeat(int(tmpy[0]+pixelsize/2),pixelsize)]
# [np.repeat(int(tmpx[0]-pixelsize/2),pixelsize),np.repeat(int(tmpx[0]+pixelsize/2),pixelsize),np.arange(int(tmpx[0]-pixelsize/2),int(tmpx[0]+pixelsize/2)),np.arange(int(tmpx[0]-pixelsize/2),int(tmpx[0]+pixelsize/2))]





# add imc location
imcmask = readimage_crop(imc_mask_file, [int(xmin), int(ymin), int(xmax), int(ymax)])
imcmaskch = skimage.morphology.convex_hull_image(imcmask>0)
imcmaskchi = skimage.morphology.isotropic_erosion(imcmaskch, 1)
imcmaskb = np.logical_and(np.logical_not(imcmaskchi),imcmaskch)
imcmaskbr = skimage.transform.resize(imcmaskb, postIMScut.shape, preserve_range = True)[:,:,0]
postIMScut[imcmaskbr] = [255,255,255]


saveimage_tile(postIMScut, ims_to_postIMS_regerror_image, resolution)

# import matplotlib.pyplot as plt
# plt.imshow(postIMScut)
# plt.show()

logging.info("Finished")
