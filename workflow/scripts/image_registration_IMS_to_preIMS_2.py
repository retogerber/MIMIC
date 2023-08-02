import pandas as pd
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
from image_registration_IMS_to_preIMS_utils import readimage_crop, prepare_image_for_sam, create_ring_mask, composite2affine, saveimage_tile, normalize_image, get_image_shape
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
# stepsize = 20
# stepsize = 10
stepsize = float(snakemake.params["IMS_pixelsize"])
# pixelsize = 16
# pixelsize = 8 
pixelsize = stepsize*float(snakemake.params["IMS_shrink_factor"])
# resolution = 1
resolution = float(snakemake.params["IMC_pixelsize"])
# rotation_imz = 180
# rotation_imz = 0
rotation_imz = float(snakemake.params["IMS_rotation_angle"])
assert(rotation_imz in [-270,-180,-90,0,90,180,270])
logging.info("Rotation angle: "+str(rotation_imz))
logging.info("IMS stepsize: "+str(stepsize))
logging.info("IMS pixelsize: "+str(pixelsize))
logging.info("Microscopy pixelsize: "+str(resolution))
# postIMS_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/postIMS/test_split_ims_postIMS_reduced.ome.tiff" 
# postIMSr_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/postIMS/test_split_ims_postIMS_reduced_mask.ome.tiff"
# imzmlfile = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMS/IMS_test_split_ims_1.imzML" 
# imc_mask_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_001_transformed.ome.tiff" 
# output_table = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMS/test_split_ims_IMS_to_postIMS_matches.csv"
# output_table="/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMS/test_split_ims_test_split_ims_2_IMS_to_postIMS_matches.csv"
# postIMS_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/postIMS/test_combined_postIMS.ome.tiff"
# postIMS_file = "/home/retger/Downloads/cirrhosis_TMA_postIMS_reduced.ome.tiff"
# postIMS_file = "/home/retger/Downloads/Lipid_TMA_3781_postIMS_reduced.ome.tiff"
# postIMS_file = "/home/retger/Downloads/Lipid_TMA_3781_postIMS.ome.tiff"
# postIMS_file = "/home/retger/Downloads/NASH_HCC_TMA_postIMS.ome.tiff"
# resolution = 0.22537
postIMS_file = snakemake.input["postIMS_downscaled"]
# postIMSr_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/postIMS/test_combined_postIMS_reduced_mask.ome.tiff"
# postIMSr_file = "/home/retger/Downloads/cirrhosis_TMA_postIMS_reduced_mask.ome.tiff"
# postIMSr_file = "/home/retger/Downloads/Lipid_TMA_3781_postIMS_reduced_mask.ome.tiff"
# postIMSr_file = "/home/retger/Downloads/NASH_HCC_TMA_postIMS_reduced_mask.ome.tiff"
postIMSr_file = snakemake.input["postIMSmask_downscaled"]
# imzmlfile = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMS/IMS_test_combined.imzML"
# imzmlfile = "/home/retger/Downloads/cirrhosis_TMA_IMS.imzML"
# imzmlfile = "/home/retger/Downloads/pos_mode_lipids_tma_02022023_imzml.imzML"
# imzmlfile = "/home/retger/Downloads/hcc-tma-3_aaxl_20raster_06132022-total ion count.imzML"
imzmlfile = snakemake.input["imzml"]
# imc_mask_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_002_transformed.ome.tiff"
# imc_mask_file = "/home/retger/Downloads/Lipid_TMA_37819_009_transformed.ome.tiff"
# imc_mask_file = "/home/retger/Downloads/NASH_HCC_TMA-2_001_transformed.ome.tiff"
imc_mask_file = snakemake.input["IMCmask"]
# output_table = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMS/test_combined_test_combined_IMS_to_postIMS_matches.csv"
# output_table = "/home/retger/Downloads/cirrhosis_TMA_cirrhosis_TMA_IMS_IMS_to_postIMS_matches.csv"
# output_table = "/home/retger/Downloads/Lipid_TMA_3781_pos_mode_lipids_tma_02022023_imzml_IMS_to_postIMS_matches.csv"
# output_table = "/home/retger/Downloads/NASH_HCC_TMA_NASH_HCC_TMA_IMS_IMS_to_postIMS_matches.csv"
output_table = snakemake.input["IMS_to_postIMS_matches"]

# ims_to_postIMS_regerror = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_001_IMS_to_postIMS_reg_auto_metrics.json"
ims_to_postIMS_regerror = snakemake.output["IMS_to_postIMS_error"]
ims_to_postIMS_regerror_image = snakemake.output["IMS_to_postIMS_error_image"]

# ims_to_postIMS_matching_points_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_001_IMS_to_postIMS_reg_auto_matching_points.png"

# coordsfile_out = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMS/postIMS_to_IMS_test_split_pre-IMSML-coords.h5"
coordsfile_out = snakemake.output["imsml_coords_fp"]
output_dir = os.path.dirname(coordsfile_out)

imspixel_inscale = 4
imspixel_outscale = 2


logging.info("Read imzml")
# read imzml file
imz = napari_imsmicrolink.data.ims_pixel_map.PixelMapIMS(imzmlfile)
# stepsize (not actually used)
imz.ims_res = stepsize
# create image mask
imzimg = imz._make_pixel_map_at_ims(randomize=False, map_type="minimized")
# rotate 180 degrees
imzimg = skimage.transform.rotate(imzimg,rotation_imz, preserve_range=True)

# create imz region image
y_extent, x_extent, y_coords, x_coords = imz._get_xy_extents_coords(map_type="minimized")
imzregions = np.zeros((y_extent, x_extent), dtype=np.uint8)
imzregions[y_coords, x_coords] = imz.regions
imzregions = skimage.transform.rotate(imzregions,rotation_imz, preserve_range=True)
imzregions = np.round(imzregions)
imzuqregs = np.unique(imz.regions)

# reference coordinates, actually in data
imzrefcoords = np.stack([imz.y_coords_min,imz.x_coords_min],axis=1)
del imz


logging.info("Read postIMS region bounding box")
# read crop bbox
dfmeta = pd.read_csv(output_table)
imc_samplename = os.path.splitext(os.path.splitext(os.path.split(imc_mask_file)[1])[0])[0].replace("_transformed","")
# imc_project = "cirrhosis_TMA"
# imc_project="test_split_ims"
# imc_project="test_combined"
# imc_project="NASH_HCC_TMA"
# imc_project = "Lipid_TMA_3781"
imc_project = os.path.split(os.path.split(os.path.split(os.path.split(imc_mask_file)[0])[0])[0])[1]

project_name = "postIMS_to_IMS_"+imc_project+"_"+imc_samplename

img_shape = get_image_shape(postIMS_file)
inds_arr = np.logical_and(dfmeta["project_name"] == imc_project, dfmeta["sample_name"] == imc_samplename)
xmin = dfmeta[inds_arr]["postIMS_xmin"].tolist()[0]-int(imspixel_outscale*stepsize)
xmin = 0 if xmin<0 else xmin
ymin = dfmeta[inds_arr]["postIMS_ymin"].tolist()[0]-int(imspixel_outscale*stepsize)
ymin = 0 if ymin<0 else ymin
xmax = dfmeta[inds_arr]["postIMS_xmax"].tolist()[0]+int(imspixel_outscale*stepsize)
xmax = img_shape[0] if xmax>img_shape[0] else xmax
ymax = dfmeta[inds_arr]["postIMS_ymax"].tolist()[0]+int(imspixel_outscale*stepsize)
ymax = img_shape[1] if ymax>img_shape[1] else ymax
logging.info(f"xmin: {xmin}")
logging.info(f"xmax: {xmax}")
logging.info(f"ymin: {ymin}")
logging.info(f"ymax: {ymax}")
# needed:
regionimz = dfmeta[inds_arr]["imzregion"].tolist()[0]


logging.info("Read cropped postIMS")
# xmin = postIMSxmins[0]
# ymin = postIMSymins[0]
# xmax = postIMSxmaxs[0]
# ymax = postIMSymaxs[0]
# subset mask
postIMScut = readimage_crop(postIMS_file, [int(xmin/resolution), int(ymin/resolution), int(xmax/resolution), int(ymax/resolution)])
# postIMScut = readimage_crop(postIMS_file, [xmin, ymin, xmax, ymax])
postIMScut = prepare_image_for_sam(postIMScut, 1)
# postIMScut = prepare_image_for_sam(postIMScut, resolution)
logging.info("Median filter")
# postIMSmpre = skimage.filters.median(postIMScut, skimage.morphology.disk( np.round(((stepsize-pixelsize)/resolution)/3)))
ksize = np.round(((stepsize-pixelsize)/resolution)/3).astype(int)*2
ksize = ksize+1 if ksize%2==0 else ksize
postIMSmpre = cv2.medianBlur(postIMScut, ksize)


# sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
# sitk_postIMScut = sitk.GetImageFromArray(postIMScut, sitk.sitkUInt8)
# sitk_postIMSmpre = sitk.Median(sitk_postIMScut, [9, 9, 9])
del postIMScut

logging.info("Read cropped postIMS mask")
postIMSrcut = readimage_crop(postIMSr_file, [xmin, ymin, xmax, ymax])
logging.info("Resize")
postIMSrcut = skimage.transform.resize(postIMSrcut, postIMSmpre.shape, preserve_range = True)
postIMSrcut = np.round(postIMSrcut).astype(np.uint8)
logging.info("Create ringmask")
postIMSringmask = create_ring_mask(postIMSrcut, (1/resolution)*stepsize*imspixel_outscale, (1/resolution)*stepsize*imspixel_inscale)
logging.info("Isotropic dilation")
postIMSoutermask = skimage.morphology.isotropic_dilation(postIMSrcut, (1/resolution)*stepsize*imspixel_outscale)

# ksize=int((1/resolution)*stepsize*4*2)
# kernel = np.ones((ksize,ksize),dtype=int)
# postIMSoutermask2 = cv2.dilate(src=postIMSrcut, kernel = kernel)

# subset and filter postIMS image
kersize = int(stepsize/resolution/2)
kersize = kersize-1 if kersize%2==0 else kersize
kernel = np.zeros((kersize,kersize))
kernel[int((kersize-1)/2),:]=1
kernel[:,int((kersize-1)/2)]=1
logging.info(f"Disk radius for rank threshold filter: {kersize}")

# local rank filter, results in binary image
tmp1 = skimage.filters.rank.threshold(postIMSmpre, skimage.morphology.disk(kersize))
logging.info("Mean filter")
# mean filter, with cross shape footprint
# tmp2 = skimage.filters.rank.mean(tmp1*255, skimage.morphology.disk(kersize))
# tmp2 = skimage.filters.rank.mean(tmp1*255, kernel)
tmp2 = cv2.filter2D(src = tmp1*255, ddepth=-1, kernel=kernel/np.sum(kernel))
# combine with original image
tmp3 = 0.5*postIMSmpre + 0.5*tmp2
tmp3 = skimage.exposure.equalize_adapthist(normalize_image(tmp3))
tmp3 = normalize_image(tmp3)*255
tmp3 = tmp3.astype(np.uint8)
postIMSmforpoints = tmp3.copy()
del tmp1, tmp2, tmp3
postIMSm = postIMSmforpoints.copy()
postIMSm[np.logical_not(postIMSringmask)] = 0

# from: https://stackoverflow.com/a/26392655
def get_angle(p0, p1=np.array([0,0]), p2=None):
    ''' compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    '''
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)



logging.info("Find best threshold for IMS laser ablation marks detection")
from sklearn.neighbors import KDTree
def points_from_mask(
        mask: np.ndarray, 
        pixelsize: np.double, 
        resolution: np.double,
        stepsize: np.double):
    '''
    Extract point from binary mask
    '''
    
    # filter detected regions to obtain ablation marks
    labs = skimage.measure.label(mask.astype(np.double))
    regs = skimage.measure.regionprops(labs)
    del labs
    # filter by area 
    areas = np.asarray([c.area for c in regs])
    # minimal area is a circle with radius pixelsize -3 microns
    # min_radius = ((pixelsize-6)/2/resolution)
    # def radi(x):
    #     return (x/8+(x/8-1)*2.5)
    # ys=np.array([8,16,24,30,36])
    # xs=np.array([radi(y) for y in ys])
    # plt.scatter(xs,ys/2)
    # plt.show()
    min_radius = (pixelsize/8+(pixelsize/8-1)*2.5)/resolution
    min_area = min_radius**2*np.pi if min_radius>0 else np.pi
    area_range = [min_area,(pixelsize/resolution)**2]
    inran = np.logical_and(areas > area_range[0], areas < area_range[1])
    del areas
    regsred = np.asarray(regs)[inran]
    del regs, inran
    # check length
    if not isinstance(regsred, np.ndarray) or len(regsred)<6:
        return np.zeros((0,2))

    # filter by ratio of x-slice to y-slice
    x_slice = np.array([r.slice[0].stop-r.slice[0].start for r in regsred])
    y_slice = np.array([r.slice[1].stop-r.slice[1].start for r in regsred])
    slice_ratio = np.asarray([x_slice/y_slice]).flatten()
    # is_round = np.abs(np.log10(slice_ratio))<np.log10(pixelsize/(pixelsize-4))
    is_round = np.abs(np.log10(slice_ratio))<np.log10(2)
    regsred = np.asarray(regsred)[is_round]

    # check length
    if not isinstance(regsred, np.ndarray) or len(regsred)<6:
        return np.zeros((0,2))
    cents = np.asarray([r.centroid for r in regsred])
    # to IMS scale
    centsred = cents*resolution/stepsize
    # centsred = cents/((1/resolution)*stepsize)
    del cents

    # create neighborhood adjacency matrix, connections based on:
    #   - distance
    #   - angle 
    # filter according to distance to nearest neighbors,
    # expected for a grid are distances close to 1
    kdt = KDTree(centsred, leaf_size=30, metric='euclidean')
    distances, indices = kdt.query(centsred, k=5, return_distance=True)
    dists = distances[:,1:]
    del kdt, distances
    to_keep_dist = np.logical_and(dists>0.925, dists < 1.075)

    # restrict angles to nearest neighbors to certain ranges
    angles = list()
    for i in range(indices.shape[0]):
        angles.append(np.array([get_angle(centsred[j,:]-centsred[i,:],[0,0],[1,0]) for j in indices[i,1:]]))
    absangles = np.abs(np.array(angles))
    to_keep_angle = np.logical_or(
            np.logical_or(absangles < 10, absangles > 170),
            np.logical_and(absangles > 80, absangles < 100),
    )
    to_keep = np.logical_and(to_keep_angle, to_keep_dist)
    some_neighbors = np.sum(to_keep, axis=1)>0
    
    from scipy.sparse import lil_array
    # create adjaceny matrix
    adjmat = lil_array((to_keep.shape[0],to_keep.shape[0]),dtype=bool)
    indices_sub = indices[some_neighbors,:]
    to_keep_sub = to_keep[some_neighbors,:]
    for i in range(len(indices_sub[:,0])):
        for j in indices_sub[i,1:][to_keep_sub[i,:]]:
            adjmat[indices_sub[i,0],j]=True
            adjmat[j,indices_sub[i,0]]=True

    from scipy.sparse.csgraph import connected_components
    concomp = connected_components(adjmat)
    components, n_points_in_component = np.unique(concomp[1], return_counts=True)
    comps_to_keep = components[n_points_in_component>9]
    to_keep_adj = np.array([c in comps_to_keep for c in concomp[1]])
    centsred = centsred[to_keep_adj,:]
    return centsred

logging.info("Find best threshold for points")
# find best threshold by maximizing number of points that fullfill criteria
# grid search
# broad steps
thresholds = list(range(127,250,10))
n_points = []
for th in thresholds:
    # threshold
    postIMSmb = postIMSm>th
    centsred = points_from_mask(postIMSmb, pixelsize=pixelsize, resolution=resolution, stepsize=stepsize)
    logging.info(f"threshold: {th}, npoints= {centsred.shape[0]}")
    n_points.append(centsred.shape[0])

threshold = np.asarray(thresholds)[n_points == np.max(n_points)][0]
# fine steps
thresholds = list(range(threshold-9,threshold+9))
n_points = []
for th in thresholds:
    # threshold
    postIMSmb = postIMSm>th
    centsred = points_from_mask(postIMSmb, pixelsize=pixelsize, resolution=resolution, stepsize=stepsize)
    logging.info(f"threshold: {th}, npoints= {centsred.shape[0]}")
    n_points.append(centsred.shape[0])

threshold = np.asarray(thresholds)[n_points == np.max(n_points)][0]

logging.info("Max number of IMS pixels detected: "+str(np.max(n_points)))
logging.info("Corresponding threshold: "+str(threshold))

logging.info("Apply threshold")
postIMSm = postIMSmforpoints.copy()
postIMSm[np.logical_not(postIMSoutermask)] = 0
postIMSmb = postIMSm>threshold

# get points from complete image
centsred = points_from_mask(postIMSmb, pixelsize=pixelsize, resolution=resolution, stepsize=stepsize)
# plt.imshow(postIMSmb)
# plt.scatter(centsred[:,1]*stepsize/resolution,centsred[:,0]*stepsize/resolution)
# plt.show()

# rotate IMS coordinates 
if rotation_imz in [-180, 180]:
    rotmat = np.asarray([[-1, 0], [0, -1]])
elif rotation_imz in [90, -270]:
    rotmat = np.asarray([[0, 1], [-1, 0]])
elif rotation_imz in [-90, 270]:
    rotmat = np.asarray([[0, -1], [1, 0]])
else:
    rotmat = np.asarray([[1, 0], [0, 1]])

from typing import Union
def create_imz_coords(imzimg: np.ndarray, mask: Union[None, np.ndarray], imzrefcoords: np.ndarray, bbox, rotmat):
    # create coordsmatrices for IMS
    indmatx = np.zeros(imzimg.shape)
    for i in range(imzimg.shape[0]):
        indmatx[i,:] = list(range(imzimg.shape[1]))
    indmatx = indmatx.astype(np.uint32)
    indmaty = np.zeros(imzimg.shape)
    for i in range(imzimg.shape[1]):
        indmaty[:,i] = list(range(imzimg.shape[0]))
    indmaty = indmaty.astype(np.uint32)

    xminimz = bbox[0]
    yminimz = bbox[1]
    xmaximz = bbox[2]
    ymaximz = bbox[3]

    # create coordinates for registration
    if mask is None:
        imzxcoords = indmatx[xminimz:xmaximz,yminimz:ymaximz].flatten()
        imzycoords = indmaty[xminimz:xmaximz,yminimz:ymaximz].flatten()
    else: 
        imzxcoords = indmatx[xminimz:xmaximz,yminimz:ymaximz][mask]
        imzycoords = indmaty[xminimz:xmaximz,yminimz:ymaximz][mask]
    imzcoords = np.stack([imzycoords, imzxcoords],axis=1)

    center_point=np.max(imzrefcoords,axis=0)/2
    imzrefcoords = np.dot(rotmat, (imzrefcoords - center_point).T).T + center_point

    # filter for coordinates that are in data
    in_ref = []
    for i in range(imzcoords.shape[0]):
        in_ref.append(np.any(np.logical_and(imzcoords[i,0] == imzrefcoords[:,0],imzcoords[i,1] == imzrefcoords[:,1])))

    in_ref = np.array(in_ref)
    imzcoords = imzcoords[in_ref,:]
    return imzcoords

logging.info("IMS bbox:")
# subset region for IMS
imz_bbox = skimage.measure.regionprops((imzregions == regionimz).astype(np.uint8))[0].bbox
xminimz, yminimz, xmaximz, ymaximz = imz_bbox
logging.info(f"xminimz: {xminimz}")
logging.info(f"xmaximz: {xmaximz}")
logging.info(f"yminimz: {yminimz}")
logging.info(f"ymaximz: {ymaximz}")
# logging.info("Create IMS coordinates")
# imzcoords = create_imz_coords(imzimg, None, imzrefcoords, imz_bbox, rotmat)

# init_translation = (-np.min(imzcoords,axis=0).astype(float)+np.min(centsred,axis=0).astype(float)).astype(int)

logging.info("Prepare for initial registration with sitk")
logging.info("Prepare postIMS image")
postIMSpimg = np.zeros(postIMSm.shape, dtype=bool)
stepsize_px = int(stepsize/resolution)
stepsize_px_half = int(stepsize_px/2)
for i in range(centsred.shape[0]):
    xc = int(centsred[i,0]*stepsize_px)
    xr = [xc-stepsize_px_half,xc+stepsize_px_half]
    if xr[0]<0:
        xr[0]=0
    if (xr[1]+1)>postIMSpimg.shape[0]:
        xr[1]=postIMSpimg.shape[0]-1
    yc = int(centsred[i,1]*stepsize_px)
    yr = [yc-stepsize_px_half,yc+stepsize_px_half]
    if yr[0]<0:
        yr[0]=0
    if (yr[1]+1)>postIMSpimg.shape[1]:
        yr[1]=postIMSpimg.shape[1]-1
    postIMSpimg[xr[0]:(xr[1]+1),yr[0]:(yr[1]+1)] = True

logging.info("    - Closing")
postIMSpimg = cv2.morphologyEx(src=postIMSpimg.astype(np.uint8), op = cv2.MORPH_CLOSE, kernel = skimage.morphology.square(int(stepsize/resolution/4))).astype(bool)

logging.info("    - Remove small holes")
postIMSpimg = skimage.morphology.remove_small_holes(postIMSpimg, int((stepsize/resolution*10)**2))
logging.info("    - Rescale")
postIMSpimg = skimage.transform.rescale(postIMSpimg, 1/(stepsize/resolution)*10)
postIMSpimg = postIMSpimg.astype(float)*255

logging.info("Prepare IMS image")
imzimg_regin = imzimg[xminimz:xmaximz,yminimz:ymaximz]
logging.info("    - Rescale")
imzimg_regin = skimage.transform.rescale(imzimg_regin.astype(bool), 10)
imzimg_regin = imzimg_regin.astype(float)*255




logging.info("Convert images to sitk images")
fixed = sitk.GetImageFromArray(imzimg_regin)
moving = sitk.GetImageFromArray(postIMSpimg)

# initial transformation
init_transform = sitk.Euler2DTransform()
init_transform.SetTranslation(np.min(centsred,axis=0)[[1,0]]*10)

logging.info("Setup registration")
R = sitk.ImageRegistrationMethod()
R.SetMetricAsCorrelation()
R.SetMetricSamplingStrategy(R.REGULAR)
R.SetMetricSamplingPercentage(1)
R.SetInterpolator(sitk.sitkNearestNeighbor)
R.SetOptimizerAsGradientDescent(
    learningRate=1000, numberOfIterations=1000, 
    convergenceMinimumValue=1e-9, convergenceWindowSize=30,
    estimateLearningRate=R.EachIteration
)
R.SetOptimizerScalesFromIndexShift()
R.SetInitialTransform(init_transform)

def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.8f} "
        + f": {method.GetOptimizerPosition()}"
    )
# setup sitk images
R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

logging.info("Run registration")
# run registration
transform = R.Execute(fixed, moving)

# visualize
resampler = sitk.ResampleImageFilter()
resampler.SetTransform(transform)
resampler.SetReferenceImage(fixed)
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
postIMSro_trans = sitk.GetArrayFromImage(resampler.Execute(sitk.GetImageFromArray(postIMSpimg)))
tmpfilename = f"{os.path.dirname(snakemake.log['stdout'])}/{os.path.basename(snakemake.log['stdout']).split('.')[0]}_sitk_registration.ome.tiff"
logging.info(f"Save Image difference as: {tmpfilename}")
saveimage_tile(postIMSro_trans-imzimg_regin, tmpfilename, resolution)
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(nrows=1, ncols=3)
# ax[0].imshow(skimage.transform.resize(postIMSpimg,imzimg_regin.shape)-imzimg_regin)
# ax[1].imshow(postIMSro_trans-imzimg_regin)
# ax[2].imshow(skimage.transform.resize(postIMSpimg,imzimg_regin.shape)-postIMSro_trans)
# plt.show()

# plt.imshow(postIMSro_trans.astype(float)+imzimg_regin.astype(float))
# plt.show()

# inverse transformation
tinv = transform.GetInverse()
# tinv.SetTranslation(-np.array(tinv.GetTranslation())[[1,0]]/10)
if rotation_imz==180:
# works for rotation=180
    # tinv.SetTranslation(np.array(tinv.GetTranslation())[[0,1]]/10-0.5)
    tinv.SetTranslation(-np.array(tinv.GetTranslation())[[1,0]]/10+1.5)
elif rotation_imz==0:
    tinv.SetTranslation(-np.array(tinv.GetTranslation())[[1,0]]/10+1.5)
# tinv.SetMatrix(transform.GetMatrix())



imzringmask = create_ring_mask(imzimg[xminimz:xmaximz,yminimz:ymaximz], imspixel_outscale, imspixel_inscale)
logging.info("Create IMS boundary coordinates")
imzcoords = create_imz_coords(imzimg, imzringmask, imzrefcoords, imz_bbox, rotmat)
# plt.imshow(imzimg[xminimz:xmaximz,yminimz:ymaximz])
# plt.scatter(imzcoords_in[:,1], imzcoords_in[:,0],color="red",alpha=0.5)
# plt.show()

logging.info("Create IMS boundary coordinates")
# init_translation = -np.min(imzcoords,axis=0).astype(int)
init_translation = -np.array([xminimz,yminimz]).astype(int)
imzcoords_in = imzcoords + init_translation
imzcoordsfilttrans = np.array([tinv.TransformPoint(imzcoords_in[i,:].astype(float)) for i in range(imzcoords_in.shape[0])])

tmpfilename = f"{os.path.dirname(snakemake.log['stdout'])}/{os.path.basename(snakemake.log['stdout']).split('.')[0]}_sitk_registration.svg"
plt.close()
# plt.imshow(imzimg[xminimz:xmaximz,yminimz:ymaximz])
# plt.scatter(imzcoords_in[:,1], imzcoords_in[:,0],color="black",alpha=0.5)
plt.scatter(imzcoordsfilttrans[:,1], imzcoordsfilttrans[:,0],color="red",alpha=0.5)
plt.scatter(centsred[:,1], centsred[:,0],color="blue",alpha=0.5)
plt.title("matching points")
fig = plt.gcf()
fig.set_size_inches(20,20)
fig.savefig(tmpfilename)
# plt.show()


logging.info("Create postIMS boundary from observed points")
# create new ringmask based on found points
postIMSn = np.zeros(postIMSmforpoints.shape)
for i in range(centsred.shape[0]):
    postIMSn[int(centsred[i,0]/resolution*stepsize),int(centsred[i,1]/resolution*stepsize)] = 1
postIMSn = skimage.morphology.convex_hull_image(postIMSn)
postIMSnringmask = create_ring_mask(postIMSn, imspixel_outscale*stepsize/resolution, imspixel_inscale*stepsize/resolution)
# plt.imshow(postIMSnringmask)
# plt.scatter(centsred[:,1]/resolution*stepsize,centsred[:,0]/resolution*stepsize)
# plt.show()

logging.info("Extract postIMS boundary points")
postIMSm = postIMSmforpoints.copy()
postIMSm[np.logical_not(postIMSnringmask)] = 0
postIMSmb = postIMSm>threshold
centsred = points_from_mask(postIMSmb, pixelsize=pixelsize, resolution=resolution, stepsize=stepsize)
# plt.scatter(imzcoordsfilttrans[:,0], imzcoordsfilttrans[:,1],color="red")
# plt.scatter(centsred[:,0], centsred[:,1],color="blue")
# plt.title("matching points")
# plt.show()


n_points_ls = []
for xsh in np.linspace(-3,3,25):
    for ysh in np.linspace(-3,3,25):
        matches = skimage.feature.match_descriptors(centsred, imzcoordsfilttrans + np.array([xsh,ysh]), max_distance=1)
        n_points_ls.append([matches.shape[0],xsh,ysh])
n_points_arr = np.array(n_points_ls)
n_points_arr[n_points_arr[:,0] == np.max(n_points_arr[:,0]),:]
xsh = n_points_arr[n_points_arr[:,0] == np.max(n_points_arr[:,0]),1][0]
ysh = n_points_arr[n_points_arr[:,0] == np.max(n_points_arr[:,0]),2][0]
# plt.scatter(imzcoordsfilttrans[:,0]+xsh, imzcoordsfilttrans[:,1]+ysh,color="red")
# plt.scatter(centsred[:,0], centsred[:,1],color="blue")
# plt.title("matching points")
# plt.show()



logging.info("Find matching IMS and postIMS points")
# eval matching points
kdt = KDTree(imzcoordsfilttrans+np.array([xsh,ysh]), leaf_size=30, metric='euclidean')
centsred_distances, indices = kdt.query(centsred, k=1, return_distance=True)
# centsred_has_match = centsred_distances.flatten()<1
centsred_has_match = centsred_distances.flatten()<0.75
kdt = KDTree(centsred, leaf_size=30, metric='euclidean')
imz_distances, indices = kdt.query(imzcoordsfilttrans+np.array([xsh,ysh]), k=1, return_distance=True)
# imz_has_match = imz_distances.flatten()<1
imz_has_match = imz_distances.flatten()<0.75

centsredfilt = centsred[centsred_has_match,:]
imzcoordsfilt = imzcoordsfilttrans[imz_has_match,:]

# plt.scatter(imzcoordsfilt[:,0]+xsh, imzcoordsfilt[:,1]+ysh,color="red")
# plt.scatter(centsredfilt[:,0], centsredfilt[:,1],color="blue")
# plt.title("matching points")
# plt.show()



if False:
    postIMScent = skimage.measure.regionprops(skimage.measure.label(postIMSnringmask))[0].centroid
    postIMScentred = np.array(postIMScent)/stepsize*resolution

    # plt.imshow(postIMSmpre)
    # plt.scatter(centsred[:,1]*stepsize,centsred[:,0]*stepsize)
    # plt.scatter(postIMScentred[1]*stepsize,postIMScentred[0]*stepsize)
    # plt.show()

    angles = np.array([get_angle(centsred[i,:],postIMScentred) for i in range(centsred.shape[0])])
    angles = angles[centsred_has_match]
    angles = angles +180
    angles_order = pd.factorize(list(angles), sort=True)[0]
    angles_sort = np.array(sorted(angles))
    angles_sort_order = pd.factorize(list(angles_sort), sort=True)[0]

    angle_max_error = 2.5
    angles_min = angles_sort
    angles_max = angles_sort+angle_max_error*2

    angles_ls = [[angles_min[0], angles_max[0]]]
    counts_ls = [1]
    pos=0
    angles_group = [[angles_min[0], pos]]
    for i in range(len(angles_sort)-1):
        if (angles_ls[pos][1]-angles_min[i+1])>0:
            angles_ls[pos][1]=angles_max[i+1]
            counts_ls[pos]+=1
        else:
            print(f"new: {i}")
            angles_ls.append([angles_min[i+1],angles_max[i+1]])
            counts_ls.append(1)
            pos+=1
        angles_group.append([angles_min[i], pos])
    
    angles_group = np.array(angles_group)
    angles_group = np.concatenate([angles_group,np.reshape(np.array(list(range(angles_group.shape[0]))),(angles_group.shape[0],1))],axis=1)
    if len(angles_ls)>1:
        do_merge=False
        if angles_ls[-1][1]>360 and angles_ls[0][0]<0:
            do_merge=True
        elif angles_ls[-1][1]>360 and angles_ls[0][0]>=0:
            if angles_ls[-1][1]%360 - angles_ls[0][0] > 0:
                do_merge=True
        elif angles_ls[-1][1]<=360 and angles_ls[0][0]<0:
            if angles_ls[-1][1] - angles_ls[0][0]%360 < 0:
                do_merge=True
        if do_merge:
            angles_ls[0][0]=angles_ls[-1][0]
            del angles_ls[-1]
            counts_ls[0]+=counts_ls[-1]
            del counts_ls[-1]
            angles_group[angles_group[:,1]==angles_group[-1,1],1] = 0
    angles_ls = [np.array(ang)-angle_max_error for ang in angles_ls]
    angles_ls
    counts_ls

    if len(angles_ls) > 1:
        total_angles = [(ang[1]-ang[0])%360 for ang in angles_ls]
        points_per_angle = np.array(counts_ls)/np.array(total_angles)
        grps = np.unique(angles_group[:,1])
        if np.min(points_per_angle)*3<np.max(points_per_angle):
            wanted_ratio = 2*np.min(points_per_angle)
            inds_to_keep = []
            for grp in grps[points_per_angle>wanted_ratio]:
                n_points = np.array(counts_ls)[grps == grp][0]
                npoints_to_keep = int(np.floor(n_points/(points_per_angle[grps==grp]-wanted_ratio))[0])
                import random
                ransam = random.sample(range(n_points),npoints_to_keep)
                inds_to_keep += list(angles_group[angles_group[:,1] == grp,:][ransam,2])
            for grp in grps[points_per_angle<=wanted_ratio]:
                inds_to_keep += list(angles_group[angles_group[:,1] == grp,:][:,2])



    # imzcent = np.mean(imzcoords,axis=0)
    # angles = np.array([get_angle(imzcoords[i,:].astype(float),imzcent) for i in range(imzcoords.shape[0])])
    # angles = angles +180
    # # plt.hist(angles, bins=360)
    # # plt.show()

    # boolls = []
    # for i in range(len(angles_ls)):
    #     if angles_ls[i][0]>angles_ls[i][1]:
    #         boolls.append(np.logical_or(angles > angles_ls[i][0],angles < angles_ls[i][1]))
    #     else:
    #         boolls.append(np.logical_and(angles > angles_ls[i][0],angles < angles_ls[i][1]))
    # to_keep = np.sum(np.stack(boolls),axis=0).astype(bool)

    # imzcoords = imzcoords[to_keep,:]

    # # # plt.imshow(postIMSmb)
    # plt.scatter(imzcoords[:,1]*stepsize/resolution,imzcoords[:,0]*stepsize/resolution)
    # plt.scatter(imzcent[1]*stepsize/resolution,imzcent[0]*stepsize/resolution)
    # plt.show()


    centsredfilt = centsredfilt[np.array(inds_to_keep).astype(int),:]
    matches = skimage.feature.match_descriptors(centsredfilt, imzcoordsfilt + np.array([xsh,ysh]), max_distance=1)
    imzcoordsfilt = imzcoordsfilt[matches[:,1],:]

# matches = skimage.feature.match_descriptors(centsredfilt, imzcoordsfilt + np.array([xsh,ysh]), max_distance=1)
# dst = centsredfilt[matches[:,0],:]
# src = imzcoordsfilt[matches[:,1],:]
# import random
# random.seed(45)
# model_robust, inliers = skimage.measure.ransac((src, dst), skimage.transform.EuclideanTransform, min_samples=3, residual_threshold=0.2, max_trials=500)
# model_robust
# R_reg = model_robust.params[:2,:2]
# t_reg = model_robust.translation
# postIMScoordsout = np.matmul(centsredfilt,R_reg)+t_reg



logging.info("Run point cloud registration")
reg = pycpd.RigidRegistration(Y=centsredfilt.astype(float), X=imzcoordsfilt.astype(float), w=0, s=1, scale=False)
postIMScoordsout, (s_reg, R_reg, t_reg) = reg.register()
# postIMScoordsin = centsredfilt[centsred_has_match,:]
# postIMScoordsout = postIMScoordsout[centsred_has_match,:]
# imzcoordsin = imzcoordsfilt[imz_has_match,:]


tmpfilename = f"{os.path.dirname(snakemake.log['stdout'])}/{os.path.basename(snakemake.log['stdout']).split('.')[0]}_pycpd_registration.svg"
plt.close()
plt.scatter(imzcoordsfilt[:,0]*stepsize, imzcoordsfilt[:,1]*stepsize,color="red",alpha=0.5)
plt.scatter(postIMScoordsout[:,0]*stepsize, postIMScoordsout[:,1]*stepsize,color="blue",alpha=0.5)
plt.title("matching points")
# plt.show()
fig = plt.gcf()
fig.set_size_inches(20,20)
fig.savefig(tmpfilename)


# imzcoordsfilttrans2 = np.matmul(imzcoordsfilttrans,R_reg.T) + -t_reg

pycpd_transform = sitk.Euler2DTransform()
pycpd_transform.SetCenter(np.array([0.0,0.0]).astype(np.double))
pycpd_transform.SetMatrix(R_reg.flatten().astype(np.double))
pycpd_transform.SetTranslation(-t_reg)

tmpfilename = f"{os.path.dirname(snakemake.log['stdout'])}/{os.path.basename(snakemake.log['stdout']).split('.')[0]}_pycpd_registration_all.svg"
plt.close()
imzcoordsfilttrans2 = np.array([pycpd_transform.TransformPoint(imzcoordsfilttrans[i,:].astype(float)) for i in range(imzcoordsfilttrans.shape[0])])
plt.scatter(centsred[:,0]*stepsize/resolution, centsred[:,1]*stepsize/resolution,color="blue",alpha=0.5)
plt.scatter(imzcoordsfilttrans2[:,0]*stepsize/resolution, imzcoordsfilttrans2[:,1]*stepsize/resolution,color="red",alpha=0.5)
fig = plt.gcf()
fig.set_size_inches(20,20)
fig.savefig(tmpfilename)
# plt.imshow(postIMSmpre.T)
# plt.title("matching points")
# plt.show()

tm = sitk.CompositeTransform(2)
tm.AddTransform(pycpd_transform)
tm.AddTransform(tinv)
pycpd_transform_comb = composite2affine(tm, [0,0])
pycpd_transform_comb.GetParameters()



imzcoords_all = create_imz_coords(imzimg, None, imzrefcoords, imz_bbox, rotmat)
# imzcoords_in = imzcoords_all - np.min(imzcoords_all,axis=0)
imzcoords_in = imzcoords_all + init_translation
imzcoordstransformed = np.array([pycpd_transform_comb.TransformPoint(imzcoords_in[i,:].astype(float)) for i in range(imzcoords_in.shape[0])])
# plt.scatter(imzcoordstransformed[:,0]*stepsize/resolution, imzcoordstransformed[:,1]*stepsize/resolution,color="red")
# plt.scatter(centsred[:,0]*stepsize/resolution, centsred[:,1]*stepsize/resolution,color="blue")
# plt.imshow(postIMSmpre.T)
# plt.title("matching points")
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


logging.info("Final pycpd registration:")
logging.info(f"Number of points IMS: {imzcoordsfilt.shape[0]}")
logging.info(f"Number of points postIMS: {postIMScoordsout.shape[0]}")
logging.info(f"Translation: {pycpd_transform_comb.GetTranslation()}")
logging.info(f"Rotation: {pycpd_transform_comb.GetMatrix()}")

# t_reg = t_reg + init_translation
# logging.info(f"Translation plus init translation: {t_reg}")
# plt.scatter(TY[:,0]*stepsize, TY[:,1]*stepsize,color="blue")
# plt.scatter(centsredfilt[:,0]*stepsize, centsredfilt[:,1]*stepsize,color="red")
# plt.imshow(postIMSm.T)
# plt.title("matching points")
# plt.show()
# # plt.savefig(ims_to_postIMS_matching_points_file, dpi=500)



logging.info("Get mean error after registration")
kdt = KDTree(imzcoordstransformed, leaf_size=30, metric='euclidean')
distances, indices = kdt.query(centsred, k=1, return_distance=True)
mean_error = np.mean(distances)*stepsize
logging.info("Error: "+ str(mean_error))
logging.info("Number of points: "+ str(len(distances)))

# mean error
reg_measure_dic = {
    "IMS_to_postIMS_part_error": str(mean_error),
    "n_points_part": str(len(distances))
    }

# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].scatter(TY[:,0], TY[:,1])
# ax[0].set_title("IMS")
# ax[1].scatter(centsred[:,0], centsred[:,1])
# ax[1].set_title("postIMS")
# plt.show()

logging.info("Initiate napari_imsmicrolink widget to save data")
## data wrangling to get correct output
vie = napari_imsmicrolink._dock_widget.IMSMicroLink(napari.Viewer(show=False))
vie.ims_pixel_map = napari_imsmicrolink.data.ims_pixel_map.PixelMapIMS(imzmlfile)
vie.ims_pixel_map.ims_res = stepsize
vie._tform_c.tform_ctl.target_mode_combo.setItemText(int(0),'Microscopy')
vie._data.micro_d.res_info_input.setText(str(resolution))
# vie._data.micro_d.res_info_input.setText(str(1))
# vie._data.micro_d.res_info_input.setText(str(1/resolution))
# vie.image_transformer.target_pts = centsredfilt
# vie.image_transformer.source_pts = imzcoordsfilt
for regi in imzuqregs[imzuqregs != regionimz]:
    vie.ims_pixel_map.delete_roi(roi_name = str(regi), remove_padding=False)
#     vie.ims_pixel_map.delete_roi(roi_name = str(regi), remove_padding=True)


logging.info("Create Transformation matrix")
# initial rotation of imz
tm1 = sitk.Euler2DTransform()
tm1.SetTranslation([0,0])
# transparamtm1 = np.flip((np.asarray(imzimg.shape).astype(np.double))/2*stepsize-stepsize/2).astype(np.double)
# transparamtm1 = np.flip((np.asarray(imzimg.shape).astype(np.double))/2*stepsize).astype(np.double)
transparamtm1 = ((np.asarray(imzimg.shape).astype(np.double))/2*stepsize-stepsize/2).astype(np.double)
tm1.SetCenter(transparamtm1)
tm1.SetMatrix(rotmat.flatten().astype(np.double))
logging.info("1. Transformation: Rotation")
logging.info(tm1.GetParameters())

# Translation because of IMS crop
tm2 = sitk.TranslationTransform(2)
# yshift = ymin-(imzimg.shape[1]-ymaximz)*stepsize/resolution-stepsize/resolution
# yshift = ymin-stepsize/resolution
# yshift = ymin/resolution
yshift = init_translation[1]*stepsize
# xshift = xmin-(imzimg.shape[0]-xmaximz)*stepsize/resolution-stepsize/resolution
# xshift = xmin-stepsize/resolution
# xshift = xmin/resolution
xshift = init_translation[0]*stepsize
tm2.SetParameters(np.array([xshift,yshift]).astype(np.double))
logging.info("2. Transformation: Translation")
logging.info(tm2.GetParameters())

# Registration of points 
tm3 = sitk.Euler2DTransform()
# t_reg = t_reg + init_translation
# tm3.SetCenter(np.array([yshift+t_reg[1]*stepsize/resolution,xshift+t_reg[0]*stepsize/resolution]).astype(np.double))
# tm3.SetCenter(np.array([yshift+t_reg[1]*stepsize/resolution+stepsize/resolution/2,xshift+t_reg[0]*stepsize/resolution+stepsize/resolution/2]).astype(np.double))
# tm3.SetCenter(np.array([xshift+stepsize/2,yshift+stepsize/2]).astype(np.double))
# tm3.SetCenter(np.array([xshift,yshift]).astype(np.double))
tm3.SetCenter(np.array([0,0]).astype(np.double))
tm3_rotmat = pycpd_transform_comb.GetMatrix()
tm3.SetMatrix(tm3_rotmat)
# tm3_translation = np.flip(np.array(pycpd_transform_comb.GetTranslation()))*stepsize/resolution + init_translation*stepsize/resolution
tm3_translation = np.array(pycpd_transform_comb.GetTranslation())*stepsize
# tm3.SetTranslation(np.flip(tm3_translation))
if rotation_imz==180:
    # tm3.SetTranslation(tm3_translation*-1)
    tm3.SetTranslation(tm3_translation)
elif rotation_imz==0:
    tm3.SetTranslation(tm3_translation)
logging.info("3. Transformation: Euler2D")
logging.info(tm3.GetParameters())


# Translation because of postIMS crop
tm4 = sitk.TranslationTransform(2)
# yshift = ymin-(imzimg.shape[1]-ymaximz)*stepsize/resolution-stepsize/resolution
# yshift = ymin-stepsize/resolution
yshift = ymin
# yshift = ymin/resolution + init_translation[1]/resolution*stepsize
# xshift = xmin-(imzimg.shape[0]-xmaximz)*stepsize/resolution-stepsize/resolution
# xshift = xmin-stepsize/resolution
xshift = xmin
# xshift = xmin/resolution + init_translation[0]/resolution*stepsize
tm4.SetParameters(np.array([xshift,yshift]).astype(np.double))
logging.info("4. Transformation: Translation")
logging.info(tm4.GetParameters())



# combine transforms to single affine transform
tm = sitk.CompositeTransform(2)
tm.AddTransform(tm4)
tm.AddTransform(tm3)
tm.AddTransform(tm2)
tm.AddTransform(tm1)
tmfl = composite2affine(tm, [0,0])
logging.info("Combined Transformation: Affine")
logging.info(tmfl.GetParameters())

if rotation_imz==0:
    # Since axis are flipped, rotate transformation
    tmfl.SetTranslation(np.flip(np.array(tmfl.GetTranslation())))
    tmpmat = tmfl.GetMatrix()
    tmfl.SetMatrix(np.array([tmpmat[:2],tmpmat[2:]]).T.flatten())
if rotation_imz==180:
    tmfl.SetTranslation(np.flip(np.array(tmfl.GetTranslation())))
    tmpmat = tmfl.GetMatrix()
    tmfl.SetMatrix(np.array([tmpmat[:2],tmpmat[2:]]).T.flatten())

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


# plt.scatter(tmpcoordstrans[:,1]-ymin,tmpcoordstrans[:,0]-xmin)
# plt.scatter(centsred[:,1]*stepsize,centsred[:,0]*stepsize)
# plt.show()







logging.info("Apply transformation")
vie.image_transformer.affine_transform = tmfl
vie.image_transformer.inverse_affine_transform = tmfl.GetInverse()
vie.image_transformer.output_spacing = [resolution, resolution]
# vie.image_transformer.output_spacing = [1,1]
# vie.image_transformer.output_spacing = [1/resolution,1/resolution]
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

# vie._tform_c.tform_ctl.target_mode_combo.currentText()
# vie._transform_ims_coords_to_microscopy()


from pathlib import Path
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
# pmap_coord_data["xy_micro_ims_px"] = transformed_coords_ims
pmap_coord_data["x_micro_physical"] = transformed_coords_micro[:, 0]
pmap_coord_data["y_micro_physical"] = transformed_coords_micro[:, 1]
# pmap_coord_data["xy_micro_physical"] = transformed_coords_micro
pmap_coord_data["x_micro_px"] = transformed_coords_micro_px[:, 0]
pmap_coord_data["y_micro_px"] = transformed_coords_micro_px[:, 1]
# pmap_coord_data["xy_micro_px"] = transformed_coords_micro_px

# plt.scatter(pmap_coord_data["x_original"].to_list(),pmap_coord_data["y_original"].to_list())
# plt.scatter(pmap_coord_data["x_micro_physical"].to_list(),pmap_coord_data["y_micro_physical"].to_list())
# plt.show()


logging.info("Get mean error after registration")
centsredfilttrans = centsredfilt*stepsize+np.array([xshift,yshift])
imzcoordstrans = np.stack([np.array(pmap_coord_data["y_micro_physical"].to_list()).T,np.array(pmap_coord_data["x_micro_physical"].to_list()).T], axis=1)

kdt = KDTree(centsredfilttrans, leaf_size=30, metric='euclidean')
distances, indices = kdt.query(imzcoordstrans, k=1, return_distance=True)
point_to_keep = distances[:,0]<0.5*stepsize
imzcoordstransfilt = imzcoordstrans[point_to_keep,:]


logging.info("Error: "+ str(np.mean(distances[point_to_keep,0])))
logging.info("Number of points: "+ str(np.sum(point_to_keep)))
reg_measure_dic['IMS_to_postIMS_error'] = str(np.mean(distances[point_to_keep,0]))
reg_measure_dic['n_points'] = str(np.sum(point_to_keep))

json.dump(reg_measure_dic, open(ims_to_postIMS_regerror,"w"))


vie.image_transformer.target_pts = centsredfilttrans
vie.image_transformer.source_pts = imzcoordstransfilt


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
        xinds = (np.array(pmap_coord_data["y_micro_physical"].to_list())/resolution+i-xmin/resolution).astype(int)
        yinds = (np.array(pmap_coord_data["x_micro_physical"].to_list())/resolution+j-ymin/resolution).astype(int)
        xb = np.logical_and(xinds >= 0, xinds <= (postIMScut.shape[0]-1))
        yb = np.logical_and(yinds >= 0, yinds <= (postIMScut.shape[1]-1))
        inds = np.logical_and(xb,yb)
        xinds = xinds[inds]
        yinds = yinds[inds]
        postIMScut[xinds,yinds,:] = [0,0,255]

saveimage_tile(postIMScut, ims_to_postIMS_regerror_image, resolution)

# import matplotlib.pyplot as plt
# plt.imshow(postIMScut)
# plt.show()





# image_crop = postIMS[xmin:xmax,ymin:ymax]

# saveimage_tile(image_crop, ims_to_postIMS_regerror_image, resolution)
# skimage.io.imsave(ims_to_postIMS_regerror_image, image_crop)

# plt.imshow(postIMS)
# plt.scatter(pmap_coord_data["x_micro_physical"],pmap_coord_data["y_micro_physical"])
# plt.show()

# plt.imshow(skimage.transform.rescale(imzimg,stepsize, order=0))
# plt.scatter(pmap_coord_data["x_micro_physical"],pmap_coord_data["y_micro_physical"])
# plt.show()

# plt.imshow(imzimg)
# plt.scatter(pmap_coord_data["x_micro_physical"]/stepsize,pmap_coord_data["y_micro_physical"]/stepsize)
# plt.show()



# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].scatter(imzcoordstransfilt[:,0],imzcoordstransfilt[:,1], color="red")
# ax[0].scatter(centsredfilttrans[:,0],centsredfilttrans[:,1], color="blue")
# plt.show()



logging.info("Finished")
