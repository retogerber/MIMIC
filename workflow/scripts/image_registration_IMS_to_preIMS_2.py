import pandas as pd
import gc
import cv2
import SimpleITK as sitk
import napari_imsmicrolink
import skimage
import numpy as np
import json
import matplotlib.pyplot as plt
from image_registration_IMS_to_preIMS_utils import readimage_crop, prepare_image_for_sam, create_ring_mask, composite2affine, saveimage_tile, normalize_image, get_image_shape, create_imz_coords,get_rotmat_from_angle, concave_boundary_from_grid, concave_boundary_from_grid_holes
from sklearn.neighbors import KDTree
from scipy.sparse import lil_array
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import distance_transform_edt
import shapely
import shapely.affinity
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

# threads = 2
threads = int(snakemake.threads)
cv2.setNumThreads(threads)
logging.info("Start")

# parameters
# stepsize = 20
# stepsize = 10
stepsize = float(snakemake.params["IMS_pixelsize"])
# pixelsize = 16
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
# postIMS_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/postIMS/test_split_ims_postIMS_reduced.ome.tiff" 
# postIMSr_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/postIMS/test_split_ims_postIMS_reduced_mask.ome.tiff"
# imzmlfile = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMS/IMS_test_split_ims_1.imzML" 
# imc_mask_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_001_transformed.ome.tiff" 
# output_table = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMS/test_split_ims_IMS_to_postIMS_matches.csv"
# output_table="/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMS/test_split_ims_test_split_ims_2_IMS_to_postIMS_matches.csv"
# postIMS_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/postIMS/test_combined_postIMS_reduced.ome.tiff"
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
# imc_mask_file = "/home/retger/Downloads/Lipid_TMA_37819_025_transformed.ome.tiff"
# imc_mask_file = "/home/retger/Downloads/NASH_HCC_TMA-2_010_transformed.ome.tiff"
imc_mask_file = snakemake.input["IMCmask"]
# output_table = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMS/test_combined_IMS_test_combined_IMS_to_postIMS_matches.csv"
# output_table = "/home/retger/Downloads/cirrhosis_TMA_cirrhosis_TMA_IMS_IMS_to_postIMS_matches.csv"
# output_table = "/home/retger/Downloads/Lipid_TMA_3781_pos_mode_lipids_tma_02022023_imzml_IMS_to_postIMS_matches.csv"
# output_table = "/home/retger/Downloads/NASH_HCC_TMA_NASH_HCC_TMA_IMS_IMS_to_postIMS_matches.csv"
output_table = snakemake.input["IMS_to_postIMS_matches"]

masks_transform_filename = snakemake.output["masks_transform"]
gridsearch_transform_filename = snakemake.output["gridsearch_transform"]

postIMS_ablation_centroids_filename = snakemake.output["postIMS_ablation_centroids"]
metadata_to_save_filename = snakemake.output["metadata"]

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
gc.collect()


logging.info("Read postIMS region bounding box")
# read crop bbox
dfmeta = pd.read_csv(output_table)
imc_samplename = os.path.splitext(os.path.splitext(os.path.split(imc_mask_file)[1])[0])[0].replace("_transformed","")
imc_project = os.path.split(os.path.split(os.path.split(os.path.split(imc_mask_file)[0])[0])[0])[1]
# imc_project = "cirrhosis_TMA"
# imc_project="test_split_ims"
# imc_project="test_combined"
# imc_project="NASH_HCC_TMA"
# imc_project = "Lipid_TMA_3781"

project_name = "postIMS_to_IMS_"+imc_project+"-"+imc_samplename
# project_name = "postIMS_to_IMS_"+imc_project+"_"+imc_samplename

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
# subset mask
postIMScut = readimage_crop(postIMS_file, [int(xmin/resolution), int(ymin/resolution), int(xmax/resolution), int(ymax/resolution)])
postIMScut = prepare_image_for_sam(postIMScut, 1)
logging.info("Median filter")
ksize = np.round(((stepsize-pixelsize)/resolution)/3).astype(int)*2
ksize = ksize+1 if ksize%2==0 else ksize
postIMSmpre = cv2.medianBlur(postIMScut, ksize)
del postIMScut
gc.collect()

logging.info("Read cropped postIMS mask")
postIMSrcut = readimage_crop(postIMSr_file, [xmin, ymin, xmax, ymax])
logging.info("Resize")
postIMSrcut = skimage.transform.resize(postIMSrcut, postIMSmpre.shape, preserve_range = True)
postIMSrcut = np.round(postIMSrcut).astype(np.uint8)
logging.info("Create ringmask")
postIMSringmask = create_ring_mask(postIMSrcut, (1/resolution)*stepsize*imspixel_outscale, (1/resolution)*stepsize*imspixel_inscale)
logging.info("Isotropic dilation")
# postIMSoutermask = skimage.morphology.isotropic_dilation(postIMSrcut, (1/resolution)*stepsize*imspixel_outscale)
postIMSoutermask = cv2.morphologyEx(src=postIMSrcut.astype(np.uint8), op = cv2.MORPH_DILATE, kernel = skimage.morphology.square(2*int((1/resolution)*stepsize*imspixel_outscale))).astype(bool)
# postIMSoutermask_small = skimage.morphology.isotropic_dilation(postIMSrcut, (1/resolution)*stepsize)
postIMSoutermask_small = cv2.morphologyEx(src=postIMSrcut.astype(np.uint8), op = cv2.MORPH_DILATE, kernel = skimage.morphology.square(2*int((1/resolution)*stepsize))).astype(bool)
# postIMSinnermask = skimage.morphology.isotropic_erosion(postIMSrcut, (1/resolution)*stepsize*imspixel_inscale)
postIMSinnermask = cv2.morphologyEx(src=postIMSrcut.astype(np.uint8), op = cv2.MORPH_ERODE, kernel = skimage.morphology.square(2*int((1/resolution)*stepsize*imspixel_inscale))).astype(bool)
del postIMSrcut
gc.collect()

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
# tmp2 = skimage.filters.rank.mean(tmp1*255, kernel)
tmp2 = cv2.filter2D(src = tmp1*255, ddepth=-1, kernel=kernel/np.sum(kernel))
del tmp1
gc.collect()

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
def points_from_mask(
        mask: np.ndarray, 
        pixelsize: np.double, 
        resolution: np.double,
        stepsize: np.double,
        min_n: int = 9):
    '''
    Extract point from binary mask
    '''
    
    # filter detected regions to obtain ablation marks
    labs = skimage.measure.label(mask.astype(np.double))
    regs = skimage.measure.regionprops(labs)
    del labs
    # filter by area 
    areas = np.asarray([c.area for c in regs])
    # minimal area is a circle with radius (empirical guess based on pixelsize and resolution) 
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
    is_round = np.abs(np.log10(slice_ratio))<np.log10(2)
    regsred = np.asarray(regsred)[is_round]

    # check length
    if not isinstance(regsred, np.ndarray) or len(regsred)<6:
        return np.zeros((0,2))
    cents = np.asarray([r.centroid for r in regsred])
    # to IMS scale
    centsred = cents*resolution/stepsize
    del cents
    centsred = filter_points(centsred, min_n=min_n)
    return centsred

def filter_points(points, min_n:int = 9):
    # create neighborhood adjacency matrix, connections based on:
    #   - distance
    #   - angle 
    # filter according to distance to nearest neighbors,
    # expected for a grid are distances close to 1
    kdt = KDTree(points, leaf_size=30, metric='euclidean')
    distances, indices = kdt.query(points, k=5, return_distance=True)
    dists = distances[:,1:]
    del kdt, distances
    to_keep_dist = np.logical_and(dists>0.93, dists < 1.07)

    # restrict angles to nearest neighbors to certain ranges
    angles = list()
    for i in range(indices.shape[0]):
        angles.append(np.array([get_angle(points[j,:]-points[i,:],[0,0],[1,0]) for j in indices[i,1:]]))
    absangles = np.abs(np.array(angles))
    to_keep_angle = np.logical_or(
            np.logical_or(absangles < 9, absangles > 171),
            np.logical_and(absangles > 81, absangles < 99),
    )
    to_keep = np.logical_and(to_keep_angle, to_keep_dist)
    some_neighbors = np.sum(to_keep, axis=1)>0
    
    # create adjaceny matrix
    adjmat = lil_array((to_keep.shape[0],to_keep.shape[0]),dtype=bool)
    indices_sub = indices[some_neighbors,:]
    to_keep_sub = to_keep[some_neighbors,:]
    for i in range(len(indices_sub[:,0])):
        for j in indices_sub[i,1:][to_keep_sub[i,:]]:
            adjmat[indices_sub[i,0],j]=True
            adjmat[j,indices_sub[i,0]]=True

    concomp = connected_components(adjmat)
    components, n_points_in_component = np.unique(concomp[1], return_counts=True)
    comps_to_keep = components[n_points_in_component>min_n]
    to_keep_adj = np.array([c in comps_to_keep for c in concomp[1]])
    points = points[to_keep_adj,:]
    return points

def combine_points(p1, p2, threshold=0.5):
    pcomb = np.vstack([p1,p2])
    n = [pcomb.shape[0]]
    for _ in range(100):
        kdt = KDTree(pcomb, leaf_size=30, metric='euclidean')
        distances, indices = kdt.query(pcomb, k=2, return_distance=True)

        same_pixel = distances[:,1] < threshold

        x=indices[same_pixel,0]
        y=indices[same_pixel,1]
        pairs = []
        for i in range(len(x)):
            pairs.append(sorted([x[i],y[i]]))
        pairs = sorted(pairs)
        doublet_to_keep = []
        for i in range(len(pairs)):
            if pairs[i][0] not in doublet_to_keep:
                doublet_to_keep.append(pairs[i][0])

        pcomb_1 = pcomb[np.logical_not(same_pixel),:]
        pcomb_2 = pcomb[doublet_to_keep,:]
        pcomb = np.vstack([pcomb_1,pcomb_2])
        n.append(pcomb.shape[0])
        if n[-1]==n[-2]:
            break

    return pcomb


def scores_points_from_mask(th, img, maskb_dist, min_n:int = 9):
    maskb = img>th
    centsred = points_from_mask(maskb, pixelsize=pixelsize, resolution=resolution, stepsize=stepsize, min_n=min_n)
    cents = (centsred*stepsize/resolution).astype(int)
    dists = maskb_dist[cents[:,0],cents[:,1]]/stepsize*resolution
    inv_dists = np.array([1/d if d!=0 else 0 for d in dists])
    return (centsred.shape[0], np.sum(dists<=2), np.sum(inv_dists), centsred)

from multiprocessing import Pool, Value
import functools
def find_threshold(img: np.ndarray, maskb_dist: np.ndarray, thr_range=[127,250], min_n:int = 9, threads:int=1):

    # find best threshold by maximizing number of points that fullfill criteria
    # grid search
    # broad steps
    scorer = functools.partial(scores_points_from_mask, img=img, maskb_dist = maskb_dist, min_n=min_n)
    thresholds = list(range(thr_range[0],thr_range[1],10))
    if threads>1:
        with Pool(threads) as p:
            n_points, n_border_points, weighted_dist, centsredls1 = zip(*p.map(scorer, thresholds))
    else:
        n_points, n_border_points, weighted_dist, centsredls1 = zip(*map(scorer, thresholds))

    wt1 = np.array(weighted_dist)
    wt1m = 1 if np.max(wt1)==0 else np.max(wt1)
    nt1 = np.array(n_points)
    nt1m = 1 if np.max(nt1)==0 else np.max(nt1)
    bt1 = np.array(n_border_points)
    bt1m = 1 if np.max(bt1)==0 else np.max(bt1)
    scores1 = 2*wt1/wt1m+nt1/nt1m+4*bt1/bt1m
    max_score=np.max(scores1)
    max_points = np.max(n_points)
    max_border_points = np.max(n_border_points)
    max_weighted_dist = np.max(weighted_dist)

    threshold = np.asarray(thresholds)[scores1 == max_score][0]

    # finer steps
    thresholds = list(range(threshold-9,threshold+10,3))
    if threads>1:
        with Pool(threads) as p:
            n_points, n_border_points, weighted_dist, centsredls2 = zip(*p.map(scorer, thresholds))
    else:
        n_points, n_border_points, weighted_dist, centsredls2 = zip(*map(scorer, thresholds))

    wt2 = np.array(weighted_dist)
    wt2m = 1 if np.max(wt2)==0 else np.max(wt2)
    nt2 = np.array(n_points)
    nt2m = 1 if np.max(nt2)==0 else np.max(nt2)
    bt2 = np.array(n_border_points)
    bt2m = 1 if np.max(bt2)==0 else np.max(bt2)
    scores2 = 2*wt2/wt2m+nt2/nt2m+4*bt2/bt2m
    max_score=np.max(scores2)
    max_points = np.max(n_points)
    max_border_points = np.max(n_border_points)
    max_weighted_dist = np.max(weighted_dist)

    threshold = np.asarray(thresholds)[scores2 == max_score][0]

    # fine steps
    thresholds = list(range(threshold-2,threshold+3))
    if threads>1:
        with Pool(threads) as p:
            n_points, n_border_points, weighted_dist, centsredls3 = zip(*p.map(scorer, thresholds))
    else:
        n_points, n_border_points, weighted_dist, centsredls3 = zip(*map(scorer, thresholds))

    wt3 = np.array(weighted_dist)
    wt3m = 1 if np.max(wt3)==0 else np.max(wt3)
    nt3 = np.array(n_points)
    nt3m = 1 if np.max(nt3)==0 else np.max(nt3)
    bt3 = np.array(n_border_points)
    bt3m = 1 if np.max(bt3)==0 else np.max(bt3)
    scores3 = 2*wt3/wt3m+nt3/nt3m+4*bt3/bt3m
    max_score=np.max(scores3)
    max_points = np.max(n_points)
    max_border_points = np.max(n_border_points)
    max_weighted_dist = np.max(weighted_dist)

    threshold = np.asarray(thresholds)[scores3 == max_score][0]

    wt=np.concatenate([wt1,wt2,wt3])
    wtm = 1 if np.max(wt)==0 else np.max(wt)
    nt=np.concatenate([nt1,nt2,nt3])
    ntm = 1 if np.max(nt)==0 else np.max(nt)
    bt=np.concatenate([bt1,bt2,bt3])
    btm = 1 if np.max(bt)==0 else np.max(bt)
    scores = 2*wt/wtm+nt/ntm+4*bt/btm
    max_score=np.max(scores3)

    centsredls = centsredls1 + centsredls2 + centsredls3
    scoresred = np.array([scores[i] for i in range(len(scores)) if centsredls[i].shape[0]>0])
    centsredls = [c for c in centsredls if c.shape[0]>0]
    
    inds = np.arange(len(scoresred))[np.flip(np.argsort(scoresred))]
    inds = inds[scoresred[np.flip(np.argsort(scoresred))] >= 0.9*max_score]
    centsredlssort = [centsredls[i] for i in inds]

    from functools import reduce
    if len(centsredlssort)>0:
        if len(centsredlssort[0])>0:
            centsred = reduce(combine_points, centsredlssort)
            centsred = filter_points(centsred, min_n)
        else:
            centsred = centsredlssort[0]
    else:
        centsred = np.zeros((0,0))

    return threshold, max_points, max_border_points, max_weighted_dist, centsred

# def score_find_w(w, img_median, img_convolved, maskb_dist, mask2, threads):
def score_find_w(w):
    postIMSm = w*img_median_in + (1-w)*img_convolved_in
    postIMSm = skimage.exposure.equalize_adapthist(normalize_image(postIMSm))
    postIMSm = normalize_image(postIMSm)*255
    postIMSm = postIMSm.astype(np.uint8)
    postIMSm[np.logical_not(mask2_in)] = 0
    if threads_in>1:
        if current_threshold_in.value>0:
            thr_range = [int(current_threshold_in.value)-31,int(current_threshold_in.value)+31]
            thr_range[0] = 0 if thr_range[0]<0 else thr_range[0]
            thr_range[1] = 255 if thr_range[1]>255 else thr_range[1]
        else:
            thr_range = [127,250]
    else:
        thr_range = [127,250]

    threshold, max_points, max_border_points, max_weighted_dist, centsred= find_threshold(postIMSm, maskb_dist_in, thr_range=thr_range, threads=1)
    logging.info(f"weight: {w}, threshold: {threshold}, n_points: {max_points}, border points: {max_border_points}, sum of inverse distances {max_weighted_dist}")
    if threads_in>1:
        current_threshold_in.value = threshold
    return (threshold, max_points, max_border_points, max_weighted_dist, w, centsred)

def init_worker(img_median, img_convolved, maskb_dist, mask2, threads, current_threshold):
    global img_median_in
    img_median_in = img_median
    global img_convolved_in
    img_convolved_in = img_convolved
    global maskb_dist_in
    maskb_dist_in = maskb_dist
    global mask2_in
    mask2_in = mask2
    global threads_in
    threads_in = threads
    global current_threshold_in
    current_threshold_in = current_threshold


logging.info("Find best threshold for points (outer points)")
def find_w(img_median: np.ndarray, img_convolved: np.ndarray, mask1: np.ndarray, mask2: np.ndarray, ws, threads:int=1):
    maskb_dist = distance_transform_edt(mask1) - distance_transform_edt(~mask1)

    current_threshold = Value('i', 0)    
    with Pool(threads, initializer= init_worker, initargs=(img_median, img_convolved, maskb_dist, mask2, threads, current_threshold)) as p:
        thresholds, n_points, n_border_points, weighted_dist, wsobs, centsredls = zip(*p.map(score_find_w, ws))

    wt = np.array(weighted_dist)
    wtm = 1 if np.max(wt)==0 else np.max(wt)
    nt = np.array(n_points)
    ntm = 1 if np.max(nt)==0 else np.max(nt)
    bt = np.array(n_border_points)
    btm = 1 if np.max(bt)==0 else np.max(bt)
    scores = 2*wt/wtm+nt/ntm+4*bt/btm
    max_score=np.max(scores)
    threshold = np.asarray(thresholds)[scores == max_score][0]
    w = np.asarray(wsobs)[scores == max_score][0]

    scoresred = [scores[i] for i in range(len(scores)) if centsredls[i].shape[0]>0]
    centsredls = [c for c in centsredls if c.shape[0]>0]
    inds = np.arange(len(scoresred))[np.flip(np.argsort(scoresred))]
    centsredlssort = [centsredls[i] for i in inds]

    from functools import reduce
    centsred = reduce(combine_points, centsredlssort)
    centsred = filter_points(centsred)

    return max_score, threshold, w, centsred

ws = [0.001,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
max_score_outer, threshold_outer, w_outer, centsred_outer = find_w(postIMSmpre, tmp2, postIMSoutermask_small, postIMSringmask, ws, threads=threads)
del postIMSringmask, postIMSoutermask_small
gc.collect()

logging.info("Find best threshold for points (inner points)")
ws = [0.001,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
max_score_inner, threshold_inner, w_inner, centsred_inner = find_w(postIMSmpre, tmp2, postIMSinnermask, postIMSinnermask, ws, threads=threads)

logging.info("Max score (outer): "+str(max_score_outer))
logging.info("Corresponding threshold (outer): "+str(threshold_outer))
logging.info("Corresponding weight (outer): "+str(w_outer))
logging.info("Max score (inner): "+str(max_score_inner))
logging.info("Corresponding threshold (inner): "+str(threshold_inner))
logging.info("Corresponding weight (inner): "+str(w_inner))

def points_from_mask_two_thresholds(
        img_median: np.ndarray,
        img_convolved: np.ndarray,
        mask_outer: np.ndarray,
        mask_inner: np.ndarray,
        w_outer: np.double,
        w_inner: np.double,
        threshold_outer:np.double,
        threshold_inner:np.double,
        pixelsize: np.double,
        resolution: np.double,
        stepsize: np.double,
        min_n_outer: int = 9, 
        min_n_inner: int = 9):

    logging.info("Apply threshold (outer)")
    tmpimg = w_outer*img_median + (1-w_outer)*img_convolved
    tmpimg = skimage.exposure.equalize_adapthist(normalize_image(tmpimg))
    tmpimg = normalize_image(tmpimg)*255
    tmpimg = tmpimg.astype(np.uint8)
    tmpimg[np.logical_not(mask_outer)] = 0
    tmpimgb = tmpimg>threshold_outer
    # get points from complete image
    centsred_outer = points_from_mask(tmpimgb, pixelsize=pixelsize, resolution=resolution, stepsize=stepsize, min_n=min_n_outer)

    logging.info("Apply threshold (inner)")
    tmpimg = w_inner*img_median + (1-w_inner)*img_convolved
    tmpimg = skimage.exposure.equalize_adapthist(normalize_image(tmpimg))
    tmpimg = normalize_image(tmpimg)*255
    tmpimg = tmpimg.astype(np.uint8)
    tmpimg[np.logical_not(mask_inner)] = 0
    tmpimgb = tmpimg>threshold_inner
    # get points from complete image
    centsred_inner = points_from_mask(tmpimgb, pixelsize=pixelsize, resolution=resolution, stepsize=stepsize, min_n=min_n_inner)

    logging.info("Combine points")
    centsred = np.vstack([centsred_outer,centsred_inner])
    kdt = KDTree(centsred, leaf_size=30, metric='euclidean')
    distances, indices = kdt.query(centsred, k=2, return_distance=True)

    same_pixel = distances[:,1]<0.25

    x=np.arange(centsred.shape[0])[same_pixel]
    y=indices[same_pixel,1]
    pairs = []
    for i in range(len(x)):
        pairs.append(sorted([x[i],y[i]]))
    pairs = sorted(pairs)
    doublet_to_keep = []
    for i in range(len(pairs)):
        if pairs[i][0] not in doublet_to_keep:
            doublet_to_keep.append(pairs[i][0])
    
    centsred_1 = centsred[np.logical_not(same_pixel),:]
    centsred_2 = centsred[doublet_to_keep,:]
    centsred = np.vstack([centsred_1,centsred_2])
    return centsred


# centsred = points_from_mask_two_thresholds(
#     img_median = postIMSmpre,
#     img_convolved = tmp2,
#     mask_outer = postIMSoutermask,
#     mask_inner = postIMSinnermask,
#     w_outer = w_outer,
#     w_inner = w_inner,
#     threshold_outer = threshold_outer,
#     threshold_inner = threshold_inner,
#     pixelsize = pixelsize,
#     resolution = resolution,
#     stepsize = stepsize,
#     min_n_outer=6,
#     min_n_inner=4
# )
centsred = combine_points(centsred_outer, centsred_inner)
centsred = filter_points(centsred)
del postIMSoutermask, postIMSinnermask
gc.collect()

logging.info("Number of unique IMS pixels detected (both): "+str(centsred.shape[0]))
# plt.scatter(centsred_inner[:,1]*stepsize/resolution,centsred_inner[:,0]*stepsize/resolution)
# plt.scatter(centsred_outer[:,1]*stepsize/resolution,centsred_outer[:,0]*stepsize/resolution)
# plt.imshow(postIMSmpre)
# # plt.scatter(centsred[:,1]*stepsize/resolution,centsred[:,0]*stepsize/resolution, c="blue")
# plt.scatter(centsred2[:,1]*stepsize/resolution,centsred2[:,0]*stepsize/resolution)
# plt.show()
logging.info("IMS bbox:")
# subset region for IMS
imz_bbox = skimage.measure.regionprops((imzregions == regionimz).astype(np.uint8))[0].bbox
del imzregions
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
postIMSpimg = np.zeros(postIMSmpre.shape, dtype=bool)
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
centsrange = np.max(centsred,axis=0)-np.min(centsred,axis=0)
imzsrange = np.array([xmaximz-xminimz,ymaximz-yminimz])
centsmin_adj = np.min(centsred,axis=0)+(centsrange-imzsrange)/2
init_transform = sitk.Euler2DTransform()
init_transform.SetTranslation(centsmin_adj[[1,0]]*10)

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
# plt.imshow(postIMSro_trans.astype(float)+imzimg_regin.astype(float))
del imzimg_regin, postIMSpimg, R, resampler
gc.collect()
# plt.show()

# inverse transformation
tinv = transform.GetInverse()
tinv.SetTranslation(-np.array(tinv.GetTranslation())[[1,0]]/10+1.5)

imzringmask = create_ring_mask(imzimg[xminimz:xmaximz,yminimz:ymaximz], imspixel_outscale, imspixel_inscale+1)
logging.info("Create IMS boundary coordinates")
imzcoords = create_imz_coords(imzimg, imzringmask, imzrefcoords, imz_bbox, rotmat)
del imzringmask

logging.info("Create IMS boundary coordinates")
# init_translation = -np.min(imzcoords,axis=0).astype(int)
init_translation = -np.array([xminimz,yminimz]).astype(int)
imzcoords_in = imzcoords + init_translation
imzcoordsfilttrans = np.array([tinv.TransformPoint(imzcoords_in[i,:].astype(float)) for i in range(imzcoords_in.shape[0])])

tmpfilename = f"{os.path.dirname(snakemake.log['stdout'])}/{os.path.basename(snakemake.log['stdout']).split('.')[0]}_sitk_registration.svg"
plt.close()
plt.scatter(imzcoordsfilttrans[:,1], imzcoordsfilttrans[:,0],color="red",alpha=0.5)
plt.scatter(centsred[:,1], centsred[:,0],color="blue",alpha=0.5)
plt.title("matching points")
fig = plt.gcf()
fig.set_size_inches(20,20)
fig.savefig(tmpfilename)
# plt.show()


logging.info("Create postIMS boundary from observed points")
# create new ringmask based on found points
postIMSn = np.zeros(postIMSmpre.shape)
for i in range(centsred.shape[0]):
    postIMSn[int(centsred[i,0]/resolution*stepsize),int(centsred[i,1]/resolution*stepsize)] = 1
postIMSn = skimage.morphology.convex_hull_image(postIMSn)
postIMSnringmask = create_ring_mask(postIMSn, imspixel_outscale*stepsize/resolution, imspixel_inscale*stepsize/resolution)

logging.info("Extract postIMS boundary points")
ws = [0.001,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
max_score_outer, threshold_outer, w_outer, centsred_outer = find_w(postIMSmpre, tmp2, postIMSnringmask, postIMSnringmask, ws, threads=threads)

logging.info("Find best threshold for points (inner points)")
ws = [0.001,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
max_score_inner, threshold_inner, w_inner, centsred_inner = find_w(postIMSmpre, tmp2, postIMSn, postIMSn, ws, threads=threads)

centsred = combine_points(centsred_outer, centsred_inner)
centsred = filter_points(centsred)

# centsred = points_from_mask_two_thresholds(
#     img_median = postIMSmpre,
#     img_convolved = tmp2,
#     mask_outer = postIMSnringmask,
#     mask_inner = postIMSn,
#     w_outer = w_outer,
#     w_inner = w_inner,
#     threshold_outer = threshold_outer,
#     threshold_inner = threshold_inner,
#     pixelsize = pixelsize,
#     resolution = resolution,
#     stepsize = stepsize,
#     min_n_outer=8,
#     min_n_inner=8
# )
del tmp2
gc.collect()

# postIMSm = postIMSmforpoints.copy()
# postIMSm[np.logical_not(postIMSnringmask)] = 0
# postIMSmb = postIMSm>threshold
# centsred = points_from_mask(postIMSmb, pixelsize=pixelsize, resolution=resolution, stepsize=stepsize)
# plt.scatter(imzcoordsfilttrans[:,0], imzcoordsfilttrans[:,1],color="red")
# plt.scatter(centsred[:,0], centsred[:,1],color="blue")
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
    # angles = angles[centsred_has_match]
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

logging.info("Grid search for fine transformation")
# create polygon to check if centsred points are within polygon

# create polygons of neighboring pixels, then combine to global polygon,
# more exact than covex hull
# kdt = KDTree(imzcoordsfilttrans, leaf_size=30, metric='euclidean')
# imz_distances, indices = kdt.query(imzcoordsfilttrans, k=9, return_distance=True)
# close = imz_distances<np.sqrt(1)+0.01
# tmpch=[]
# for i in range(len(imzcoordsfilttrans)):
#     tmpind = indices[i,:][close[i,:]]
#     tmpch.append(shapely.geometry.MultiPoint(imzcoordsfilttrans[tmpind,:]).convex_hull)
# poly = shapely.unary_union(tmpch)
# del tmpch
# try:
#     poly = shapely.geometry.Polygon(poly.exterior)
#     logging.info("\tUse Grid")
# except:
#     logging.info("\tUse Convex hull")
#     poly = shapely.geometry.MultiPoint(imzcoordsfilttrans).convex_hull

try:
    poly = concave_boundary_from_grid_holes(imzcoordsfilttrans, direction=2)
    if poly.geom_type == "LineString":
        poly = shapely.Polygon(poly)
except Exception as error:
    logging.info(f"concave_boundary_from_grid failed! using alpha hull: {error}")
    logging.info(f"Using alpha hull")
    poly = shapely.concave_hull(shapely.geometry.MultiPoint(imzcoordsfilttrans), ratio=0.01)
poly = poly.buffer(0.15)
logging.info(f"area of polygon: {poly.area/(1000**2):6.5}mm")
# centsred points
tpls = [shapely.geometry.Point(centsred[i,:]) for i in range(centsred.shape[0])]
# only keep points close to the border
poly_small = poly.buffer(-7)
pconts = np.array([poly_small.contains(tpls[i]) for i in range(len(tpls))])
del poly_small
tpls = np.array(tpls)[np.logical_not(pconts)]
centsred_tmp = centsred[np.logical_not(pconts),:]

# weight by location on boundary, i.e. get angle of points to centroid, 
# divided possible space of angles (0-360) into 10 groups, normalize weight of points per group
postIMScent = skimage.measure.regionprops(skimage.measure.label(postIMSnringmask))[0].centroid
postIMScentred = np.array(postIMScent)/stepsize*resolution
angles = np.array([get_angle(centsred_tmp[i,:],postIMScentred) for i in range(np.sum(np.logical_not(pconts)))])
del postIMScent, postIMScentred
gc.collect()
def get_grp_weights(angles,n_groups=10):
    grps = (angles/(360/n_groups)).astype(int)
    np_grps = np.array([np.sum(grps==i) for i in range(n_groups)]).astype(np.double)
    is_nonzero = np_grps!=0
    grp_weights=np_grps*0
    grp_weights[is_nonzero] = 1/np_grps[is_nonzero]
    weights = grps*0.0
    for i in range(n_groups):
        weights[grps==i] = grp_weights[i]
    return weights

angles = angles +180
wls = []
for n_groups in [2,3,4,5,6,7,8,9,10,12]:
    for sh in np.arange(-45,50,5):
        wls.append(get_grp_weights((angles+sh)%360,n_groups))
weights = np.mean(np.stack(wls),axis=0)**(1.5)
weights = weights/np.sum(weights)*len(weights)
# weights=weights/np.max(weights)
# plt.scatter(angles,weights)
# plt.scatter(centsred[np.logical_not(pconts),:][:,1],centsred[np.logical_not(pconts),:][:,0],c=np.sqrt(weights))
# plt.show()


# template transform
tmp_transform = sitk.Euler2DTransform()
tmp_transform.SetCenter(((poly.bounds[2]-poly.bounds[0])/2+poly.bounds[0],(poly.bounds[3]-poly.bounds[1])/2+poly.bounds[1]))

# KDtree for distance calculations
kdt = KDTree(centsred_tmp, leaf_size=30, metric='euclidean')
cents_indices = np.arange(centsred_tmp.shape[0])
del centsred_tmp
imz_indices = np.arange(imzcoordsfilttrans.shape[0])

# def score_init_transform(tz, poly, tpls, imzcoordsfilttrans, cents_indices, imz_indices):
def score_init_transform(tz):
    xsh, ysh, rot = tz
    tmp_transform = sitk.Euler2DTransform()
    tmp_transform.SetCenter(((poly.bounds[2]-poly.bounds[0])/2+poly.bounds[0],(poly.bounds[3]-poly.bounds[1])/2+poly.bounds[1]))
    tmp_transform.SetParameters((rot,xsh,ysh))
    tpoly = shapely.affinity.rotate(poly,rot, use_radians=True)
    tpoly = shapely.affinity.translate(tpoly,xsh,ysh)
    shapely.prepare(tpoly)
    # check if points are in polygon
    does_contain = np.array([tpoly.contains(tpls[i]) for i in range(len(tpls))])
    pconts = np.mean(does_contain)

    if pconts<0.95:
        return [0,0,pconts,999999,999999,xsh,ysh,rot]

    # transform points
    tmpimzrot = np.array([tmp_transform.TransformPoint(imzcoordsfilttrans[i,:]) for i in range(imzcoordsfilttrans.shape[0])])
    # calculate distances and matches 
    imz_distances, match_indices = kdt.query(tmpimzrot, k=1, return_distance=True)
    bool_ind = imz_distances<1
    a=np.stack([
        cents_indices[match_indices[bool_ind]],
        imz_indices[bool_ind[:,0]],
        imz_distances[bool_ind],
        does_contain.astype(int)[match_indices[bool_ind]]]).T
    a = a[a[:, 0].argsort()]
    alsd = np.split(a[:, 2], np.unique(a[:, 0], return_index=True)[1][1:]) 
    bool_ind2 = np.concatenate([alsd[i]==np.min(alsd[i]) for i in range(len(alsd))])
    bool_ind_comb = np.logical_and(bool_ind2, a[:,3]==1)
    matches = a[bool_ind_comb,:2].astype(int)
    weighted_points = np.mean(weights[matches[:,0]])
    mean_dist = np.mean(a[bool_ind_comb,2])
    # weighted_mean_dist = np.sum(a[bool_ind_comb,2]*weights[matches[:,0]])/np.sum(weights[matches[:,0]])/len(a[bool_ind_comb,2])
    weighted_mean_dist = np.sum(a[bool_ind_comb,2]*weights[matches[:,0]])/len(a[bool_ind_comb,2])

    # add metrics
    return [weighted_points, matches.shape[0],pconts, mean_dist,weighted_mean_dist,xsh,ysh,rot]

def init_worker(poly_in, tpls_in, imzcoordsfilttrans_in, cents_indices_in, imz_indices_in, weights_in):
    global poly
    poly = poly_in
    global tpls
    tpls = tpls_in
    global imzcoordsfilttrans
    imzcoordsfilttrans = imzcoordsfilttrans_in
    global cents_indices
    cents_indices = cents_indices_in
    global imz_indices
    imz_indices = imz_indices_in
    global weights
    weights = weights_in

import itertools
xrinit = np.linspace(-3,3,25)
yrinit = np.linspace(-3,3,25)
rotrinit = np.linspace(-np.pi/144,np.pi/144,11)
logging.info(f"\tNumber of iterations: {len(xrinit)*len(yrinit)*len(rotrinit)}")
tz = itertools.product(xrinit, yrinit, rotrinit)
with Pool(threads, initializer= init_worker, initargs=(poly, tpls, imzcoordsfilttrans, cents_indices, imz_indices, weights)) as p:
    n_points_ls = p.map(score_init_transform, tz)

n_points_arr_init = np.array(n_points_ls)
n_points_arr_init = n_points_arr_init[np.isfinite(n_points_arr_init).all(axis=1)]
if np.max(n_points_arr_init[:,2])<0.95:
    pcont_threshold = np.max(n_points_arr_init[:,2])
    n_points_arr_init_red = n_points_arr_init[np.logical_or(n_points_arr_init[:,2] >= pcont_threshold,n_points_arr_init[:,2]==np.max(n_points_arr_init[:,2])),:]
    xrinit2 = np.linspace(-3,3,7)
    yrinit2 = np.linspace(-3,3,7)
    rotrinit2 = np.linspace(-np.pi/144,np.pi/144,7)
else:
    pcont_threshold = np.quantile(n_points_arr_init[:,2],0.9)
    n_points_arr_init_red = n_points_arr_init[np.logical_or(n_points_arr_init[:,2] >= pcont_threshold,n_points_arr_init[:,2]==np.max(n_points_arr_init[:,2])),:]
    xrinit2 = np.arange(np.min(n_points_arr_init_red[:,5])-0.5,np.max(n_points_arr_init_red[:,5])+1,0.1)
    yrinit2 = np.arange(np.min(n_points_arr_init_red[:,6])-0.5,np.max(n_points_arr_init_red[:,6])+1,0.1)
    rotrinit2 = np.arange(np.min(n_points_arr_init_red[:,7])-2*np.pi/1440,np.max(n_points_arr_init_red[:,7])+2*np.pi/1440,np.pi/2880)


import itertools
logging.info(f"\tNumber of iterations: {len(xrinit2)*len(yrinit2)*len(rotrinit2)}")
tz = itertools.product(xrinit2,yrinit2,rotrinit2)
with Pool(threads, initializer= init_worker, initargs=(poly, tpls, imzcoordsfilttrans, cents_indices, imz_indices, weights)) as p:
    n_points_ls = p.map(score_init_transform, tz)

n_points_arr_init2 = np.array(n_points_ls)
n_points_arr_init2 = n_points_arr_init2[np.isfinite(n_points_arr_init2).all(axis=1)]
if np.max(n_points_arr_init2[:,2])<0.95:
    pcont_threshold = np.max(n_points_arr_init2[:,2])
    n_points_arr_init2_red = n_points_arr_init2[np.logical_or(n_points_arr_init2[:,2] >= pcont_threshold,n_points_arr_init2[:,2]==np.max(n_points_arr_init2[:,2])),:]
    xr = np.linspace(-3,3,7)
    yr = np.linspace(-3,3,7)
    rotr = np.linspace(-np.pi/144,np.pi/144,7)
else:
    pcont_threshold = np.quantile(n_points_arr_init2[:,2],1)
    n_points_arr_init2_red = n_points_arr_init2[np.logical_or(n_points_arr_init2[:,2] >= pcont_threshold,n_points_arr_init2[:,2]==np.max(n_points_arr_init2[:,2])),:]
    xr = np.arange(np.min(n_points_arr_init2_red[:,5])-0.1,np.max(n_points_arr_init2_red[:,5])+0.2,0.05)
    yr = np.arange(np.min(n_points_arr_init2_red[:,6])-0.1,np.max(n_points_arr_init2_red[:,6])+0.2,0.05)
    rotr = np.arange(np.min(n_points_arr_init2_red[:,7])-np.pi/2880,np.max(n_points_arr_init2_red[:,7])+2*np.pi/2880,np.pi/5760)

import itertools
logging.info(f"\tNumber of iterations: {len(xr)*len(yr)*len(rotr)}")
tz = itertools.product(xr,yr,rotr)
with Pool(threads, initializer= init_worker, initargs=(poly, tpls, imzcoordsfilttrans, cents_indices, imz_indices, weights)) as p:
    n_points_ls = p.map(score_init_transform, tz)
del poly, tpls
gc.collect()
n_points_arr = np.array(n_points_ls)
n_points_arr = n_points_arr[np.isfinite(n_points_arr).all(axis=1)]
n_points_arr = np.vstack([n_points_arr_init_red,n_points_arr_init2_red, n_points_arr])
logging.info(f"\tNumber of finite points:{n_points_arr.shape[0]}/{len(n_points_ls)}")
if np.max(n_points_arr[:,2])<0.95:
    xsh=0
    ysh=0
    rot=0
else:
    # filter by proportion of points in polygon
    pcont_threshold = np.quantile(n_points_arr[:,2],0.995)
    logging.info(f"\tFilter threshold: {pcont_threshold}")
    logging.info(f"\tCounts: {np.sum(n_points_arr[:,2]>=pcont_threshold)}")
    n_points_arr_red = n_points_arr[np.logical_or(n_points_arr[:,2] >= pcont_threshold,n_points_arr[:,2]==np.max(n_points_arr[:,2])),:]
    logging.info(f"\tNumber of points: {n_points_arr_red.shape[0]}/{n_points_arr.shape[0]}")


    if np.min(n_points_arr_red[:,2])==np.max(n_points_arr_red[:,2]):
        b = np.ones(len(n_points_arr_red[:,2]))
    else:
        b=(n_points_arr_red[:,2]-np.min(n_points_arr_red[:,2]))/(np.max(n_points_arr_red[:,2])-np.min(n_points_arr_red[:,2]))
    if np.min(n_points_arr_red[:,1])==np.max(n_points_arr_red[:,1]):
        d = np.ones(len(n_points_arr_red[:,1]))
    else:
        d=(n_points_arr_red[:,1]-np.min(n_points_arr_red[:,1]))/(np.max(n_points_arr_red[:,1])-np.min(n_points_arr_red[:,1]))
    if np.min(n_points_arr_red[:,4])==np.max(n_points_arr_red[:,4]):
        e = np.ones(len(n_points_arr_red[:,4]))
    else:
        e=1-(n_points_arr_red[:,4]-np.min(n_points_arr_red[:,4]))/(np.max(n_points_arr_red[:,4])-np.min(n_points_arr_red[:,4]))

    # score based on number of points and distances
    # weighted_score = (2*b+4*d+e)/7
    weighted_score = (b+d+e)/3
    # weighted_score = (n_points_arr_red[:,2]-0.99)*100*e
    inds = np.arange(0,len(weighted_score),1)
    # inds = inds[d>0.95]
    # weighted_score = weighted_score[d>0.95]
    ind = inds[weighted_score == np.max(weighted_score)][0]

    # best transformation parameters
    xsh = n_points_arr_red[ind,5]
    ysh = n_points_arr_red[ind,6]
    rot = n_points_arr_red[ind,7]
    del n_points_arr_red

del n_points_ls, n_points_arr
gc.collect()

logging.info(f"Parameters: rotation: {rot}, x shift: {xsh}, yshift: {ysh}")

tmp_transform.SetParameters((rot,xsh,ysh))

tmpimzrot = np.array([tmp_transform.TransformPoint(imzcoordsfilttrans[i,:]) for i in range(imzcoordsfilttrans.shape[0])])

tmpfilename = f"{os.path.dirname(snakemake.log['stdout'])}/{os.path.basename(snakemake.log['stdout']).split('.')[0]}_gridsearch_registration.svg"
plt.close()
plt.scatter(tmpimzrot[:,1], tmpimzrot[:,0],color="red",alpha=0.5)
plt.scatter(centsred[:,1], centsred[:,0],color="blue",alpha=0.5)
plt.title("matching points")
fig = plt.gcf()
fig.set_size_inches(20,20)
fig.savefig(tmpfilename)
# plt.show()


logging.info("Save results")

sitk.WriteTransform(tmp_transform, gridsearch_transform_filename)
sitk.WriteTransform(tinv, masks_transform_filename)

np.savetxt(postIMS_ablation_centroids_filename,centsred,delimiter=',')


postIMS_bbox = (xmin, ymin, xmax, ymax)
metadata_to_save = {
    'IMS_bbox': list(imz_bbox), 
    'postIMS_bbox': list(postIMS_bbox),
    'IMS_regions': [int(p) for p in imzuqregs],
    'Matching_IMS_region': regionimz
    }
with open(metadata_to_save_filename, "w") as fp:
    json.dump(metadata_to_save , fp) 

logging.info("Finished")
