import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
import pandas as pd
import cv2
import SimpleITK as sitk
import napari_imsmicrolink
from ome_types import from_tiff
import skimage
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import shapely
import shapely.affinity
from image_utils import readimage_crop, convert_and_scale_image, saveimage_tile, get_image_shape
from registration_utils import create_imz_coords,get_rotmat_from_angle, concave_boundary_from_grid_holes, get_angle, get_angle_vec, create_ring_mask
from utils import setNThreads, snakeMakeMock
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["IMS_pixelsize"] = 30
    snakemake.params["IMS_shrink_factor"] = 0.8
    snakemake.params["microscopy_pixelsize"] = 0.22537
    snakemake.params["IMS_rotation_angle"] = 180
    snakemake.params["IMS_to_postIMS_n_splits"] = 19
    snakemake.params["IMS_to_postIMS_init_gridsearch"] = 3
    snakemake.input["postIMS_downscaled"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/postIMS/NASH_HCC_TMA_postIMS.ome.tiff"
    snakemake.input["postIMSmask_downscaled"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/postIMS/NASH_HCC_TMA_postIMS_reduced_mask.ome.tiff"
    snakemake.input["imzml"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/IMS/NASH_HCC_TMA_IMS.imzML"
    snakemake.input["IMCmask"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/IMC_mask/NASH_HCC_TMA-2_012_transformed_on_postIMS.ome.tiff"
    snakemake.input["IMS_to_postIMS_matches"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/IMS/NASH_HCC_TMA_NASH_HCC_TMA_IMS_IMS_to_postIMS_matches.csv"

    snakemake.input["masks_transform"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/registration_metric/NASH_HCC_TMA-2_012_masks_transform.txt"
    snakemake.input["gridsearch_transform"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/registration_metric/NASH_HCC_TMA-2_012_gridsearch_transform.txt"
    snakemake.input["postIMS_ablation_centroids"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/registration_metric/NASH_HCC_TMA-2_012_postIMS_ablation_centroids.csv"
    snakemake.input["metadata"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/registration_metric/NASH_HCC_TMA-2_012_step1_metadata.json"


    snakemake.input["sam_weights"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/Misc/sam_vit_h_4b8939.pth"
    snakemake.input["microscopy_image"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/test_split_pre_postIMC.ome.tiff"
    snakemake.input["IMC_location"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_location/test_split_pre_IMC_mask_on_postIMC_A1.geojson"
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)


# params
stepsize = float(snakemake.params["IMS_pixelsize"])
pixelsize = stepsize*float(snakemake.params["IMS_shrink_factor"])
resolution = float(snakemake.params["microscopy_pixelsize"])
rotation_imz = float(snakemake.params["IMS_rotation_angle"])
assert(rotation_imz in [-270,-180,-90,0,90,180,270])
rotmat = get_rotmat_from_angle(rotation_imz)
IMS_to_postIMS_n_splits = snakemake.params["IMS_to_postIMS_n_splits"]
logging.info(f"IMS_to_postIMS_n_splits: {IMS_to_postIMS_n_splits}")
assert(IMS_to_postIMS_n_splits in [3,5,7,9,11,13,15,17,19])
IMS_to_postIMS_init_gridsearch = snakemake.params["IMS_to_postIMS_init_gridsearch"]
logging.info(f"IMS_to_postIMS_init_gridsearch: {IMS_to_postIMS_init_gridsearch}")
assert(IMS_to_postIMS_init_gridsearch in [0,1,2,3])
postIMSmask_extraction_constraint = snakemake.params["postIMSmask_extraction_constraint"]

imspixel_inscale = 4
if postIMSmask_extraction_constraint == "preIMS":
    imspixel_outscale = 8
else:
    imspixel_outscale = 2

threads = snakemake.threads

# inputs
postIMS_file = snakemake.input["postIMS_downscaled"]
postIMSr_file = snakemake.input["postIMSmask_downscaled"]
imzmlfile = snakemake.input["imzml"]
output_table = snakemake.input["IMS_to_postIMS_matches"]

# outputs
masks_transform_filename = snakemake.output["masks_transform"]
gridsearch_transform_filename = snakemake.output["gridsearch_transform"]

postIMS_ablation_centroids_filename = snakemake.output["postIMS_ablation_centroids"]
metadata_to_save_filename = snakemake.output["metadata"]


postIMS_ome = from_tiff(postIMS_file)
postIMS_resolution = postIMS_ome.images[0].pixels.physical_size_x
logging.info(f"postIMS resolution: {postIMS_resolution}")
assert postIMS_resolution == resolution

postIMSr_ome = from_tiff(postIMSr_file)
postIMSr_resolution = postIMSr_ome.images[0].pixels.physical_size_x
logging.info(f"postIMS mask resolution: {postIMSr_resolution}")

rescale = postIMSr_resolution/resolution
logging.info(f"rescale: {rescale}")



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
logging.info(f"imz regions: {imzregions.shape}")
logging.info(f"imz regions: {imzuqregs}")

# reference coordinates, actually in data
imzrefcoords = np.stack([imz.y_coords_min,imz.x_coords_min],axis=1)
del imz


logging.info("Read postIMS region bounding box")
# read crop bbox
dfmeta = pd.read_csv(output_table)
logging.info(f"{dfmeta}")
imc_samplename = os.path.splitext(os.path.splitext(os.path.split(metadata_to_save_filename)[1])[0])[0].replace("_step1_metadata","")
imc_project = os.path.split(os.path.split(os.path.split(os.path.split(metadata_to_save_filename)[0])[0])[0])[1]
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
xmin = xmin*rescale
xmin = 0 if xmin<0 else xmin
ymin = dfmeta[inds_arr]["postIMS_ymin"].tolist()[0]-int(imspixel_outscale*stepsize)
ymin = ymin*rescale
ymin = 0 if ymin<0 else ymin
xmax = dfmeta[inds_arr]["postIMS_xmax"].tolist()[0]+int(imspixel_outscale*stepsize)
xmax = xmax*rescale
xmax = img_shape[0] if xmax>img_shape[0] else xmax
ymax = dfmeta[inds_arr]["postIMS_ymax"].tolist()[0]+int(imspixel_outscale*stepsize)
ymax = ymax*rescale
ymax = img_shape[1] if ymax>img_shape[1] else ymax
logging.info(f"Bounding box postIMS:")
logging.info(f"\txmin: {xmin}")
logging.info(f"\txmax: {xmax}")
logging.info(f"\tymin: {ymin}")
logging.info(f"\tymax: {ymax}")

xmin_mask = xmin/rescale
ymin_mask = ymin/rescale
xmax_mask = xmax/rescale
ymax_mask = ymax/rescale
logging.info(f"Bounding box postIMS mask:")
logging.info(f"\txmin: {xmin_mask}")
logging.info(f"\txmax: {xmax_mask}")
logging.info(f"\tymin: {ymin_mask}")
logging.info(f"\tymax: {ymax_mask}")

# needed:
regionimz = dfmeta[inds_arr]["imzregion"].tolist()[0]
logging.info(f"imz region: {regionimz}")


logging.info("Read cropped postIMS")
# subset mask
postIMSmpre = readimage_crop(postIMS_file, [int(xmin), int(ymin), int(xmax), int(ymax)])
postIMSmpre = convert_and_scale_image(postIMSmpre, 1)
logging.info("Median filter")
ksize = np.round(((stepsize-pixelsize)/resolution)/3).astype(int)*2
ksize = ksize+1 if ksize%2==0 else ksize
cv2.medianBlur(src=postIMSmpre,dst=postIMSmpre, ksize=ksize)

logging.info("Read cropped postIMS mask")
postIMSrcut = readimage_crop(postIMSr_file, [int(xmin_mask), int(ymin_mask), int(xmax_mask), int(ymax_mask)])
if rescale != 1:
    logging.info(f"Resize postIMS mask.info")
    logging.info(f"\tshape before: {postIMSrcut.shape}")
    postIMSrcut = cv2.resize(postIMSrcut, (postIMSmpre.shape[1],postIMSmpre.shape[0]), interpolation=cv2.INTER_NEAREST)
    logging.info(f"\tshape after: {postIMSrcut.shape}")
# to np.uint8
cv2.convertScaleAbs(src=postIMSrcut, dst=postIMSrcut)
logging.info("Create ringmask")
postIMSringmask = create_ring_mask(postIMSrcut, (1/resolution)*stepsize*imspixel_outscale, (1/resolution)*stepsize*imspixel_inscale)
logging.info("Isotropic dilation")
postIMSoutermask = cv2.morphologyEx(src=postIMSrcut, op = cv2.MORPH_DILATE, kernel = skimage.morphology.square(2*int((1/resolution)*stepsize*imspixel_outscale))).astype(bool)
postIMSoutermask_small = cv2.morphologyEx(src=postIMSrcut, op = cv2.MORPH_DILATE, kernel = skimage.morphology.square(2*int((1/resolution)*stepsize))).astype(bool)
postIMSinnermask = cv2.morphologyEx(src=postIMSrcut, op = cv2.MORPH_ERODE, kernel = skimage.morphology.square(2*int((1/resolution)*stepsize*imspixel_inscale))).astype(bool)
del postIMSrcut

# subset and filter postIMS image
kersize = int(stepsize/2)
kersize = kersize-1 if kersize%2==0 else kersize
kernel = np.zeros((kersize,kersize))
kernel[int((kersize-1)/2),:]=1
kernel[:,int((kersize-1)/2)]=1
logging.info(f"Disk radius for rank threshold filter: {kersize}")

# resize to 1um spacing
if resolution != 1:
    wn = int(postIMSmpre.shape[0]*resolution)
    hn = int(postIMSmpre.shape[1]*resolution)
    tmp2 = cv2.resize(postIMSmpre, (hn,wn), interpolation=cv2.INTER_NEAREST)
else:
    tmp2 = postIMSmpre.copy()

# local rank filter, results in binary image
tmp2 = skimage.filters.rank.threshold(tmp2, skimage.morphology.disk(kersize))
cv2.normalize(tmp2, tmp2, 0, 255, cv2.NORM_MINMAX)

logging.info("Mean filter")
# mean filter, with cross shape footprint
cv2.filter2D(src = tmp2, dst = tmp2, ddepth=-1, kernel=kernel/np.sum(kernel))
    
if resolution != 1:
    tmp2 = cv2.resize(tmp2, (postIMSmpre.shape[1],postIMSmpre.shape[0]), interpolation=cv2.INTER_NEAREST)


def normalize_scores(arr):
    return 1 if np.max(arr) == 0 else np.max(arr)


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
    _, _, stats, cents = cv2.connectedComponentsWithStatsWithAlgorithm(mask.astype(np.uint8), connectivity=4, ltype=cv2.CV_32S, ccltype=cv2.CCL_SAUF)
    stats=stats[1:,:]
    cents=np.flip(cents[1:,:],axis=1)
    
    # filter by area 
    # areas = stats[:,4]
    # widths = stats[:,2]
    # heights = stats[:,3]

    # minimal area is a circle with radius (empirical guess based on pixelsize and resolution) 
    min_radius = (pixelsize/8+(pixelsize/8-1)*2.5)/resolution
    min_area = min_radius**2*np.pi if min_radius>0 else np.pi
    area_range = [min_area,(pixelsize/resolution)**2]
    inran = np.logical_and(stats[:,4] > area_range[0], stats[:,4] < area_range[1])

    cents = cents[inran,:]
    # check length
    if not isinstance(cents, np.ndarray) or len(cents)<6:
        return np.zeros((0,2))

    # filter by ratio of x-slice to y-slice
    slice_ratio = stats[:,2][inran]/stats[:,3][inran]
    is_round = np.abs(np.log10(slice_ratio))<np.log10(2)
    cents = cents[is_round,:]

    # check length
    if not isinstance(cents, np.ndarray) or len(cents)<6:
        return np.zeros((0,2))
    # to IMS scale
    centsred = cents*resolution/stepsize
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
    to_keep_dist = np.logical_and(distances[:,1:]>0.93, distances[:,1:] < 1.07)

    # restrict angles to nearest neighbors to certain ranges
    diffs = points[indices[:, 1:], :] - points[:, None, :]
    absangles = np.abs(get_angle_vec(diffs,[0,0],[1,0]))
    to_keep_angle = np.logical_or(
            np.logical_or(absangles < 9, absangles > 171),
            np.logical_and(absangles > 81, absangles < 99),
    )
    to_keep = np.logical_and(to_keep_angle, to_keep_dist)
    some_neighbors = np.sum(to_keep, axis=1)>0
    
    # create adjaceny matrix
    indices_sub = indices[some_neighbors,:]
    to_keep_sub = to_keep[some_neighbors,:]
    # Get the indices where to_keep_sub is True
    i, j = np.where(to_keep_sub)
    # Get the corresponding values from indices_sub
    rows = indices_sub[np.arange(len(indices_sub)), 0][i]
    cols = indices_sub[i, j+1]
    # Create the adjacency matrix
    adjmat = csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(to_keep.shape[0], to_keep.shape[0]))    
    # Make the matrix symmetric
    adjmat = adjmat + adjmat.T

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
            nt1, bt1, wt1, centsredls1 = zip(*p.map(scorer, thresholds))
    else:
        nt1, bt1, wt1, centsredls1 = zip(*map(scorer, thresholds))

    nt1=np.array(nt1)
    bt1=np.array(bt1)
    wt1=np.array(wt1)
    scores1 = 2*wt1/normalize_scores(wt1) + nt1/normalize_scores(nt1) + 4*bt1/normalize_scores(bt1)
    max_score=np.max(scores1)
    max_points = np.max(nt1)
    max_border_points = np.max(bt1)
    max_weighted_dist = np.max(wt1)

    threshold = np.asarray(thresholds)[scores1 == max_score][0]

    # finer steps
    thresholds = list(range(threshold-9,threshold+10,3))
    if threads>1:
        with Pool(threads) as p:
            nt2, bt2, wt2, centsredls2 = zip(*p.map(scorer, thresholds))
    else:
        nt2, bt2, wt2, centsredls2 = zip(*map(scorer, thresholds))

    nt2=np.array(nt2)
    bt2=np.array(bt2)
    wt2=np.array(wt2)
    scores2 = 2*wt2/normalize_scores(wt2) + nt2/normalize_scores(nt2) + 4*bt2/normalize_scores(bt2)
    max_score=np.max(scores2)
    max_points = np.max(nt2)
    max_border_points = np.max(bt2)
    max_weighted_dist = np.max(wt2)

    threshold = np.asarray(thresholds)[scores2 == max_score][0]

    # fine steps
    thresholds = list(range(threshold-2,threshold+3))
    if threads>1:
        with Pool(threads) as p:
            nt3, bt3, wt3, centsredls3 = zip(*p.map(scorer, thresholds))
    else:
        nt3, bt3, wt3, centsredls3 = zip(*map(scorer, thresholds))

    nt3=np.array(nt3)
    bt3=np.array(bt3)
    wt3=np.array(wt3)
    scores3 = 2*wt3/normalize_scores(wt3) + nt3/normalize_scores(nt3) + 4*bt3/normalize_scores(bt3)
    max_score=np.max(scores3)
    max_points = np.max(nt3)
    max_border_points = np.max(bt3)
    max_weighted_dist = np.max(wt3)

    threshold = np.asarray(thresholds)[scores3 == max_score][0]

    wt=np.concatenate([wt1,wt2,wt3])
    nt=np.concatenate([nt1,nt2,nt3])
    bt=np.concatenate([bt1,bt2,bt3])
    scores = 2*wt/normalize_scores(wt) + nt/normalize_scores(nt) + 4*bt/normalize_scores(bt)
    max_score=np.max(scores)

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
    postIMSm = cv2.createCLAHE().apply(postIMSm.astype(np.uint8))
    postIMSm[np.logical_not(mask2_in)] = 0
    threshold, max_points, max_border_points, max_weighted_dist, centsred= find_threshold(postIMSm, maskb_dist_in, thr_range=[127,250], threads=1)
    logging.info(f"weight: {w}, threshold: {threshold}, n_points: {max_points}, border points: {max_border_points}, sum of inverse distances {max_weighted_dist}")
    return (threshold, max_points, max_border_points, max_weighted_dist, w, centsred)

def init_worker(img_median, img_convolved, maskb_dist, mask2, threads):
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


logging.info("Find best threshold for points (outer points)")
def find_w(img_median: np.ndarray, img_convolved: np.ndarray, mask1: np.ndarray, mask2: np.ndarray, ws, threads:int=1):

    # Compute the distance transforms
    maskb_dist = cv2.distanceTransform(mask1.astype(np.uint8), cv2.DIST_L2, 3) - cv2.distanceTransform((~mask1).astype(np.uint8), cv2.DIST_L2, 3)

    with Pool(threads, initializer= init_worker, initargs=(img_median, img_convolved, maskb_dist, mask2, threads)) as p:
        thresholds, nt, bt, wt, wsobs, centsredls = zip(*p.map(score_find_w, ws))

    wt = np.array(wt)
    nt = np.array(nt)
    bt = np.array(bt)
    scores = 2*wt/normalize_scores(wt) + nt/normalize_scores(nt) + 4*bt/normalize_scores(bt)
    max_score=np.max(scores)
    threshold = np.asarray(thresholds)[scores == max_score][0]
    w = np.asarray(wsobs)[scores == max_score][0]

    scoresred = [scores[i] for i in range(len(scores)) if centsredls[i].shape[0]>0]
    centsredls = [c for c in centsredls if c.shape[0]>0]
    inds = np.arange(len(scoresred))[np.flip(np.argsort(scoresred))]
    centsredlssort = [centsredls[i] for i in inds]

    from functools import reduce
    if len(centsredlssort)>0:
        if len(centsredlssort[0])>0:
            centsred = reduce(combine_points, centsredlssort)
            centsred = filter_points(centsred)
        else:
            centsred = centsredlssort[0]
    else:
        centsred = np.zeros((0,0))

    return max_score, threshold, w, centsred

ws = np.array([0.001,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99])
wsindx = np.round(np.linspace(0,len(ws)-1,IMS_to_postIMS_n_splits)).astype(int)
ws = ws[wsindx]

max_score_outer, threshold_outer, w_outer, centsred_outer = find_w(postIMSmpre, tmp2, postIMSoutermask_small, postIMSringmask, ws, threads=threads)


def plot_plotly_scatter_image(img, pts):
    import plotly.graph_objects as go
    from PIL import Image
    import base64
    from io import BytesIO

    pil_img = Image.fromarray(img) # PIL image object
    prefix = "data:image/png;base64,"
    with BytesIO() as stream:
        pil_img.save(stream, format="png")
        base64_string = prefix + base64.b64encode(stream.getvalue()).decode("utf-8")
    fig = px.scatter(x=pts[:,1], y=-pts[:,0])
    # Add images
    fig = fig.add_layout_image(
            dict(
                source=base64_string,
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=img.shape[1],
                sizey=img.shape[0],
                sizing="stretch",
                opacity=0.5,
                layer="below")
    )
    return fig

# fig = plot_plotly_scatter_image(postIMSmpre, centsred_outer*stepsize/resolution)
# fig.show()

del postIMSringmask, postIMSoutermask_small

logging.info("Find best threshold for points (inner points)")
max_score_inner, threshold_inner, w_inner, centsred_inner = find_w(postIMSmpre, tmp2, postIMSinnermask, postIMSinnermask, ws, threads=threads)

# fig = plot_plotly_scatter_image(tmp2, centsred_inner*stepsize/resolution)
# fig.show()

logging.info("Max score (outer): "+str(max_score_outer))
logging.info("Corresponding threshold (outer): "+str(threshold_outer))
logging.info("Corresponding weight (outer): "+str(w_outer))
logging.info("Max score (inner): "+str(max_score_inner))
logging.info("Corresponding threshold (inner): "+str(threshold_inner))
logging.info("Corresponding weight (inner): "+str(w_inner))

assert centsred_outer.shape[0] > 0 or centsred_inner.shape[0] > 0
if centsred_outer.shape[0] == 0:
    logging.info(f"No outer points detected, rerun with new mask from convex hull of inner points")
    import skimage.morphology

    centsred_inner_scaled = centsred_inner*stepsize/resolution
    cropped_postIMSrcut = np.zeros(postIMSmpre.shape, dtype=bool)
    for i in range(centsred_inner_scaled.shape[0]):
        cropped_postIMSrcut[int(centsred_inner_scaled[i,0]),int(centsred_inner_scaled[i,1])]=True
    cropped_postIMSrcut = skimage.morphology.convex_hull_image(cropped_postIMSrcut)
    new_postIMSringmask = create_ring_mask(cropped_postIMSrcut, (1/resolution)*stepsize*imspixel_outscale, (1/resolution)*stepsize*imspixel_inscale)
    new_postIMSoutermask_small = cv2.morphologyEx(src=cropped_postIMSrcut.astype(np.uint8)*255, op = cv2.MORPH_DILATE, kernel = skimage.morphology.square(2*int((1/resolution)*stepsize))).astype(bool)
    max_score_outer, threshold_outer, w_outer, centsred_outer = find_w(postIMSmpre, tmp2, new_postIMSoutermask_small, new_postIMSringmask, ws, threads=threads)
    del new_postIMSringmask, new_postIMSoutermask_small, cropped_postIMSrcut

    if centsred_outer.shape[0] == 0:
        logging.info("No outer points detected, using inner points")
        centsred = centsred_inner
    else:
        centsred = combine_points(centsred_outer, centsred_inner)
elif centsred_inner.shape[0] == 0:
    centsred = centsred_outer
else:
    centsred = combine_points(centsred_outer, centsred_inner)
centsred = filter_points(centsred)

# fig = plot_plotly_scatter_image(tmp2, centsred*stepsize/resolution)
# fig.show()
del postIMSoutermask, postIMSinnermask

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

logging.info("Prepare for initial registration with sitk")
logging.info("Prepare postIMS image")
postIMSpimg = np.zeros(postIMSmpre.shape, dtype=np.uint8)
stepsize_px = int(stepsize/resolution)
stepsize_px_half = int(stepsize_px/2)

# Calculate x and y coordinates
x_coords = (centsred[:, 0] * stepsize_px).astype(int)
y_coords = (centsred[:, 1] * stepsize_px).astype(int)

# Calculate ranges
x_ranges = np.clip([x_coords - stepsize_px_half, x_coords + stepsize_px_half], 0, postIMSpimg.shape[0]-1)
y_ranges = np.clip([y_coords - stepsize_px_half, y_coords + stepsize_px_half], 0, postIMSpimg.shape[1]-1)

# Fill the array
for xr, yr in zip(x_ranges.T, y_ranges.T):
    postIMSpimg[xr[0]:(xr[1]+1), yr[0]:(yr[1]+1)] = 255

logging.info(f"    - {np.sum(postIMSpimg)/255/np.prod(postIMSpimg.shape)} area proportion")

logging.info("    - Closing")
cv2.morphologyEx(src=postIMSpimg, dst=postIMSpimg, op = cv2.MORPH_CLOSE, kernel = skimage.morphology.square(int(stepsize/resolution/4))).astype(bool)

logging.info(f"    - {np.sum(postIMSpimg)/np.prod(postIMSpimg.shape)} area proportion")

logging.info("    - Remove small holes")
postIMSpimg = skimage.morphology.remove_small_holes(postIMSpimg, int((stepsize/resolution*10)**2))
logging.info(f"    - {np.sum(postIMSpimg)/np.prod(postIMSpimg.shape)} area proportion")
logging.info("    - Rescale")
wn = int(postIMSpimg.shape[0]*(1/(stepsize/resolution)*10))
hn = int(postIMSpimg.shape[1]*(1/(stepsize/resolution)*10))
postIMSpimg = cv2.resize(postIMSpimg.astype(np.uint8), (hn,wn), interpolation=cv2.INTER_NEAREST)
postIMSpimg = postIMSpimg.astype(float)*255


logging.info("Prepare IMS image")
imzimg_regin = imzimg[xminimz:xmaximz,yminimz:ymaximz]
logging.info("    - Rescale")
wn = int(imzimg_regin.shape[0]*10)
hn = int(imzimg_regin.shape[1]*10)
imzimg_regin = cv2.resize(imzimg_regin.astype(bool).astype(np.uint8), (hn,wn), interpolation=cv2.INTER_NEAREST)
imzimg_regin = imzimg_regin.astype(float)*255

boundary_buffer = 20
logging.info(f"    - Pad with {boundary_buffer} pixels")
postIMSpimg = np.pad(postIMSpimg, boundary_buffer, mode='constant', constant_values=0)
imzimg_regin = np.pad(imzimg_regin, boundary_buffer, mode='constant', constant_values=0)


logging.info("Convert images to sitk images")
fixed = sitk.GetImageFromArray(imzimg_regin)
moving = sitk.GetImageFromArray(postIMSpimg)

def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.8f} "
        + f": {method.GetOptimizerPosition()}"
    )

def init_mask_registration(fixed, moving, yx_translation):
    init_transform = sitk.Euler2DTransform()
    init_transform.SetTranslation(yx_translation)

    logging.info("Setup registration")
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetMetricSamplingStrategy(R.NONE)
    R.SetInterpolator(sitk.sitkNearestNeighbor)
    R.SetOptimizerAsGradientDescent(
        learningRate=1000, numberOfIterations=1000, 
        convergenceMinimumValue=1e-9, convergenceWindowSize=30,
        estimateLearningRate=R.EachIteration
    )
    R.SetOptimizerScalesFromIndexShift()
    R.SetInitialTransform(init_transform)

    # setup sitk images
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    # run registration
    transform = R.Execute(fixed, moving)
    return transform, R.GetMetricValue()


# initial transformation
centsrange = np.max(centsred,axis=0)-np.min(centsred,axis=0)
imzsrange = np.array([xmaximz-xminimz,ymaximz-yminimz])
if np.any(np.abs(centsrange-imzsrange)>imzsrange*0.2):
    # if the range of the points is not similar to the range of the IMS image, shift the points
    shifted_axis = np.array([0,1])[np.abs(centsrange-imzsrange)>imzsrange*0.2]
    shifts_all = []
    for s in shifted_axis:
        shifts = np.ones((3,2))*0.5
        shifts[:,s] = np.array([0,0.5,1])
        shifts_all.append(shifts)
    shifts = np.vstack(shifts_all)
    centsmin_adj_ls = [ np.min(centsred,axis=0)+(centsrange-imzsrange)*s for s in shifts ]

    logging.info("Run registrations")
    trls = []
    metls = []
    for xytr in centsmin_adj_ls:
        tr,met = init_mask_registration(fixed, moving, xytr[[1,0]]*10)
        trls.append(tr)
        metls.append(met)
    trls = np.array(trls)
    metls = np.array(metls)
    transform = trls[np.argmin(metls)]
else:
    centsmin_adj = np.min(centsred,axis=0)+(centsrange-imzsrange)/2

    logging.info("Run registration")
    transform,_ = init_mask_registration(fixed, moving, centsmin_adj[[1,0]]*10)


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
del imzimg_regin, postIMSpimg, resampler
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
postIMSn = np.zeros(postIMSmpre.shape, dtype=np.uint8)
hull = cv2.convexHull((np.flip(centsred, axis=1) / resolution * stepsize).astype(int).reshape(-1, 1, 2))
postIMSn = cv2.drawContours(postIMSn, [hull], 0, (1), thickness=cv2.FILLED)
postIMSn = postIMSn.astype(bool)
postIMSnringmask = create_ring_mask(postIMSn, imspixel_outscale*stepsize/resolution, imspixel_inscale*stepsize/resolution)

logging.info("Extract postIMS boundary points")
max_score_outer, threshold_outer, w_outer, centsred_outer = find_w(postIMSmpre, tmp2, postIMSnringmask, postIMSnringmask, ws, threads=threads)

logging.info("Find best threshold for points (inner points)")
max_score_inner, threshold_inner, w_inner, centsred_inner = find_w(postIMSmpre, tmp2, postIMSn, postIMSn, ws, threads=threads)

centsred = combine_points(centsred_outer, centsred_inner)
centsred = filter_points(centsred)

# fig = plot_plotly_scatter_image(postIMSmpre, centsred*stepsize/resolution)
# fig.show()
del tmp2

logging.info("Grid search for fine transformation")
# create polygon to check if centsred points are within polygon
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
    if np.sum(bool_ind)<3:
        return [0,0,pconts,999999,999999,xsh,ysh,rot]
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

if IMS_to_postIMS_init_gridsearch==0:
    n_points_arr = np.zeros((1,5))
else:
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


if IMS_to_postIMS_init_gridsearch==1:
    n_points_arr = n_points_arr_init
elif IMS_to_postIMS_init_gridsearch>1:
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
    
if IMS_to_postIMS_init_gridsearch==2:
    n_points_arr = np.vstack([n_points_arr_init_red,n_points_arr_init2_red])
elif IMS_to_postIMS_init_gridsearch>2:
    import itertools
    logging.info(f"\tNumber of iterations: {len(xr)*len(yr)*len(rotr)}")
    tz = itertools.product(xr,yr,rotr)
    with Pool(threads, initializer= init_worker, initargs=(poly, tpls, imzcoordsfilttrans, cents_indices, imz_indices, weights)) as p:
        n_points_ls = p.map(score_init_transform, tz)
    del poly, tpls
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
    weighted_score = (b+d+e)/3
    inds = np.arange(0,len(weighted_score),1)
    ind = inds[weighted_score == np.max(weighted_score)][0]

    # best transformation parameters
    xsh = n_points_arr_red[ind,5]
    ysh = n_points_arr_red[ind,6]
    rot = n_points_arr_red[ind,7]
    del n_points_arr_red

del n_points_arr

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
    'Matching_IMS_region': regionimz,
    'rescale': 1,
    'resolution': resolution,
    }
with open(metadata_to_save_filename, "w") as fp:
    json.dump(metadata_to_save , fp) 

logging.info("Finished")
