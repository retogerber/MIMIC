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
from sklearn.neighbors import KDTree
import shapely
import shapely.affinity
from pathlib import Path
from image_utils import readimage_crop, saveimage_tile
from registration_utils import composite2affine, create_imz_coords,get_rotmat_from_angle,  concave_boundary_from_grid_holes, indices_sequence_from_ordered_points, angle_code_from_point_sequence, image_from_points, get_sigma, get_angle, get_angle_vec
from utils import setNThreads, snakeMakeMock
import sys,os
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["IMS_pixelsize"] = 30
    snakemake.params["IMS_shrink_factor"] = 0.8
    snakemake.params["IMC_pixelsize"] = 0.22537
    snakemake.params["IMS_rotation_angle"] = 180
    snakemake.params['within_IMC_fine_registration'] = True
    snakemake.params["min_index_length"] = 6
    snakemake.params["max_index_length"] = 30
    snakemake.input["imzml"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/IMS/NASH_HCC_TMA_IMS.imzML"
    snakemake.input["IMCmask"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/IMC_mask/NASH_HCC_TMA-2_012_transformed_on_postIMS.ome.tiff"
    snakemake.input["postIMSmask_downscaled"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/postIMS/NASH_HCC_TMA_postIMS.ome.tiff"
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
resolution = float(snakemake.params["IMC_pixelsize"])
rotation_imz = float(snakemake.params["IMS_rotation_angle"])
assert(rotation_imz in [-270,-180,-90,0,90,180,270])
rotmat = get_rotmat_from_angle(rotation_imz)
if not isinstance(snakemake.params['within_IMC_fine_registration'], bool):
    assert(isinstance(snakemake.params['within_IMC_fine_registration'], str))
    assert(snakemake.params['within_IMC_fine_registration'].strip() in ["","True","true","False","false"])
    do_within_IMC_fine_registration = True if snakemake.params['within_IMC_fine_registration'] in ["True","true"] else False
else:
    do_within_IMC_fine_registration = snakemake.params['within_IMC_fine_registration']
min_n = snakemake.params["min_index_length"]
max_n = snakemake.params["max_index_length"]

# inputs
imzmlfile = snakemake.input["imzml"]
imc_mask_file = snakemake.input["IMCmask"]
imc_samplename = os.path.splitext(os.path.splitext(os.path.split(imc_mask_file)[1])[0])[0].replace("_transformed_on_postIMS","")
imc_project = os.path.split(os.path.split(os.path.split(os.path.split(imc_mask_file)[0])[0])[0])[1]
project_name = "postIMS_to_IMS_"+imc_project+"-"+imc_samplename
postIMS_file = snakemake.input["postIMS_downscaled"]
masks_transform_filename = snakemake.input["masks_transform"]
gridsearch_transform_filename = snakemake.input["gridsearch_transform"]
postIMS_ablation_centroids_filename = snakemake.input["postIMS_ablation_centroids"]
metadata_to_save_filename = snakemake.input["metadata"]

# outputs
ims_to_postIMS_regerror = snakemake.output["IMS_to_postIMS_error"]
ims_to_postIMS_regerror_image = snakemake.output["IMS_to_postIMS_error_image"]

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
postIMS_shape = (int(postIMS_bbox[2])-int(postIMS_bbox[0]), int(postIMS_bbox[3])-int(postIMS_bbox[1]))

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
del imz

centsred = np.loadtxt(postIMS_ablation_centroids_filename, delimiter=',')

def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.8f} "
        + f": {method.GetOptimizerPosition()}"
    )


########### sitk registration of points
### Get points
logging.info("Read and prepare points")
IMSoutermask = skimage.morphology.isotropic_dilation(imzimg[xminimz:xmaximz,yminimz:ymaximz], (1/resolution)*stepsize*imspixel_outscale)
imzcoords = create_imz_coords(imzimg, IMSoutermask, imzrefcoords, imz_bbox, rotmat)
init_translation = -np.array([xminimz,yminimz]).astype(int)
imzcoords_in = imzcoords + init_translation
imzcoordsfilttrans = np.array([tinv.TransformPoint(imzcoords_in[i,:].astype(float)) for i in range(imzcoords_in.shape[0])])
tmpimzrot = np.array([tmp_transform.TransformPoint(imzcoordsfilttrans[i,:]) for i in range(imzcoordsfilttrans.shape[0])])


logging.info("Create polygons")
tmpIMSpoly = concave_boundary_from_grid_holes(tmpimzrot, direction=2)
if tmpIMSpoly.geom_type == "LineString":
    tmpIMSpoly = shapely.Polygon(tmpIMSpoly)
IMSpoly = tmpIMSpoly.buffer(1.5, cap_style='square', join_style='mitre')
IMSpoly_small = tmpIMSpoly.buffer(-1.5, cap_style='square', join_style='mitre')
if tmpIMSpoly.geom_type == "Polygon":
    ordered_imz_border_all = np.array(tmpIMSpoly.exterior.coords.xy).T[:-1,:]
else:
    ordered_imz_border_all = np.array(tmpIMSpoly.xy).T

try:
    tch1 = concave_boundary_from_grid_holes(centsred)
    tch1 = shapely.Polygon(np.array(tch1.exterior.coords.xy).T)
    gc1 = shapely.make_valid(tch1)
    if type(gc1) is shapely.geometry.collection.GeometryCollection:
        n = shapely.get_num_geometries(gc1)
        gs = [shapely.get_geometry(gc1,i) for i in range(n)]
        ts = [type(gs[i]) is shapely.geometry.polygon.Polygon for i in range(len(gs))]
        if np.sum(ts)==1:
            tch11 = gs[np.array(ts).argmax()]
        else:
            tch11=tch1
    elif type(gc1) is shapely.geometry.polygon.Polygon:
        tch11 = gc1
    else:
        tch11 = tch1

    tch2 = concave_boundary_from_grid_holes(centsred,direction=2)
    tch2 = shapely.Polygon(np.array(tch2.exterior.coords.xy).T)
    gc2 = shapely.make_valid(tch2)
    if type(gc2) is shapely.geometry.collection.GeometryCollection:
        n = shapely.get_num_geometries(gc2)
        gs = [shapely.get_geometry(gc2,i) for i in range(n)]
        ts = [type(gs[i]) is shapely.geometry.polygon.Polygon for i in range(len(gs))]
        if np.sum(ts)==1:
            tch22 = gs[np.array(ts).argmax()]
        else:
            tch22=tch2
    elif type(gc2) is shapely.geometry.polygon.Polygon:
        tch22 = gc2
    else:
        tch22 = tch2

    tch = shapely.union(tch11,tch22)
    if not tch.is_valid:
        tch = tch11

    postIMSpoly_outer = tch.buffer(0.5, cap_style='square', join_style='mitre')
    postIMSpoly_inner = tch.buffer(-0.5, cap_style='square', join_style='mitre')
    if tch.geom_type == "Polygon":
        ordered_centsred_border_all = np.array(tch.exterior.coords.xy).T[:-1,:]
    else:
        ordered_centsred_border_all = np.array(tch.xy).T[:-1,:]
except Exception as error:
    logging.info(f"Error in concave_boundary_from_grid_holes: {error}")
    postIMSpoly_outer = shapely.concave_hull(shapely.geometry.MultiPoint(centsred), ratio=0.01).buffer(0.5, cap_style='square', join_style='mitre')
    postIMSpoly_inner = shapely.concave_hull(shapely.geometry.MultiPoint(centsred), ratio=0.01).buffer(-0.5, cap_style='square', join_style='mitre')
    postIMSpoly = shapely.concave_hull(shapely.geometry.MultiPoint(centsred), ratio=0.01)
    ordered_centsred_border_all = np.array(postIMSpoly.exterior.coords.xy).T[:-1,:]

kdt_ordered_imz_border_all = KDTree(ordered_imz_border_all, leaf_size=30, metric='euclidean')
kdt_tmpimzrot = KDTree(tmpimzrot, leaf_size=30, metric='euclidean')

logging.info(f"shape boundary points: {ordered_centsred_border_all.shape}")
logging.info("Filter points at boundary of polygon")
tpls_all = [shapely.geometry.Point(centsred[i,:]) for i in range(centsred.shape[0])]
pconts1 = np.array([postIMSpoly_outer.contains(tpls_all[i]) for i in range(len(tpls_all))])
logging.info(f"pconts1 n nan: {np.unique(np.isnan(pconts1), return_counts=True)}")
pconts1[np.isnan(pconts1)] = False
pconts2 = np.array([postIMSpoly_inner.contains(tpls_all[i]) for i in range(len(tpls_all))])
logging.info(f"pconts2 n nan: {np.unique(np.isnan(pconts2), return_counts=True)}")
pconts2[np.isnan(pconts2)] = False
centsred_border_all = centsred[np.logical_and(pconts1,~pconts2)]
inds_centsred_border_all = np.arange(centsred.shape[0])[np.logical_and(pconts1,~pconts2)]
tpls = [shapely.geometry.Point(centsred[i,:]) for i in range(centsred.shape[0])]
pconts3 = np.array([IMSpoly.contains(tpls[i]) for i in range(len(tpls))])
logging.info(f"pconts3 n nan: {np.unique(np.isnan(pconts3), return_counts=True)}")
pconts3[np.isnan(pconts3)] = False
pconts4 = np.array([IMSpoly_small.contains(tpls[i]) for i in range(len(tpls))])
logging.info(f"pconts4 n nan: {np.unique(np.isnan(pconts4), return_counts=True)}")
pconts3[np.isnan(pconts4)] = False
centsred_border = centsred[np.logical_and(np.logical_and(pconts1,~pconts2),np.logical_and(pconts3,~pconts4))]
logging.info(f"number of points at border: {centsred_border.shape[0]}")
inds_centsred_border = np.arange(centsred.shape[0])[np.logical_and(np.logical_and(pconts1,~pconts2),np.logical_and(pconts3,~pconts4))]

logging.info("Match postIMS to IMS points based on angles to neighbors")
logging.info(f"\t   ID\tmatches\ttested\ttotal")
kdt_centsred_border = KDTree(centsred_border, leaf_size=30, metric='euclidean')
kdt_ordered_centsred_border_all = KDTree(ordered_centsred_border_all, leaf_size=30, metric='euclidean')
distances, all_indices = kdt_ordered_centsred_border_all.query(centsred_border, k=1, return_distance=True)
indices = all_indices[distances==0].flatten()
nn1s=list(range(31))
nn2s=list(range(31))
max_dist_diff=0.2
max_angle_diff=15
from itertools import product
nn_combinations = np.array(list(product(nn1s, nn2s)))
nn_combinations = np.concatenate([nn_combinations,np.ones(nn_combinations.shape[0]).reshape(-1,1)*-1],axis=1).astype(int)
tl = np.array([p[0]+p[1] for p in nn_combinations])
nn_combinations = nn_combinations[np.logical_and(tl >= min_n,tl<=max_n)]
results_matching_array = np.zeros((len(centsred_border),len(nn1s),len(nn2s)),dtype=np.uint32)*np.nan
for k in range(len(nn_combinations)):
    nn1 = nn_combinations[k][0]
    nn2 = nn_combinations[k][1]
    ind1 = nn_combinations[:,0]==nn1
    ind2 = nn_combinations[:,1]<=nn2
    ind3 = nn_combinations[:,0]<=nn1
    ind4 = nn_combinations[:,1]==nn2
    if np.any(nn_combinations[np.logical_or(np.logical_and(ind1,ind2),np.logical_and(ind3,ind4)),2]==0):
        nn_combinations[k][2]=0
        logging.info(f"\t{nn1:02}_{nn2:02}\t{0:6}\t{0:6}\t{len(indices):6} - skipped")
        continue

    ind_to_keep = []
    # for all points, test if have sufficient neighbors to create code
    for i in range(len(indices)):
        tmpind = indices[i]
        tmpinds = indices_sequence_from_ordered_points(tmpind,nn1,nn2,len(ordered_centsred_border_all))
        tmp = ordered_centsred_border_all[tmpinds,:]
        if tmp.shape[0]==(nn1+nn2+1):
            dists = np.sqrt(np.sum(np.diff(tmp, axis=0)**2, axis=1))
            # dists = np.array([np.sqrt(np.sum((tmp[j,:]-tmp[j+1,:])**2)) for j in range(len(tmp)-1)])
            angles = get_angle_vec(-np.diff(tmp,axis=0).reshape(1,-1,2),[0,0],[1,0])
            # angles = np.array([get_angle(tmp[j,:]-tmp[j+1,:],[0,0],[1,0]) for j in range(len(tmp)-1)])
            to_keep_dist = np.logical_and(dists > 1-max_dist_diff, dists < 1+max_dist_diff)
            absangles = np.abs(np.array(angles))
            to_keep_angle = np.logical_or(
                    np.logical_or(absangles < 0+max_angle_diff, absangles > 180-max_angle_diff),
                    np.logical_and(absangles > 90-max_angle_diff, absangles < 90+max_angle_diff),
            )
            to_keep = np.logical_and(to_keep_angle, to_keep_dist)
            if np.all(to_keep):
                ind_to_keep.append(i)
    if len(ind_to_keep) == 0:
        nn_combinations[k][2]=0
        logging.info(f"\t{nn1:02}_{nn2:02}\t{0:6}\t{0:6}\t{len(indices):6}")
        continue
    codes = []
    # create codes
    for i in ind_to_keep:
        tmpind = indices[i]
        tmpinds = indices_sequence_from_ordered_points(tmpind,nn1,nn2,len(ordered_centsred_border_all))
        tmp = ordered_centsred_border_all[tmpinds,:]
        codes.append(angle_code_from_point_sequence(tmp))

    # find neighboring ims pixels
    tmppoints = ordered_centsred_border_all[indices[np.array(ind_to_keep)],:]
    tmp_distances, indices_tmp = kdt_tmpimzrot.query(tmppoints, k=9, return_distance=True)
    indices_tmp_ls = list(indices_tmp)
    distances_ls = list(tmp_distances)
    # filter by distance
    indices_tmp_ls = [indices_tmp_ls[i][distances_ls[i]<1.75] for i in range(len(indices_tmp_ls))]
    # filter by matching border point
    indices_tmp_ls = [indices_tmp_ls[i][(kdt_ordered_imz_border_all.query(tmpimzrot[indices_tmp_ls[i]], k=1, return_distance=True)[0]==0).flatten()] for i in range(len(indices_tmp_ls))]

    inds_to_keep = np.arange(len(indices_tmp_ls))[np.array([len(p) for p in indices_tmp_ls])>0]
    indices_tmp_ls = [indices_tmp_ls[p] for p in inds_to_keep]
    codes = np.array(codes)[inds_to_keep]
    ind_to_keep = np.array(ind_to_keep)[inds_to_keep]

    close_ims = [tmpimzrot[indices_tmp_ls[i]] for i in range(len(indices_tmp_ls))]

    # create codes for neighboring ims pixels
    ims_codes = []
    for j in range(len(close_ims)):
        ims_codes.append([])
        for i in range(len(close_ims[j])):
            tmpind = kdt_ordered_imz_border_all.query(close_ims[j][i,:].reshape(1,-1), k=1, return_distance=False)[0][0]
            tmpinds = indices_sequence_from_ordered_points(tmpind,nn1,nn2,len(ordered_imz_border_all))
            tmp = ordered_imz_border_all[tmpinds,:]
            ims_codes[j].append(angle_code_from_point_sequence(tmp))

    # compare matchings
    matches = [np.where(np.array(ims_codes[i]) == codes[i])[0] for i in range(len(codes))]

    n_matches = np.array([len(p) for p in matches])
    if np.sum(n_matches==1) == 0:
        logging.info(f"\t{nn1:02}_{nn2:02}\t{0:6}\t{len(ind_to_keep):6}\t{len(indices):6}")
        nn_combinations[k][2]=0
        continue

    close_ims_inds = np.arange(len(close_ims))[n_matches==1]

    logging.info(f"\t{nn1:02}_{nn2:02}\t{np.sum(n_matches==1):6}\t{len(ind_to_keep):6}\t{len(indices):6}")

    # create matching points
    ind_to_keep_filt = np.array(ind_to_keep)[np.array(n_matches)==1]
    tmppoints_filt = ordered_centsred_border_all[indices[np.array(ind_to_keep_filt)],:]
    ind_to_keep_filt_trans = kdt_centsred_border.query(tmppoints_filt,k=1)[1].flatten()
    tptsinds = [indices_tmp_ls[i] for i in close_ims_inds]
    matches_filt = np.array([matches[i] for i in np.arange(len(matches))[np.array(n_matches)==1]]).flatten()
    tptsinds_filt = np.array([tptsinds[i][matches_filt[i]] for i in range(len(matches_filt))])

    # save
    results_matching_array[ind_to_keep_filt_trans,np.array(nn1s)==nn1,np.array(nn2s)==nn2] = tptsinds_filt

    nn_combinations[k][2]=1
# find all points with at least 1 match
matches = np.sum(~np.isnan(results_matching_array),axis=(1,2))>0

# test if multiple matches were found
n_diff = np.array([len(np.unique(results_matching_array[i,:,:][~np.isnan(results_matching_array[i,:,:])])) for i in range(len(results_matching_array))])

# filter for matches with always the same IMS pixel
matches_filt = np.logical_and(matches, n_diff==1)
logging.info(f"Number of matches: {np.sum(matches_filt)}")


# create lenght of codes (i.e. strength of matching)
combslens = np.outer(np.ones(len(nn1s)),np.array(nn1s)) + np.outer(np.array(nn2s),np.ones(len(nn2s)))
maxlens = np.array([np.max(combslens[~np.isnan(results_matching_array[matches_filt,:,:][i,:,:])]) for i in range(np.sum(matches_filt))])

# compute maximum possible distance of points in IMS
from scipy.spatial.distance import cdist
tmppts = np.array(IMSpoly.exterior.coords.xy).T
hdist = cdist(tmppts, tmppts, metric='euclidean')
refdistmax = hdist.max()
centroid = np.mean(tmppts,axis=0)

all_maxlens = np.unique(maxlens)
logging.info(f"Unique length of matching codes before filtering: {all_maxlens}")
if np.sum(n_diff>1):
    has_ambiguous = []
    for i in range(np.sum(n_diff>1)):
        tmpind = results_matching_array[n_diff>1,:,:][i,:,:]
        tmpuqs = np.unique(tmpind[~np.isnan(tmpind)])
        has_ambiguous.append(np.min(np.array([np.max(combslens[tmpind == tmpuqs[j]]) for j in range(len(tmpuqs))])))
    maxlen_ambiguous = np.max(np.array(has_ambiguous))
    all_maxlens = all_maxlens[all_maxlens>maxlen_ambiguous]



logging.info(f"Unique length of matching codes after filtering: {all_maxlens}")
points_found = True
if len(all_maxlens)==0:
    logging.info("No matching points found")
    points_found = False
else:
    scores = np.zeros((len(all_maxlens),3))
    for i in range(len(all_maxlens)):
        tmp_matches_filt = matches_filt.copy()
        tmp_matches_filt[tmp_matches_filt] = maxlens >= all_maxlens[i]
        scores[i,0] = np.sum(tmp_matches_filt)

        centsred_borderfilt = centsred_border[tmp_matches_filt,:]
        hdist = cdist(centsred_borderfilt, centsred_borderfilt, metric='euclidean')
        scores[i,1] =  hdist.max()
        tpts = centsred_border[tmp_matches_filt]
        angles= np.array([get_angle(p1, centroid) for p1 in tpts])+180
        angles = np.sort(angles)
        angdiff = np.abs(np.diff(angles))
        angdiff = np.concatenate([angdiff,np.array((angles[0]-angles[-1])%360).reshape(1)])
        scores[i,2] = np.max(angdiff)

    logging.info(f"n_points\tmax_distance\tmax_angle") 
    logging.info(scores)
    threshold_npts = 10
    threshold_prop_maxdist = 0.75
    threadsold_maxangle = 200
    tmpsub = np.logical_and(scores[:,0]>threshold_npts, scores[:,1]>refdistmax*threshold_prop_maxdist, scores[:,2]<threadsold_maxangle)
    logging.info(f"Filtered with min number of points: {threshold_npts} and min max distance proportion: {threshold_prop_maxdist}")
    logging.info(scores[tmpsub,:])

    if np.sum(tmpsub)>0:

        n_wrong_matches=np.zeros((scores[tmpsub,:].shape[0],2))
        n_wrong_matches[:,0]=all_maxlens[tmpsub]
        for i,tmpmaxlen in enumerate(all_maxlens[tmpsub]):
            tmp_matches_filt = matches_filt.copy()
            tmp_matches_filt[tmp_matches_filt] = maxlens >= tmpmaxlen 
            tmpind_to_use = np.arange(tmp_matches_filt.shape[0])[tmp_matches_filt]

            matching_inds = np.array([np.unique(results_matching_array[tmpind_to_use[i],:,:][~np.isnan(results_matching_array[tmpind_to_use[i],:,:])])[0] for i in range(len(tmpind_to_use))]).astype(np.uint32)

            centsred_borderfilt = centsred_border[tmpind_to_use,:]
            tmpimzrotfilt = tmpimzrot[matching_inds,:]

            tmpdiffs = np.sqrt(np.sum((centsred_borderfilt - tmpimzrotfilt)**2,axis=1))
            from scipy.spatial.distance import cdist
            diffs_exp = np.stack([tmpdiffs,np.zeros(tmpdiffs.shape[0])],axis=1)
            diffs=cdist(diffs_exp, diffs_exp, metric='cityblock')
            dists=cdist(centsred_borderfilt, centsred_borderfilt, metric='euclidean')
            dists[dists==0]=np.nan
            n_wrong_matches[i,1] = np.sum(diffs>0.5+dists*0.1)

        logging.info(f"Number of pairwise wrong matches: {n_wrong_matches}")

        score_comb = (1-scores[tmpsub,2]/360)*np.sqrt(all_maxlens[tmpsub])
        score_comb[n_wrong_matches[:,1]>0] = 0
        if np.any(score_comb>0):
            maxlen_to_use = all_maxlens[np.argmax(score_comb)]
            logging.info(f"Maxlen used: {maxlen_to_use}")
            tmp_matches_filt = matches_filt.copy()
            tmp_matches_filt[tmp_matches_filt] = maxlens >= maxlen_to_use
            tmpind_to_use = np.arange(tmp_matches_filt.shape[0])[tmp_matches_filt]

            matching_inds = np.array([np.unique(results_matching_array[tmpind_to_use[i],:,:][~np.isnan(results_matching_array[tmpind_to_use[i],:,:])])[0] for i in range(len(tmpind_to_use))]).astype(np.uint32)

            centsred_borderfilt = centsred_border[tmpind_to_use,:]
            tmpimzrotfilt = tmpimzrot[matching_inds,:]

            # plt.scatter(tmpimzrot[:,0],tmpimzrot[:,1],c="red",alpha=0.2)
            # plt.scatter(centsred_border[:,0],centsred_border[:,1],c="blue",alpha=0.2)
            # plt.scatter(tmpimzrotfilt[:,0],tmpimzrotfilt[:,1],c="red")
            # plt.scatter(centsred_borderfilt[:,0],centsred_borderfilt[:,1],c="blue")
            # plt.show()
        else:
            logging.info(f"No matching points found")
            points_found = False
    else:
        logging.info("No parameters fullfil criteria!")
        points_found = False

if points_found:
    logging.info(f"Run skimage ransac registration on matching points")
    inliersls = []
    modells=[]
    for i in range(1000):
        mat, inliers = skimage.measure.ransac((centsred_borderfilt,tmpimzrotfilt), skimage.transform.AffineTransform, min_samples=7, residual_threshold=0.1, max_trials=10000)
        modells.append(mat.params)
        inliersls.append(np.array(inliers))
    inliers = np.vstack(inliersls)
    
    # np.sum(np.sum(inliers,axis=1)==np.max(np.sum(inliers,axis=1)))
    # np.array(modells)[np.array(inliersls)==np.max(inliersls)]
    outlier = np.sum(inliers,axis=0)<900
    logging.info(f"Number of outliers: {np.sum(outlier)}/{len(outlier)}")

    if np.sum(outlier)/len(outlier)>0.1:
        logging.info(f"Too many outliers, registration not used")
        R_reg = np.array([[1.0,0.0],[0.0,1.0]])
        t_reg = np.array([0.0,0.0])
        init_transform = tmp_transform.GetInverse()
        init_transform_inverse = tmp_transform
        pycpd_transform_inverse = sitk.AffineTransform(2)
        pycpd_transform = sitk.AffineTransform(2)
    else:
        centsred_borderfilt=centsred_borderfilt[~outlier,:]
        tmpimzrotfilt=tmpimzrotfilt[~outlier,:]
        logging.info(f"Filter outliers")

        logging.info(f"Run pycpd registration on matching points") 
        reg = pycpd.AffineRegistration(Y=centsred_borderfilt.astype(float), X=tmpimzrotfilt.astype(float), w=0, s=1, scale=False)
        postIMScoordsout, (R_reg, t_reg) = reg.register()
# 
        # target_pts_flat = [c for p in centsred_borderfilt[:, [1, 0]] for c in p]
        # source_pts_flat = [c for p in tmpimzrotfilt[:, [1, 0]] for c in p]
        # landmark_initializer = sitk.LandmarkBasedTransformInitializerFilter()
        # transform = sitk.AffineTransform(
            # sitk.LandmarkBasedTransformInitializer(
                # sitk.AffineTransform(2), target_pts_flat, source_pts_flat
            # )
        # )
        # tmpmat = transform.GetMatrix()
        # R_reg = np.array([[tmpmat[3],tmpmat[1]],[tmpmat[2],tmpmat[0]]])
        # t_reg = np.flip(np.array(transform.GetTranslation()))

        logging.info(f"Create transformations") 
        tmp_transform_inverse = sitk.Euler2DTransform()
        tmp_transform_inverse.SetCenter(tmp_transform.GetCenter())
        tmp_transform_inverse.SetAngle(-tmp_transform.GetAngle())
        tmp_transform_inverse.SetTranslation(-np.array(tmp_transform.GetTranslation()))
        # centsredtrans = np.array([tmp_transform_inverse.TransformPoint(centsred[i,:]) for i in range(centsred.shape[0])])
        pycpd_transform = sitk.AffineTransform(2)
        pycpd_transform.SetTranslation(t_reg)
        pycpd_transform.SetMatrix(R_reg.T.flatten())
        # postIMScoordsout = np.array([pycpd_transform.TransformPoint(centsredtrans[i,:]) for i in range(centsredtrans.shape[0])])

        tm = sitk.CompositeTransform(2)
        tm.AddTransform(pycpd_transform)
        tm.AddTransform(tmp_transform_inverse)
        init_transform = composite2affine(tm, [0,0])
    
        postIMScoordsout = np.array([init_transform.TransformPoint(centsred[i,:]) for i in range(centsred.shape[0])])

        pycpd_transform_inverse = sitk.AffineTransform(2)
        pycpd_transform_inverse.SetTranslation(-t_reg)
        R_reg_inv = np.array([[1-(R_reg[0,0]-1),-R_reg[1,0]],[-R_reg[0,1],1-(R_reg[1,1]-1)]])
        pycpd_transform_inverse.SetMatrix(R_reg_inv.flatten())

        tm = sitk.CompositeTransform(2)
        tm.AddTransform(pycpd_transform_inverse)
        tm.AddTransform(tmp_transform)
        init_transform_inverse = composite2affine(tm, [0,0])
 
        tmpfilename = f"{os.path.dirname(snakemake.log['stdout'])}/{os.path.basename(snakemake.log['stdout']).split('.')[0]}_cpd_registration.svg"
        plt.close()
        plt.scatter(imzcoordsfilttrans[:,1], imzcoordsfilttrans[:,0],color="red",alpha=0.2)
        plt.scatter(postIMScoordsout[:,1], postIMScoordsout[:,0],color="blue",alpha=0.2)
        tmppostIMScoordsout = np.array([init_transform.TransformPoint(centsred_borderfilt[i,:]) for i in range(centsred_borderfilt.shape[0])])
        plt.scatter(imzcoordsfilttrans[matching_inds,1], imzcoordsfilttrans[matching_inds,0],color="red",alpha=0.5)
        plt.scatter(tmppostIMScoordsout[:,1], tmppostIMScoordsout[:,0],color="blue",alpha=0.5)
        plt.title("matching points")
        fig = plt.gcf()
        fig.set_size_inches(20,20)
        fig.savefig(tmpfilename)
        # plt.show()

        # IMSpoly = shapely.concave_hull(shapely.geometry.MultiPoint(imzcoordsfilttrans), ratio=0.001).buffer(0.15, cap_style='square', join_style='mitre')
        tpts = np.array(tmpIMSpoly.exterior.coords.xy).T
        imzcoordsfilttranstrans = np.array([tmp_transform_inverse.TransformPoint(tpts[i,:]) for i in range(tpts.shape[0])])
        tmpIMSpolytrans = shapely.Polygon(imzcoordsfilttranstrans)
        # plt.scatter(imzcoordsfilttrans[:,0], imzcoordsfilttrans[:,1],color="blue",alpha=0.5)
        # plt.scatter(imzcoordsfilttranstrans[:,0], imzcoordsfilttranstrans[:,1],color="red",alpha=0.5)
        # shapely.plotting.plot_polygon(tmpIMSpolytrans)
        # plt.show()

        IMSpolytrans = tmpIMSpolytrans.buffer(0.15)
        shapely.prepare(IMSpolytrans)
        # shapely.plotting.plot_polygon(IMSpolytrans)
        # plt.show()

        tpls = [shapely.geometry.Point(postIMScoordsout[i,:]) for i in range(postIMScoordsout.shape[0])]
        pconts = np.array([IMSpolytrans.contains(tpls[i]) for i in range(len(tpls))])
        if np.sum(pconts)/len(pconts) < 0.98:
            logging.info(f"only {np.sum(pconts):6}/{len(pconts):6} points ({np.sum(pconts)/len(pconts):1.4}) lie inside of IMS polygon: Registration is not used!") 
            R_reg = np.array([[1.0,0.0],[0.0,1.0]])
            t_reg = np.array([0.0,0.0])
else:
    logging.info(f"Criteria not met, don't run pycpd registration") 
    R_reg = np.array([[1.0,0.0],[0.0,1.0]])
    t_reg = np.array([0.0,0.0])
    init_transform = tmp_transform.GetInverse()
    init_transform_inverse = tmp_transform
    pycpd_transform_inverse = sitk.AffineTransform(2)
    pycpd_transform = sitk.AffineTransform(2)


def resample_image(transform, fixed, moving_np):
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return sitk.GetArrayFromImage(resampler.Execute(sitk.GetImageFromArray(moving_np)))

logging.info("Prepare SITK registration")

# Read IMC image mask
imcmask = readimage_crop(imc_mask_file, [int(xmin), int(ymin), int(xmax), int(ymax)])
imcmaskch = skimage.morphology.convex_hull_image(imcmask>0)
imcmaskch = skimage.transform.resize(imcmaskch,postIMS_shape)
imcmaskch = cv2.morphologyEx(src=imcmaskch.astype(np.uint8), op = cv2.MORPH_DILATE, kernel = skimage.morphology.square(2*int((1/resolution)*stepsize*2))).astype(bool)
imcmaskch = skimage.transform.resize(imcmaskch, postIMS_shape, preserve_range = True)
# transform IMC mask
sitkimcmaskch = sitk.GetImageFromArray(imcmaskch.astype(np.uint8)) 
sitkimcmaskch.SetSpacing((resolution/stepsize,resolution/stepsize))
imcmaskch = sitk.GetArrayFromImage(sitk.Resample(sitkimcmaskch, tmp_transform.GetInverse())).astype(bool)

# polygon from IMC image 
contours,_ = cv2.findContours(imcmaskch.astype(np.uint8),cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours = np.squeeze(contours[0])
poly = shapely.geometry.Polygon(contours)
poly = poly.buffer(2)
# check number of points in mask for postIMS
tpls = [shapely.geometry.Point(centsred[i,:]/resolution*stepsize) for i in range(centsred.shape[0])]
pcontsc = np.array([poly.contains(tpls[i]) for i in range(len(tpls))])

# check number of points in mask for IMS
tpts = np.array([init_transform_inverse.TransformPoint(imzcoordsfilttrans[i,:]) for i in range(imzcoordsfilttrans.shape[0])])
tpls = [shapely.geometry.Point(tpts[i,:]/resolution*stepsize) for i in range(tpts.shape[0])]
pcontsi = np.array([poly.contains(tpls[i]) for i in range(len(tpls))])

# calculate centroids of points in IMC mask
obs_centroid = np.mean(centsred[pcontsc,:],axis=0)-np.min(centsred[pcontsc,:],axis=0)
theo_centroid = np.mean(tpts[pcontsi,:],axis=0)-np.min(tpts[pcontsi,:],axis=0)
# calculate difference to max possible value
theo_range = np.max(tpts[pcontsi,:],axis=0)-np.min(tpts[pcontsi,:],axis=0)
prop_diff = (obs_centroid - theo_centroid)/theo_range/2

tpts = np.array([init_transform.TransformPoint(centsred[i,:]) for i in range(centsred.shape[0])])
postIMSpimg = image_from_points(postIMS_shape, tpts/resolution*stepsize,get_sigma((stepsize-pixelsize)/resolution,0.95), int(pixelsize/4/resolution))
IMSpimg = image_from_points(postIMS_shape, imzcoordsfilttrans/resolution*stepsize,get_sigma((stepsize-pixelsize)/resolution,0.95), int(pixelsize/4/resolution))

logging.info(f"Number of points in IMC location: {np.sum(pcontsc)} / {np.sum(pcontsi)}")
logging.info(f"Distance between centroid of postIMS points and IMC center: {obs_centroid - theo_centroid} (Max possible: {theo_range})")
# filter for number of points and location of centroid
if (np.sum(pcontsc) > np.sum(pcontsi)/10) and np.all(prop_diff < 0.25) and do_within_IMC_fine_registration:
    
    logging.info(f"Run sitk registration")
    postIMSpimgcompl = postIMSpimg.copy()
    IMSpimgcompl = IMSpimg.copy()

    # plt.imshow(IMSpimg.astype(float)-postIMSpimg.astype(float))
    # plt.show()
    postIMSpimg[~imcmaskch] = postIMSpimg[~imcmaskch]/2
    IMSpimg[~imcmaskch] = IMSpimg[~imcmaskch]/2

    # Run registration
    logging.info("Run SITK registration")
    fixed = sitk.GetImageFromArray(IMSpimg.astype(float))
    moving = sitk.GetImageFromArray(postIMSpimg.astype(float))

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetMetricSamplingStrategy(R.REGULAR)
    R.SetMetricSamplingPercentage(0.25, seed=1234)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsGradientDescent(
        learningRate=1, numberOfIterations=1000, 
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
        estimateLearningRate=R.EachIteration
    )
    # R.SetOptimizerScalesFromIndexShift()
    R.SetOptimizerScalesFromPhysicalShift()
    # R.SetInitialTransform(init_transform3)
    R.SetInitialTransform(sitk.AffineTransform(2))
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    transform = R.Execute(fixed, moving)

    postIMSro_trans = resample_image(transform, fixed, postIMSpimg)
    tmpfilename = f"{os.path.dirname(snakemake.log['stdout'])}/{os.path.basename(snakemake.log['stdout']).split('.')[0]}_sitk_registration.ome.tiff"
    logging.info(f"Save Image difference as: {tmpfilename}")
    saveimage_tile(((postIMSro_trans.astype(float)-IMSpimg.astype(float))+255)/2, tmpfilename, resolution)

    # transform_inverse = sitk.AffineTransform(2)
    # transform_inverse.SetCenter(np.flip(np.array(transform.GetCenter())))
    # transform_inverse.SetTranslation(-np.flip(np.array(transform.GetTranslation())))
    # transform_inverse.SetMatrix(transform.GetMatrix())
    # tmpmat = transform.GetMatrix()
    # tmpmat_inv = np.array([[1-(tmpmat[3]-1),-tmpmat[2]],[-tmpmat[1],1-(tmpmat[0]-1)]])
    # transform_inverse.SetMatrix(tmpmat_inv.flatten())
    # transform_inverse.GetParameters()


    # tpts = np.array([init_transform.TransformPoint(centsred[i,:]) for i in range(centsred.shape[0])])
    # tpts_scaled = tpts/resolution*stepsize
    # tpts_scaled = np.array([transform_inverse.TransformPoint(tpts_scaled[i,:]) for i in range(tpts_scaled.shape[0])])

    # fig, (ax1, ax2) = plt.subplots(1,2)
    # ax1.imshow(IMSpimg.astype(float)[1700:2500,4100:4900]-postIMSro_trans.astype(float)[1700:2500,4100:4900])
    # ax2.scatter(imzcoordsfilttrans[:,1]/resolution*stepsize,imzcoordsfilttrans[:,0]/resolution*stepsize)
    # ax2.scatter(tpts_scaled[:,1],tpts_scaled[:,0])
    # ax2.set_xlim([4100,4900])
    # ax2.set_ylim([1700,2500])
    # ax2.invert_yaxis()
    # ax2.set_aspect('equal', 'box')
    # plt.show()

    # tpts = np.array([init_transform_inverse.TransformPoint(imzcoordsfilttrans[i,:]) for i in range(imzcoordsfilttrans.shape[0])])
    # plt.scatter(tpts[:,1],tpts[:,0])
    # plt.scatter(centsred[:,1],centsred[:,0])
    # plt.show()
    transform_scaled_inverse = sitk.AffineTransform(2)
    transform_scaled_inverse.SetCenter(np.flip(np.array(transform.GetCenter())*resolution/stepsize))
    transform_scaled_inverse.SetTranslation(-np.flip(np.array(transform.GetTranslation())*resolution/stepsize))
    tmpmat = transform.GetMatrix()
    tmpmat_inv = np.array([[1-(tmpmat[3]-1),-tmpmat[2]],[-tmpmat[1],1-(tmpmat[0]-1)]])
    transform_scaled_inverse.SetMatrix(tmpmat_inv.flatten())
    transform_scaled_inverse.GetParameters()

    # tpts = np.array([init_transform.TransformPoint(centsred[i,:]) for i in range(centsred.shape[0])])
    # tpts2 = np.array([transform_scaled_inverse.TransformPoint(tpts[i,:]) for i in range(tpts.shape[0])])
    # plt.scatter(imzcoordsfilttrans[:,1],imzcoordsfilttrans[:,0])
    # plt.scatter(tpts2[:,1],tpts2[:,0])
    # plt.show()


    transform_scaled = transform_scaled_inverse.GetInverse()

    # tpts = np.array([init_transform_inverse.TransformPoint(imzcoordsfilttrans[i,:]) for i in range(imzcoordsfilttrans.shape[0])])
    # tpts2 = np.array([transform_scaled_inverse.TransformPoint(centsred[i,:]) for i in range(centsred.shape[0])])
    # plt.scatter(tpts[:,1],tpts[:,0])
    # plt.scatter(tpts2[:,1],tpts2[:,0])
    # plt.show()

    # tpts2 = np.array([transform_scaled.TransformPoint(tpts[i,:]) for i in range(tpts.shape[0])])
    # plt.scatter(tpts2[:,1],tpts2[:,0])
    # plt.scatter(centsred[:,1],centsred[:,0])
    # plt.show()


else:
    logging.info(f"Criteria not met, do NOT run additional registration!")
    # Identity transform
    transform_scaled_inverse = sitk.AffineTransform(2)
    transform_scaled = sitk.AffineTransform(2)
    transform = sitk.AffineTransform(2)

logging.info(f"Combine Sitk with cpd Registration")
tm = sitk.CompositeTransform(2)
tm.AddTransform(init_transform_inverse)
tm.AddTransform(transform_scaled)
transform_comb = composite2affine(tm, [0,0])

tpts = np.array([transform_comb.TransformPoint(imzcoordsfilttrans[i,:]) for i in range(imzcoordsfilttrans.shape[0])])
tmpfilename = f"{os.path.dirname(snakemake.log['stdout'])}/{os.path.basename(snakemake.log['stdout']).split('.')[0]}_combined_registration.svg"
plt.close()
plt.scatter(tpts[:,1], tpts[:,0],color="red",alpha=0.5)
plt.scatter(centsred[:,1], centsred[:,0],color="blue",alpha=0.5)
plt.title("matching points")
fig = plt.gcf()
fig.set_size_inches(20,20)
fig.savefig(tmpfilename)
# plt.show()

logging.info(f"Combine Registrations")
# combined transformation steps
# tm = sitk.CompositeTransform(2)
# tm.AddTransform(transform_comb)
# tm.AddTransform(tinv)
# pycpd_transform_comb = composite2affine(tm, [0,0])
# pycpd_transform_comb.GetParameters()



imzcoords_all = create_imz_coords(imzimg, None, imzrefcoords, imz_bbox, rotmat)
imzcoords_in = imzcoords_all + init_translation

# no sitk, combined transformation steps
tm = sitk.CompositeTransform(2)
tm.AddTransform(init_transform_inverse)
tm.AddTransform(tinv)
pycpd_transform_comb_nositk = composite2affine(tm, [0,0])
pycpd_transform_comb_nositk.GetParameters()

# no sitk, combined transformation steps
tm = sitk.CompositeTransform(2)
tm.AddTransform(transform_comb)
tm.AddTransform(tinv)
pycpd_transform_comb = composite2affine(tm, [0,0])
pycpd_transform_comb.GetParameters()

imzcoordstransformed_nositk = np.array([pycpd_transform_comb_nositk.TransformPoint(imzcoords_in[i,:].astype(float)) for i in range(imzcoords_in.shape[0])])
centsredinmask = centsred[pcontsc]
kdt = KDTree(imzcoordstransformed_nositk, leaf_size=30, metric='euclidean')
distances_inmask_nositk, indices = kdt.query(centsredinmask, k=1, return_distance=True)
q50_error_inmask_nositk = np.quantile(distances_inmask_nositk,0.5)*stepsize
logging.info(f"Median error in mask is {q50_error_inmask_nositk} (no sitk)")

imzcoordstransformed = np.array([pycpd_transform_comb.TransformPoint(imzcoords_in[i,:].astype(float)) for i in range(imzcoords_in.shape[0])])
kdt = KDTree(imzcoordstransformed, leaf_size=30, metric='euclidean')
distances_inmask, indices = kdt.query(centsredinmask, k=1, return_distance=True)
q50_error_inmask = np.quantile(distances_inmask,0.5)*stepsize
logging.info(f"Median error in mask is {q50_error_inmask} (sitk)")

if q50_error_inmask > q50_error_inmask_nositk*1.2:
    logging.info(f"Median error in mask is 20% higher than median error: Don't use sitk registration!")
    pycpd_transform_comb = pycpd_transform_comb_nositk

# tm = sitk.CompositeTransform(2)
# tm.AddTransform(init_transform)
# tm.AddTransform(transform_scaled_inverse)
# transform_comb = composite2affine(tm, [0,0])

# tpts = np.array([transform_comb.TransformPoint(centsred[i,:]) for i in range(centsred.shape[0])])
# plt.scatter(imzcoordsfilttrans[:,1],imzcoordsfilttrans[:,0])
# plt.scatter(tpts[:,1],tpts[:,0])
# plt.show()


# imzcoords_all = create_imz_coords(imzimg, None, imzrefcoords, imz_bbox, rotmat)
# imzcoords_in = imzcoords_all + init_translation
imzcoordstransformed = np.array([pycpd_transform_comb.TransformPoint(imzcoords_in[i,:].astype(float)) for i in range(imzcoords_in.shape[0])])

tmpfilename = f"{os.path.dirname(snakemake.log['stdout'])}/{os.path.basename(snakemake.log['stdout']).split('.')[0]}_combined_registration_all.svg"
plt.close()
plt.scatter(imzcoordstransformed[:,1]*stepsize/resolution, imzcoordstransformed[:,0]*stepsize/resolution,color="green",alpha=0.5)
plt.scatter(centsred[:,1]*stepsize/resolution, centsred[:,0]*stepsize/resolution,color="blue",alpha=0.5)
plt.title("matching points")
fig = plt.gcf()
fig.set_size_inches(30,30)
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

logging.info("Final pycpd registration:")
logging.info(f"Number of points IMS: {imzcoords_in.shape[0]}")
logging.info(f"Number of points postIMS: {centsred.shape[0]}")
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

imzcoordstransformedinmask = imzcoordstransformed[pcontsi]
centsredinmask = centsred[pcontsc]
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
# vie._data.micro_d.res_info_input.setText(str(1))
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
yshift = ymin*resolution
xshift = xmin*resolution
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
# vie.image_transformer.output_spacing = [1, 1]
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

pmap_coord_data["x_micro_physical"] = transformed_coords_micro[:, 0]
pmap_coord_data["y_micro_physical"] = transformed_coords_micro[:, 1]


logging.info("Get mean error after registration")
centsredfilttrans = centsred*stepsize+np.array([xshift,yshift])
imzcoordstrans = np.stack([np.array(pmap_coord_data["y_micro_physical"].to_list()).T,np.array(pmap_coord_data["x_micro_physical"].to_list()).T], axis=1)

kdt = KDTree(centsredfilttrans, leaf_size=30, metric='euclidean')
distances, indices = kdt.query(imzcoordstrans, k=1, return_distance=True)
ims_point_to_keep = distances[:,0]<0.5*stepsize

kdt = KDTree(imzcoordstrans[ims_point_to_keep,:], leaf_size=30, metric='euclidean')
distances, indices = kdt.query(centsredfilttrans, k=1, return_distance=True)
point_to_keep = distances[:,0]<0.5*stepsize
centsredfilttrans = centsredfilttrans[point_to_keep,:]

matches = skimage.feature.match_descriptors(imzcoordstrans[ims_point_to_keep,:], centsredfilttrans, metric='euclidean', cross_check=True, max_distance=0.5*stepsize)

imzcoordstransfilt = np.stack([np.array(pmap_coord_data["y_padded"].to_list()).T,np.array(pmap_coord_data["x_padded"].to_list()).T], axis=1)
imzcoordstransfilt = imzcoordstransfilt[ims_point_to_keep,:][matches[:,0],:]
centsredfilttrans = centsredfilttrans[matches[:,1],:]

vie.image_transformer.source_pts = centsredfilttrans
vie.image_transformer.target_pts = imzcoordstransfilt*stepsize
logging.info("Error: "+ str(np.mean(distances[point_to_keep,0])))
logging.info("Number of points: "+ str(np.sum(point_to_keep)))
reg_measure_dic['IMS_to_postIMS_error'] = str(np.mean(distances[point_to_keep,0]))
reg_measure_dic['n_points'] = str(np.sum(point_to_keep))
json.dump(reg_measure_dic, open(ims_to_postIMS_regerror,"w"))


project_metadata, pmap_coord_data = vie._generate_pmap_coords_and_meta(project_name)
(
    transformed_coords_ims,
    transformed_coords_micro,
    transformed_coords_micro_px,
) = vie._transform_ims_coords_to_microscopy()
pmap_coord_data["x_micro_ims_px"] = transformed_coords_ims[:, 0]
pmap_coord_data["y_micro_ims_px"] = transformed_coords_ims[:, 1]
pmap_coord_data["x_micro_physical"] = transformed_coords_micro[:, 0]
pmap_coord_data["y_micro_physical"] = transformed_coords_micro[:, 1]
pmap_coord_data["x_micro_px"] = transformed_coords_micro_px[:, 0]
pmap_coord_data["y_micro_px"] = transformed_coords_micro_px[:, 1]
print(pmap_coord_data)
print(transformed_coords_ims)
print(transformed_coords_micro)


pmeta_out_fp = Path(output_dir) / f"{project_name}-IMSML-meta.json"
logging.info("Save data")
with open(pmeta_out_fp, "w") as json_out:
    json.dump(project_metadata, json_out, indent=1, cls=NpEncoder)

coords_out_fp = Path(output_dir) / f"{project_name}-IMSML-coords.h5"
napari_imsmicrolink.utils.coords.pmap_coords_to_h5(pmap_coord_data, coords_out_fp)


# check match
postIMScut = readimage_crop(postIMS_file, [int(xmin), int(ymin), int(xmax), int(ymax)])
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
        xinds = (np.array(pmap_coord_data["y_micro_physical"].to_list())/resolution+i-xmin).astype(int)
        yinds = (np.array(pmap_coord_data["x_micro_physical"].to_list())/resolution+j-ymin).astype(int)
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
