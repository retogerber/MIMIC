import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
import cv2
import SimpleITK as sitk
import pandas as pd
import json
from sklearn.neighbors import KDTree
import pycpd
import napari_imsmicrolink
import skimage
import numpy as np
from ome_types import from_tiff
from image_utils import readimage_crop, saveimage_tile
from utils import setNThreads, snakeMakeMock
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["IMS_pixelsize"] = 30
    snakemake.params["IMS_shrink_factor"] = 0.8
    snakemake.params["microscopy_pixelsize"] = 0.22537
    snakemake.params["IMS_rotation_angle"] = 180
    snakemake.params["sample_core_names"] = "id1007966_001|-_-|A1|-|-|id1007966_002|-_-|A2"
    snakemake.input["postIMSmask_downscaled"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow_biopsies/results/id1007966/data/postIMS/id1007966_postIMS_reduced_mask.ome.tiff"
    snakemake.input["imzml"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow_biopsies/results/id1007966/data/IMS/1007966-3_AAXL_01022022.imzML"
    snakemake.input["IMCmask"] = ["/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow_biopsies/results/id1007966/data/IMC_location/id1007966_IMC_mask_on_postIMS_A1.geojson", "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow_biopsies/results/id1007966/data/IMC_location/id1007966_IMC_mask_on_postIMS_A2.geojson"]
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
rescale = float(snakemake.params["out_rescale"])

# inputs
postIMSr_file = snakemake.input["postIMSmask_downscaled"]
imzmlfile = snakemake.input["imzml"]
imc_mask_files = snakemake.input["IMCmask"]
if isinstance(imc_mask_files, str):
    imc_mask_files = [imc_mask_files]
sample_core_names = snakemake.params["sample_core_names"]
# outputs
output_table = snakemake.output["IMS_to_postIMS_matches"]


postIMS_ome = from_tiff(postIMSr_file)
postIMS_resolution = postIMS_ome.images[0].pixels.physical_size_x
logging.info(f"postIMS resolution: {postIMS_resolution}")
assert abs(postIMS_resolution - resolution*rescale) < 1e-3

logging.info("IMC location bounding boxes:")
imcbboxls = list()
for imcmaskfile in imc_mask_files:
    IMC_geojson = json.load(open(imcmaskfile, "r"))
    if isinstance(IMC_geojson,list):
        IMC_geojson=IMC_geojson[0]
    boundary_points = np.array(IMC_geojson['geometry']['coordinates'])[0,:,:]
    xmin=int(np.min(boundary_points[:,1]))
    xmax=int(np.max(boundary_points[:,1]))
    ymin=int(np.min(boundary_points[:,0]))
    ymax=int(np.max(boundary_points[:,0]))
    bbox = np.array([xmin,ymin,xmax,ymax])
    if rescale != 1:
        bbox = (bbox/rescale).astype(int)
    logging.info(f"    {bbox}:{imcmaskfile}")
    imcbboxls.append(bbox)


# get file metadata and matches between core and sample name
imc_projects = [ os.path.split(os.path.split(os.path.split(os.path.split(f)[0])[0])[0])[1] for f in imc_mask_files]
core_names = [ os.path.splitext(os.path.splitext(os.path.split(imc_mask_files[f])[1])[0])[0].replace(f"{imc_projects[f]}_IMC_mask_on_postIMS_","") for f in range(len(imc_mask_files))]

sample_core_names_ls = sample_core_names.split("|-|-|")
core_names_alt = np.array([s.split("|-_-|")[1] for s in sample_core_names_ls])
sample_names_alt = np.array([s.split("|-_-|")[0] for s in sample_core_names_ls])

sample_names = list()
for s in core_names:
    sample_names.append(sample_names_alt[core_names_alt==s][0])
sample_names = np.array(sample_names)

logging.info("Sample core names input: "+sample_core_names)
logging.info("Sample names: "+",".join(sample_names.tolist()))
logging.info("Core names: "+",".join(core_names_alt.tolist()))
logging.info("imc sample names: "+",".join(sample_names))

logging.info("Read postIMS mask")
postIMSr = skimage.io.imread(postIMSr_file)
logging.info(f"\t{postIMSr.shape}\t{postIMSr_file}")

_,postIMSregin,pistats,picents = cv2.connectedComponentsWithStats(postIMSr, ltype=cv2.CV_16U)
postIMSregin = postIMSregin.astype(np.uint8)
del postIMSr
# remove region corresponding to background (=0)
pistats=pistats[1:,:]
picents=np.flip(picents[1:,:],axis=1)

logging.info(f"\tCentroids of postIMS: {picents}")

# flip x and y axis
xs = pistats[:,1]
ys = pistats[:,0]
heights = pistats[:,2]
widths = pistats[:,3]

logging.info("Find IMC to postIMS overlap")
picents_reduced = list()
postIMSregions = list()
postIMS_pre_bbox = list()
postIMS_pre_areas = list()
for bb in imcbboxls:
    tmpuqs = np.unique([postIMSregin[bb[0],bb[1]], postIMSregin[bb[0],bb[3]], postIMSregin[bb[2],bb[1]], postIMSregin[bb[2],bb[3]]])
    tmpuqs = tmpuqs[tmpuqs>0]
    assert(len(tmpuqs)==1)
    postIMSregions.append(tmpuqs[0])
    r = tmpuqs-1
    postIMS_pre_bbox.append([xs[r][0],ys[r][0],xs[r][0]+widths[r][0], ys[r][0]+heights[r][0]])
    picents_reduced.append(picents[r,:])
    postIMS_pre_areas.append(pistats[r,4][0])
picents = np.vstack(picents_reduced)
del tmpuqs

logging.info(f"postIMSregions: {postIMSregions}")
logging.info(f"postIMS_pre_bbox: {postIMS_pre_bbox}")
logging.info(f"postIMS_pre_areas: {postIMS_pre_areas}")

# filter
tmpbool = np.array([p in np.array(postIMSregions) for p in postIMSregions])
assert(np.any(tmpbool))
if len(tmpbool)==1:
    postIMS_pre_bbox = np.array(postIMS_pre_bbox)
    postIMS_pre_areas = np.array(postIMS_pre_areas)
else: 
    postIMS_pre_bbox = np.array(postIMS_pre_bbox)[tmpbool]
    postIMS_pre_areas = np.array(postIMS_pre_areas)[tmpbool]
logging.info(f"postIMS_pre_bbox filtered: {postIMS_pre_bbox}")
logging.info(f"postIMS_pre_areas filtered: {postIMS_pre_areas}")
logging.info("Find global bounding box of cores")
global_bbox = [
    np.max([0,np.min([p[0] for p in postIMS_pre_bbox])-int(1/(resolution*rescale))]),
    np.max([0,np.min([p[1] for p in postIMS_pre_bbox])-int(1/(resolution*rescale))]),
    np.min([np.max([p[2] for p in postIMS_pre_bbox])+int(1/(resolution*rescale)),postIMSregin.shape[0]]),
    np.min([np.max([p[3] for p in postIMS_pre_bbox])+int(1/(resolution*rescale)),postIMSregin.shape[1]]),
]
del tmpbool
logging.info(f"\txmin: {global_bbox[0]}")
logging.info(f"\tymin: {global_bbox[1]}")
logging.info(f"\txmax: {global_bbox[2]}")
logging.info(f"\tymax: {global_bbox[3]}")



logging.info("Read imzML file")
# read imzml file
imz = napari_imsmicrolink.data.ims_pixel_map.PixelMapIMS(imzmlfile)
# stepsize (not actually used)
imz.ims_res = stepsize
# create image mask
imzimg = imz._make_pixel_map_at_ims(randomize=False, map_type="minimized")
logging.info(f"IMZML shape: {imzimg.shape}")

logging.info("Apply rotation")

# rotate 180 degrees
imzimg = skimage.transform.rotate(imzimg,rotation_imz, preserve_range=True)
imzimg = imzimg.astype(np.uint8)

logging.info("Create IMZ image")
# create imz region image
y_extent, x_extent, y_coords, x_coords = imz._get_xy_extents_coords(map_type="minimized")
imzregions = np.zeros((y_extent, x_extent), dtype=np.uint8)
imzregions[y_coords, x_coords] = imz.regions
imzregions = skimage.transform.rotate(imzregions,rotation_imz, preserve_range=True)
imzregions = np.round(imzregions)
imzregions = imzregions.astype(np.uint8)
# remove imz areas that are clearly too small to match
imzregpop = skimage.measure.regionprops(imzregions)
imzlabels = np.array([r.label for r in imzregpop])
imzareas = np.array([r.area for r in imzregpop]) * (stepsize/(resolution*rescale))**2
logging.info(f"IMZ regions: {imzlabels}")
logging.info(f"IMZ areas: {imzareas}")
to_remove = imzareas < np.min(postIMS_pre_areas)*0.05
logging.info(f"to_remove: {imzlabels[to_remove]}")
logging.info(f"to_keep: {imzlabels[~to_remove]}")
labs_to_remove = imzlabels[to_remove]
for lab in labs_to_remove:
    imzimg[imzregions == lab]=0

imzuqregs = np.unique(imzregions)[1:]
logging.info(f"Unique regions: {imzuqregs}")

if len(np.unique(np.array(postIMSregions))) == 1 and len(np.unique(np.array(imzuqregs))) == 1:
    logging.info("postIMS and IMS contain only one region, save data")
    df1 = pd.DataFrame({
        "imzregion": [imzuqregs[0]],
        "postIMSregion": [postIMSregions[0]]
    }).set_index("postIMSregion")


    postIMSxmins=[b[0] for b in postIMS_pre_bbox]
    postIMSymins=[b[1] for b in postIMS_pre_bbox]
    postIMSxmaxs=[b[2] for b in postIMS_pre_bbox]
    postIMSymaxs=[b[3] for b in postIMS_pre_bbox]
    
    df2 = pd.DataFrame({
        "postIMSregion": postIMSregions,
        "core_name": core_names,
        "postIMS_xmin": postIMSxmins,
        "postIMS_ymin": postIMSymins,
        "postIMS_xmax": postIMSxmaxs,
        "postIMS_ymax": postIMSymaxs,
        "project_name": imc_projects,
        "sample_name": sample_names.tolist()
    }).set_index("postIMSregion")

    dfout = df2.join(df1, on=["postIMSregion"])
    dfout.to_csv(output_table)

    open(snakemake.output["IMS_to_postIMS_matches_image"], 'a').close()
    logging.info("Finished")
    sys.exit(0)


logging.info("Scale IMZ image")
# rescale to postIMS resolution
wn = int(imzimg.shape[0]*stepsize/(resolution*rescale))
hn = int(imzimg.shape[1]*stepsize/(resolution*rescale))
logging.info(f"    Shape before: {imzimg.shape}")
imzimgres = cv2.resize(imzimg, (hn,wn), interpolation=cv2.INTER_NEAREST)
imzimgres[imzimgres>0] = 255 
logging.info(f"    Shape after: {imzimgres.shape}")
del imz

logging.info("Cut postIMS mask to global bounding box")
logging.info(f"    Shape before: {postIMSregin.shape}")
postIMSregincut = postIMSregin[global_bbox[0]:global_bbox[2],global_bbox[1]:global_bbox[3]]
logging.info(f"    Shape after: {postIMSregincut.shape}")

# check if all regions still present after cutting to global bounding box
wn = int(postIMSregincut.shape[0]/100)
hn = int(postIMSregincut.shape[1]/100)
tmp = cv2.resize(postIMSregincut, (hn,wn), interpolation=cv2.INTER_NEAREST)
tmpuq = np.unique(tmp)
logging.info(f"Unique regions: {tmpuq}")
assert(np.all(np.array([tt in tmpuq[1:] for tt in postIMSregions])))
del tmp,tmpuq

logging.info("Remove cores in postIMS that are missing in imz")
boolimg = ~np.isin(postIMSregincut, postIMSregions)

logging.info("Calculate connectedComponents")
# get centroids of imz regions
_,_,imzstats,imzcents = cv2.connectedComponentsWithStatsWithAlgorithm(imzimgres,connectivity=4,  ltype=cv2.CV_16U, ccltype=cv2.CCL_BBDT)
# remove background, extract area
imzarea = imzstats[1:,4]

imzxs = imzstats[:,1]
imzys = imzstats[:,0]
imzheights = imzstats[:,2]
imzwidths = imzstats[:,3]

imz_bbox = list()
for r in range(1,imzstats.shape[0]):
    imz_bbox.append([imzxs[r],imzys[r],imzxs[r]+imzwidths[r], imzys[r]+imzheights[r]])

# remove very small test regions
too_small = imzarea < np.mean(imzarea) - 5*np.std(imzarea)
imzcents=np.flip(imzcents[1:,:],axis=1)[~too_small,:]

logging.info(f"\tCentroids of IMZ: {imzcents}")



# find good initial translation for registration:
# loop through different x and y shifts
# for each find nearest neighboring centroid from other modality
# register point clouds and get maximum distance to nearest neighbor
# the lowest maximum distance should be a good starting point for registration
def find_approx_init_translation(imzcents, picents):
    if (picents.shape[0]==1) and (imzcents.shape[0]==1):
        xy_init_shift = -np.array([imzcents[0,0]-picents[0,0]+global_bbox[0],imzcents[0,1]-picents[0,1]+global_bbox[1]])
        return xy_init_shift, xy_init_shift, (1, np.eye(2), np.array([0,0]))
    

    # one postIMS region, multiple ims regions
    if np.unique(picents, axis=0).shape[0] == 1:
        tmpostIMSregincut = postIMSregin[postIMS_pre_bbox[0][0]:postIMS_pre_bbox[0][2],postIMS_pre_bbox[0][1]:postIMS_pre_bbox[0][3]]
        tmpimzimgrescut = [imzimgres[bb[0]:bb[2],bb[1]:bb[3]] for bb in imz_bbox]

        dists = list()
        for i in range(len(tmpimzimgrescut)):
            d1 = cv2.matchShapes(tmpostIMSregincut,tmpimzimgrescut[i],cv2.CONTOURS_MATCH_I2,0)
            dists.append(d1)
        min_ind = np.argmin(dists)

        xy_init_shift = -np.array([imzcents[min_ind,0]-picents[0,0]+global_bbox[0],imzcents[min_ind,1]-picents[0,1]+global_bbox[1]])
        return xy_init_shift, xy_init_shift, (1, np.eye(2), np.array([0,0]))

    max_dists = list()
    xrange = np.abs(imzimgres.shape[0]-postIMSregin.shape[0])
    yrange = np.abs(imzimgres.shape[1]-postIMSregin.shape[1])
    xshifts = list(range(-xrange,xrange,np.round(50/(resolution*rescale)).astype(np.uint))) + [xrange]
    yshifts = list(range(-yrange,yrange,np.round(50/(resolution*rescale)).astype(np.uint))) + [yrange]
    combs = np.array(np.meshgrid(np.array(xshifts),np.array(yshifts))).T.reshape(-1,2)
    for i in range(combs.shape[0]):
        kdt = KDTree(imzcents, leaf_size=30, metric='euclidean')
        distances, indices = kdt.query(picents+np.array(combs[i,:]), k=1, return_distance=True)
        indices = [ni[0] for ni in indices]
        max_dists.append(np.max(distances))
    
    # do precise registration of points only on 200 combinations with lowest max distances
    topn = 200 if combs.shape[0]>200 else combs.shape[0]
    ind = np.argpartition(max_dists, -topn)[:topn]
    combs_red = combs[ind,:]
    max_dists_red = list()
    reg_params_ls = list()
    keep_ls = list()
    for i in range(combs_red.shape[0]):
        kdt = KDTree(imzcents, leaf_size=30, metric='euclidean')
        distances, indices = kdt.query(picents+np.array(combs_red[i,:]), k=1, return_distance=True)
        indices = [ni[0] for ni in indices]
        # if (picents.shape[0]==1) and (imzcents.shape[0]==1):
        #     max_dists_red.append(np.max(distances))
        # else:
        reg = pycpd.RigidRegistration(X=imzcents[indices,:], Y=picents+np.array(combs_red[i,:]), w=0, s=1)
        TY, (s_reg, R_reg, t_reg) = reg.register()
        kdt = KDTree(imzcents, leaf_size=30, metric='euclidean')
        distances, indices = kdt.query(TY, k=1, return_distance=True)
        max_dists_red.append(np.max(distances))
        reg_params_ls.append((s_reg, R_reg, t_reg))
        keep_ls.append(s_reg > 0.9 and s_reg < 1.1 and np.all(np.abs(R_reg-np.eye(2))<0.1))

    subset_inds = np.arange(combs_red.shape[0])[np.array(keep_ls)]
    if subset_inds.shape[0]==0:
        xy_init_shift = -np.array([imzcents[0,0]-picents[0,0]+global_bbox[0],imzcents[0,1]-picents[0,1]+global_bbox[1]])
        return xy_init_shift, 0 , (1, np.eye(2), np.array([0,0]))

    min_ind = subset_inds[np.argmin(np.array(max_dists_red)[subset_inds])]
    logging.info(f"min_ind: {min_ind}")
    (s_reg, R_reg, t_reg) = reg_params_ls[min_ind]
    logging.info(f"combs_red shape: {combs_red.shape}")
    logging.info(f"combs_red shape: {combs_red[min_ind,:].shape}")
    logging.info(f"combs_red shape: {combs_red[min_ind,:]}")
    xy_init_shift = t_reg+np.array(combs_red[min_ind,:])
    return xy_init_shift, np.min(max_dists_red), (s_reg, R_reg, t_reg)

logging.info(f"Inital Matching")
logging.info(f"picents: {picents}")
logging.info(f"imzcents: {imzcents}")
xy_init_shift, max_dist, (s_reg, R_reg, t_reg) = find_approx_init_translation(imzcents, picents)

kdt = KDTree(imzcents, leaf_size=30, metric='euclidean')
distances, indices = kdt.query(picents+np.array(xy_init_shift), k=1, return_distance=True)
max_dist_test = np.max(distances)

logging.info(f"\tInital translation: {xy_init_shift}")
logging.info(f"\tMax distance: {max_dist}")
logging.info(f"\tMax distance: {max_dist_test}")
logging.info(f"\tRigid registration: {s_reg}, {R_reg}, {t_reg}")

if (picents.shape[0]==1) and (imzcents.shape[0]==1):
    init_trans = np.round(xy_init_shift).astype(int)
elif np.unique(picents, axis=0).shape[0] == 1:
    init_trans = np.round(xy_init_shift).astype(int)
else:
    kdt = KDTree(imzcents, leaf_size=30, metric='euclidean')
    distances, indices = kdt.query(picents+xy_init_shift, k=1, return_distance=True)
    logging.info(f"Initial translation: {xy_init_shift}")
    logging.info(f"Initial distances: {distances}")
    indices = [ni[0] for ni in indices]
    reg = pycpd.RigidRegistration(X=imzcents[indices,:], Y=picents+xy_init_shift, w=0, s=1)
    TY, (s_reg, R_reg, t_reg) = reg.register()
    t_reg = t_reg+xy_init_shift
    logging.info(f"Rigid registration: {s_reg}, {R_reg}, {t_reg}")
    # actual initial transform for registration
    init_trans = np.round(t_reg+np.array([global_bbox[0],global_bbox[1]])).astype(int)

logging.info(f"Initial translation: {init_trans}")


logging.info("Register postIMS to imz")
# function used in registration
def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.5f} "
        + f": {method.GetOptimizerPosition()}"
    )

# setup sitk images
tmp_downscale_factor = 25

wn = int(imzimgres.shape[0]/tmp_downscale_factor)
hn = int(imzimgres.shape[1]/tmp_downscale_factor)
fixed_np = cv2.resize(imzimgres, (hn,wn), interpolation=cv2.INTER_NEAREST)
fixed_np = fixed_np.astype(np.float32)

logging.info(f"global_bbox: {global_bbox}")
moving_np = readimage_crop(postIMSr_file, global_bbox)
wn = int(moving_np.shape[0]/tmp_downscale_factor)
hn = int(moving_np.shape[1]/tmp_downscale_factor)
tmpoutimg = cv2.resize(moving_np, (hn,wn), interpolation=cv2.INTER_NEAREST)
logging.info(f"counts: {np.unique(tmpoutimg, return_counts=True)}")
tmpoutimg = cv2.normalize(tmpoutimg, 0, 255, cv2.NORM_MINMAX)
logging.info(f"counts: {np.unique(tmpoutimg, return_counts=True)}")
saveimage_tile(tmpoutimg, snakemake.output["IMS_to_postIMS_matches_image"].replace(".ome.tiff","_postIMS_pre.ome.tiff"), 1)

moving_np[boolimg] = 0
wn = int(moving_np.shape[0]/tmp_downscale_factor)
hn = int(moving_np.shape[1]/tmp_downscale_factor)
moving_np = cv2.resize(moving_np, (hn,wn), interpolation=cv2.INTER_NEAREST)
tmpoutimg = cv2.normalize(moving_np, 0, 255, cv2.NORM_MINMAX)
moving_np = moving_np.astype(np.float32)
logging.info(f"counts: {np.unique(tmpoutimg, return_counts=True)}")
saveimage_tile(tmpoutimg, snakemake.output["IMS_to_postIMS_matches_image"].replace(".ome.tiff","_postIMS_filt.ome.tiff"), 1)

del boolimg

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].imshow(fixed_np)
# ax[0].set_title("IMS")
# ax[1].imshow(moving_np)
# ax[1].set_title("postIMS")
# plt.show()


logging.info(f"fixed image dimensions: ")
# def register_postIMS_to_IMS(ims: np.ndarray, postIMS: np.ndarray):
fixed = sitk.GetImageFromArray(fixed_np)
moving = sitk.GetImageFromArray(moving_np)
# del fixed_np,moving_np

# initial transformation
init_transform = sitk.AffineTransform(2)
init_transform.SetTranslation((-init_trans[[1,0]].astype(np.double))/tmp_downscale_factor)

# setup registration
R = sitk.ImageRegistrationMethod()
R.SetMetricAsCorrelation()
R.SetMetricSamplingStrategy(R.RANDOM)
R.SetMetricSamplingPercentage(0.01, seed=1234)
R.SetInterpolator(sitk.sitkLinear)
R.SetOptimizerAsRegularStepGradientDescent(
    minStep=1, learningRate=1.0, numberOfIterations=1000
)
R.SetOptimizerScalesFromPhysicalShift()
R.SetInitialTransform(init_transform)
R.SetInterpolator(sitk.sitkLinear)

R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

# run registration
transform = R.Execute(fixed, moving)

logging.info(f"Final translation: {transform.GetTranslation()}")
# transform expanded, labeled mask
resampler = sitk.ResampleImageFilter()
resampler.SetTransform(transform)
wn = int(imzimgres.shape[0]/tmp_downscale_factor)
hn = int(imzimgres.shape[1]/tmp_downscale_factor)
fixed_np = cv2.resize(imzimgres, (hn,wn), interpolation=cv2.INTER_NEAREST)
resampler.SetReferenceImage(sitk.GetImageFromArray(fixed_np))
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
wn = int((global_bbox[2]-global_bbox[0])/tmp_downscale_factor)
hn = int((global_bbox[3]-global_bbox[1])/tmp_downscale_factor)
tmp1 = cv2.resize(postIMSregincut, (hn,wn), interpolation=cv2.INTER_NEAREST)
tmp1 = resampler.Execute(sitk.GetImageFromArray(tmp1))
postIMSro_trans = sitk.GetArrayFromImage(tmp1)
del tmp1, resampler


logging.info("Save Image")
# save matches image

tmpoutimg = (((postIMSro_trans>0).astype(int)-(fixed_np>0).astype(int))+1).astype(np.uint8)
logging.info(f"counts: {np.unique(tmpoutimg, return_counts=True)}")
tmpoutimg = cv2.normalize(tmpoutimg,tmpoutimg, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
logging.info(f"counts: {np.unique(tmpoutimg, return_counts=True)}")
saveimage_tile(tmpoutimg, snakemake.output["IMS_to_postIMS_matches_image"], 1)

tmpoutimg = (postIMSro_trans>0).astype(np.uint8)
logging.info(f"counts: {np.unique(tmpoutimg, return_counts=True)}")
tmpoutimg = cv2.normalize(tmpoutimg,tmpoutimg, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
logging.info(f"counts: {np.unique(tmpoutimg, return_counts=True)}")
saveimage_tile(tmpoutimg, snakemake.output["IMS_to_postIMS_matches_image"].replace(".ome.tiff","_postIMS_resampled.ome.tiff"), 1)

tmpoutimg = (moving_np>0).astype(np.uint8)
logging.info(f"counts: {np.unique(tmpoutimg, return_counts=True)}")
tmpoutimg = cv2.normalize(tmpoutimg,tmpoutimg, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
logging.info(f"counts: {np.unique(tmpoutimg, return_counts=True)}")
saveimage_tile(tmpoutimg, snakemake.output["IMS_to_postIMS_matches_image"].replace(".ome.tiff","_postIMS.ome.tiff"), 1)

tmpoutimg = (fixed_np>0).astype(np.uint8)
logging.info(f"counts: {np.unique(tmpoutimg, return_counts=True)}")
tmpoutimg = cv2.normalize(tmpoutimg,tmpoutimg, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
logging.info(f"counts: {np.unique(tmpoutimg, return_counts=True)}")
saveimage_tile(tmpoutimg, snakemake.output["IMS_to_postIMS_matches_image"].replace(".ome.tiff","_IMS.ome.tiff"), 1)

del fixed_np


logging.info("Find matching regions between postIMS and imz")
wn = int(imzregions.shape[0])
hn = int(imzregions.shape[1])
postIMSro_trans_downscaled = cv2.resize(postIMSro_trans, (hn,wn), interpolation=cv2.INTER_NEAREST)
observedpostIMSregion_match = list()
# for imz regions get matching postIMS core
for regionimz in imzuqregs:
    t1=imzregions==regionimz
    overlaps = np.unique(postIMSro_trans_downscaled[t1], return_counts=True)
    overlaps = (overlaps[0][overlaps[0]!=0],overlaps[1][overlaps[0]!=0])
    if len(overlaps[0])>0:
        observedpostIMSregion_match.append(overlaps[0][overlaps[1] == np.max(overlaps[1])][0])
    else:
        observedpostIMSregion_match.append(np.nan)

observedpostIMSregion_match = np.array(observedpostIMSregion_match)
is_nan = np.isnan(observedpostIMSregion_match)
imzuqregs = imzuqregs[np.logical_not(is_nan)]
observedpostIMSregion_match = observedpostIMSregion_match[np.logical_not(is_nan)]

logging.info("Save data")
df1 = pd.DataFrame({
    "imzregion": imzuqregs,
    "postIMSregion": observedpostIMSregion_match
}).set_index("postIMSregion")

postIMSxmins=[b[0] for b in postIMS_pre_bbox]
postIMSymins=[b[1] for b in postIMS_pre_bbox]
postIMSxmaxs=[b[2] for b in postIMS_pre_bbox]
postIMSymaxs=[b[3] for b in postIMS_pre_bbox]

df2 = pd.DataFrame({
    "postIMSregion": postIMSregions,
    "core_name": core_names,
    "postIMS_xmin": postIMSxmins,
    "postIMS_ymin": postIMSymins,
    "postIMS_xmax": postIMSxmaxs,
    "postIMS_ymax": postIMSymaxs,
    "project_name": imc_projects,
    "sample_name": sample_names.tolist()
}).set_index("postIMSregion")

dfout = df2.join(df1, on=["postIMSregion"])
dfout.to_csv(output_table)

logging.info("Finished")
