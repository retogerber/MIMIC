import pandas as pd
from sklearn.neighbors import KDTree
import pycpd
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
# stepsize = 10
stepsize = float(snakemake.params["IMS_pixelsize"])
# pixelsize = 8 
pixelsize = stepsize*float(snakemake.params["IMS_shrink_factor"])
# resolution = 1
resolution = float(snakemake.params["IMC_pixelsize"])
# rotation_imz = 0
rotation_imz = float(snakemake.params["IMS_rotation_angle"])
assert(rotation_imz in [-270,-180,-90,0,90,180,270])

# postIMSr_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/postIMS/test_split_ims_postIMS_reduced_mask.ome.tiff"
# postIMSr_file = "/home/retger/Downloads/cirrhosis_TMA_postIMS_reduced_mask.ome.tiff"
# postIMSr_file = "/home/retger/Downloads/Lipid_TMA_3781_postIMS_reduced_mask.ome.tiff"
postIMSr_file = snakemake.input["postIMSmask_downscaled"]
# imzmlfile = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMS/IMS_test_split_ims_2.imzML"
# imzmlfile = "/home/retger/Downloads/cirrhosis_TMA_IMS.imzML"
# imzmlfile = "/home/retger/Downloads/pos_mode_lipids_tma_02032023_imzml.imzML"
imzmlfile = snakemake.input["imzml"]
# imc_mask_files = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_002_transformed.ome.tiff"
# imc_mask_files = [f"/home/retger/Downloads/Cirrhosis-TMA-5_New_Detector_0{i}_transformed.ome.tiff" for i in ["01","02","03","04","05","06","07","08","09","11","12","13","14","15","16"]]
# imc_mask_files = imc_mask_files + [f"/home/retger/Downloads/Cirrhosis-TMA-5_01062022_0{i}_transformed.ome.tiff" for i in ["05","06","07","08","09"]]
# imc_mask_files = imc_mask_files + [f""/home/retger/Downloads/Cirrhosis_TMA_5_01262022_0{i}_transformed.ome.tiff" for i in ["01","02","03","04","05"]]
# imc_mask_files = ["/home/retger/Downloads/Lipid_TMA_37819_009_transformed.ome.tiff", "/home/retger/Downloads/Lipid_TMA_37819_025_transformed.ome.tiff", "/home/retger/Downloads/Lipid_TMA_37819_027_transformed.ome.tiff", "/home/retger/Downloads/Lipid_TMA_37819_029_transformed.ome.tiff", "/home/retger/Downloads/Lipid_TMA_37819_031_transformed.ome.tiff", "/home/retger/Downloads/Lipid_TMA_37819_033_transformed.ome.tiff", "/home/retger/Downloads/Lipid_TMA_37819_035_transformed.ome.tiff", "/home/retger/Downloads/Lipid_TMA_37819_037_transformed.ome.tiff", "/home/retger/Downloads/Lipid_TMA_37819_039_transformed.ome.tiff", "/home/retger/Downloads/Lipid_TMA_37819_041_transformed.ome.tiff", "/home/retger/Downloads/Lipid_TMA_37819_043_transformed.ome.tiff"]
# imc_mask_files = ["/home/retger/Downloads/Lipid_TMA_37819_012_transformed.ome.tiff", "/home/retger/Downloads/Lipid_TMA_37819_015_transformed.ome.tiff", "/home/retger/Downloads/Lipid_TMA_37819_017_transformed.ome.tiff", "/home/retger/Downloads/Lipid_TMA_37819_019_transformed.ome.tiff", "/home/retger/Downloads/Lipid_TMA_37819_021_transformed.ome.tiff", "/home/retger/Downloads/Lipid_TMA_37819_023_transformed.ome.tiff"]
imc_mask_files = snakemake.input["IMCmask"]
if isinstance(imc_mask_files, str):
    imc_mask_files = [imc_mask_files]
# sample_metadata = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/config/sample_metadata.csv"
# sample_metadata = "/home/retger/Downloads/sample_metadata.csv"
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
# imc_projects = ["cirrhosis_TMA"]*len(imc_samplenames)
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
# # outermask = skimage.morphology.isotropic_dilation(postIMSr, (1/resolution)*stepsize*2)
# postIMSlbs = skimage.measure.label(postIMSr.astype(np.uint8))
# postIMSlbs = postIMSlbs.astype(np.uint8)
# # outermask = skimage.morphology.dilation(postIMSlbs, skimage.morphology.disk((1/resolution)*stepsize*2))
# # rank maximum filter instead of dilation for speedup
# outermask = skimage.filters.rank.maximum(postIMSlbs, skimage.morphology.disk((1/resolution)*stepsize*2))
# import matplotlib.pyplot as plt
# plt.imshow(outermask)
# plt.show()
# # measure regions
# postIMSregin = skimage.measure.label(outermask.astype(np.uint8))

postIMSregin = skimage.measure.label(postIMSr.astype(np.uint8))

logging.info("Find IMC to postIMS overlap")
postIMSregions = list()
for bb in imcbboxls:
    tmpuqs = np.unique([postIMSregin[bb[0],bb[1]], postIMSregin[bb[0],bb[3]], postIMSregin[bb[2],bb[1]], postIMSregin[bb[2],bb[3]]])
    tmpuqs = tmpuqs[tmpuqs>0]
    assert(len(tmpuqs)==1)
    postIMSregions.append(tmpuqs[0])
del tmpuqs

regpops = skimage.measure.regionprops(postIMSregin)
postIMS_pre_bbox = [r.bbox for r in regpops]
postIMS_pre_labels = [r.label for r in regpops]
postIMS_pre_areas = [r.area for r in regpops]
del regpops

logging.info("Find global bounding box of cores")
tmpbool = np.array([p in np.array(postIMSregions) for p in postIMS_pre_labels])
postIMS_pre_bbox = np.array(postIMS_pre_bbox)[tmpbool]
postIMS_pre_labels = np.array(postIMS_pre_labels)[tmpbool]
postIMS_pre_areas = np.array(postIMS_pre_areas)[tmpbool]
global_bbox = [
    np.min([p[0] for p in postIMS_pre_bbox]),
    np.min([p[1] for p in postIMS_pre_bbox]),
    np.max([p[2] for p in postIMS_pre_bbox]),
    np.max([p[3] for p in postIMS_pre_bbox]),
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
imzareas = np.array([r.area for r in imzregpop]) * (stepsize*resolution)**2
to_remove = imzareas < np.min(postIMS_pre_areas)*0.5
labs_to_remove = imzlabels[to_remove]
for lab in labs_to_remove:
    imzimg[imzregions == lab]=0

imzuqregs = np.unique(imzregions)[1:]
# rescale to postIMS resolution
imzimgres = skimage.transform.rescale(imzimg, stepsize*resolution, preserve_range = True)   
imzimgres[imzimgres>0] = 255 
del imz

# imzimgres = skimage.transform.rotate(imzimgres,rotation_imz, preserve_range=True).astype(np.uint8)
imzimgres[imzimgres>0] = 255

logging.info("Cut postIMS mask to global bounding box")
postIMSrcut = postIMSr[global_bbox[0]:global_bbox[2],global_bbox[1]:global_bbox[3]]
postIMSregincut = postIMSregin[global_bbox[0]:global_bbox[2],global_bbox[1]:global_bbox[3]]

logging.info("Remove cores in postIMS that are missing in imz")
boolimgls = list()
for region in postIMSregions:
    boolimgls.append(postIMSregincut == region)
boolimg = np.sum(np.stack(boolimgls),axis=0).astype(bool)
del boolimgls
postIMSrcut[np.logical_not(boolimg)] = 0

# get centroids of imz regions
imzrps = skimage.measure.regionprops(skimage.measure.label(imzimgres))
imzarea = np.array([rp.area for rp in imzrps])
too_small = imzarea < np.mean(imzarea) - 5*np.std(imzarea)
imzrps = np.array(imzrps)[np.logical_not(too_small)]
imzcents = np.array([[rp.centroid[0],rp.centroid[1]] for rp in imzrps])
del imzrps, imzarea, too_small
logging.info(f"\tCentroids of IMZ: {imzcents}")

# get centroids of postIMS regions
tmpi = postIMSregincut
tmpi[np.logical_not(boolimg)] = 0
pirps = skimage.measure.regionprops(tmpi)
picents = np.array([[rp.centroid[0]+global_bbox[0],rp.centroid[1]+global_bbox[1]] for rp in pirps])
del tmpi, pirps
logging.info(f"\tCentroids of postIMS: {picents}")
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].imshow(imzimg)
# ax[0].set_title("IMS")
# ax[1].imshow(postIMSregincut)
# ax[1].set_title("postIMS")
# plt.show()


# find good initial translation for registration:
# loop through different x and y shifts
# for each find nearest neighboring centroid from other modality
# register point clouds and get maximum distance to nearest neighbor
# the lowest maximum distance should be a good starting point for registration
def find_approx_init_translation(imzcents, picents):
    if (picents.shape[0]==1) and (imzcents.shape[0]==1):
        xy_init_shift = -np.array([imzcents[0,0]-picents[0,0]+global_bbox[0],imzcents[0,1]-picents[0,1]+global_bbox[1]])
        return xy_init_shift, xy_init_shift

    max_dists = list()
    xrange = np.abs(imzimgres.shape[0]-postIMSregin.shape[0])
    yrange = np.abs(imzimgres.shape[1]-postIMSregin.shape[1])
    xshifts = list(range(-xrange,xrange,np.round(50/resolution).astype(np.uint))) + [xrange]
    yshifts = list(range(-yrange,yrange,np.round(50/resolution).astype(np.uint))) + [yrange]
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
    for i in range(combs_red.shape[0]):
        kdt = KDTree(imzcents, leaf_size=30, metric='euclidean')
        distances, indices = kdt.query(picents+np.array(combs_red[i,:]), k=1, return_distance=True)
        indices = [ni[0] for ni in indices]
        # if (picents.shape[0]==1) and (imzcents.shape[0]==1):
        #     max_dists_red.append(np.max(distances))
        # else:
        reg = pycpd.RigidRegistration(X=imzcents[indices,:], Y=picents, w=0)
        TY, (s_reg, R_reg, t_reg) = reg.register()
        kdt = KDTree(imzcents, leaf_size=30, metric='euclidean')
        distances, indices = kdt.query(TY, k=1, return_distance=True)
        max_dists_red.append(np.max(distances))

    xy_init_shift = combs_red[np.array(max_dists_red) == np.min(max_dists_red),:][0,:]
    return xy_init_shift, np.min(max_dists_red)

xy_init_shift, max_dist = find_approx_init_translation(imzcents, picents)

logging.info(f"\tInital translation: {xy_init_shift}")
logging.info(f"\tMax distance: {max_dist}")

# kdt = KDTree(imzcents, leaf_size=30, metric='euclidean')
# distances, indices = kdt.query(imzcents, k=2, return_distance=True)
# if max_dist > np.min(distances[:,1])/2:
#     inds = np.linspace(0,imzcents.shape[0]-1,imzcents.shape[0]).astype(int)
#     ndiff = imzcents.shape[0]-picents.shape[0]
#     outls = list()
#     for i in inds:
#         tmp_xy_init_shift, tmp_max_dist = find_approx_init_translation(imzcents[inds[inds!=i],:], picents)
#         outls.append([i, tmp_max_dist, tmp_xy_init_shift])
#     max_dists = [o[1] for o in outls]

# plt.scatter(picents[:,1], picents[:,0])
# plt.scatter(imzcents[inds[inds!=i],1], imzcents[inds[inds!=i],0], color="red")
# plt.show()



if (picents.shape[0]==1) and (imzcents.shape[0]==1):
    init_trans = np.round(xy_init_shift).astype(int)
else:
    kdt = KDTree(imzcents, leaf_size=30, metric='euclidean')
    distances, indices = kdt.query(picents+xy_init_shift, k=1, return_distance=True)
    indices = [ni[0] for ni in indices]
    reg = pycpd.RigidRegistration(X=imzcents[indices,:], Y=picents, w=0)
    TY, (s_reg, R_reg, t_reg) = reg.register()
    # actual initial transform for registration
    init_trans = np.round(t_reg+np.array([global_bbox[0],global_bbox[1]])).astype(int)

logging.info(f"Initial translation: {init_trans}")
# import matplotlib.pyplot as plt
# plt.scatter(picents[:,1]+xy_init_shift[1], picents[:,0]+xy_init_shift[0])
# plt.scatter(TY[:,1], TY[:,0])
# plt.scatter(imzcents[:,1], imzcents[:,0], color="red")
# plt.show()



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
moving_np = postIMSrcut.astype(np.float32)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].imshow(fixed_np)
# ax[0].set_title("IMS")
# ax[1].imshow(moving_np)
# ax[1].set_title("postIMS")
# plt.show()



# def register_postIMS_to_IMS(ims: np.ndarray, postIMS: np.ndarray):
fixed = sitk.GetImageFromArray(fixed_np)
moving = sitk.GetImageFromArray(moving_np)
del fixed_np,moving_np

# initial transformation
init_transform = sitk.AffineTransform(2)
init_transform.SetTranslation(-init_trans[[1,0]].astype(np.double))

# setup registration
R = sitk.ImageRegistrationMethod()
R.SetMetricAsCorrelation()
R.SetMetricSamplingStrategy(R.RANDOM)
R.SetMetricSamplingPercentage(0.01)
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
resampler.SetReferenceImage(fixed)
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
tmp1 = resampler.Execute(sitk.GetImageFromArray(postIMSregincut))
postIMSro_trans = sitk.GetArrayFromImage(tmp1)
del tmp1, resampler

# save matches image
saveimage_tile((normalize_image(((postIMSro_trans>0).astype(int)-(imzimgres>0).astype(int))+1)*255).astype(np.uint8), snakemake.output["IMS_to_postIMS_matches_image"], 1)

logging.info("Find matching regions between postIMS and imz")
postIMSro_trans_downscaled = skimage.transform.resize(postIMSro_trans, imzregions.shape, preserve_range=True, anti_aliasing=False).astype(np.uint8)
observedpostIMSregion_match = list()
# for imz region 1 get matching postIMS core
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

logging.info("Create bounding boxes for all cores")
postIMS_bboxls = list()
postIMSregpops = skimage.measure.regionprops(postIMSregin)
regpopslabs = np.asarray([t.label for t in postIMSregpops])
for i in range(len(observedpostIMSregion_match)):
    # select subset region for postIMS
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