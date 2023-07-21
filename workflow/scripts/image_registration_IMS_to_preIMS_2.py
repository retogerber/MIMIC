import pandas as pd
import napari
import pycpd 
import SimpleITK as sitk
import napari_imsmicrolink
from napari_imsmicrolink.utils.json import NpEncoder
import skimage
import numpy as np
import json
from image_registration_IMS_to_preIMS_utils import readimage_crop, prepare_image_for_sam, create_ring_mask, composite2affine, saveimage_tile
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
# rotation_imz = 180
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
# postIMS_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/postIMS/test_split_ims_postIMS_reduced.ome.tiff"
# postIMS_file = "/home/retger/Downloads/cirrhosis_TMA_postIMS_reduced.ome.tiff"
# postIMS_file = "/home/retger/Downloads/Lipid_TMA_3781_postIMS_reduced.ome.tiff"
postIMS_file = snakemake.input["postIMS_downscaled"]
# postIMSr_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/postIMS/test_split_ims_postIMS_reduced_mask.ome.tiff"
# postIMSr_file = "/home/retger/Downloads/cirrhosis_TMA_postIMS_reduced_mask.ome.tiff"
# postIMSr_file = "/home/retger/Downloads/Lipid_TMA_3781_postIMS_reduced_mask.ome.tiff"
postIMSr_file = snakemake.input["postIMSmask_downscaled"]
# imzmlfile = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMS/IMS_test_split_ims_1.imzML"
# imzmlfile = "/home/retger/Downloads/cirrhosis_TMA_IMS.imzML"
imzmlfile = snakemake.input["imzml"]
# imc_mask_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_001_transformed.ome.tiff"
# imc_mask_file = "/home/retger/Downloads/Cirrhosis-TMA-5_New_Detector_008_transformed.ome.tiff"
imc_mask_file = snakemake.input["IMCmask"]
# output_table = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_ims/data/IMS/test_split_ims_test_split_ims_1_IMS_to_postIMS_matches.csv"
# output_table = "/home/retger/Downloads/cirrhosis_TMA_cirrhosis_TMA_IMS_IMS_to_postIMS_matches.csv"
output_table = snakemake.input["IMS_to_postIMS_matches"]

# ims_to_postIMS_regerror = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_001_IMS_to_postIMS_reg_auto_metrics.json"
ims_to_postIMS_regerror = snakemake.output["IMS_to_postIMS_error"]
ims_to_postIMS_regerror_image = snakemake.output["IMS_to_postIMS_error_image"]

# ims_to_postIMS_matching_points_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_001_IMS_to_postIMS_reg_auto_matching_points.png"

# coordsfile_out = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMS/postIMS_to_IMS_test_split_pre-IMSML-coords.h5"
coordsfile_out = snakemake.output["imsml_coords_fp"]
output_dir = os.path.dirname(coordsfile_out)



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


logging.info("Read postIMS region bounding box")
# read crop bbox
dfmeta = pd.read_csv(output_table)
imc_samplename = os.path.splitext(os.path.splitext(os.path.split(imc_mask_file)[1])[0])[0].replace("_transformed","")
# imc_project = "cirrhosis_TMA"
# imc_project="test_split_ims"
imc_project = os.path.split(os.path.split(os.path.split(os.path.split(imc_mask_file)[0])[0])[0])[1]

project_name = "postIMS_to_IMS_"+imc_project+"_"+imc_samplename


inds_arr = np.logical_and(dfmeta["project_name"] == imc_project, dfmeta["sample_name"] == imc_samplename)
xmin = dfmeta[inds_arr]["postIMS_xmin"].tolist()[0]
ymin = dfmeta[inds_arr]["postIMS_ymin"].tolist()[0]
xmax = dfmeta[inds_arr]["postIMS_xmax"].tolist()[0]
ymax = dfmeta[inds_arr]["postIMS_ymax"].tolist()[0]
# needed:
regionimz = dfmeta[inds_arr]["imzregion"].tolist()[0]


logging.info("Read cropped postIMS")
# xmin = postIMSxmins[0]
# ymin = postIMSymins[0]
# xmax = postIMSxmaxs[0]
# ymax = postIMSymaxs[0]
# subset mask
postIMScut = readimage_crop(postIMS_file, [xmin, ymin, xmax, ymax])
postIMScut = prepare_image_for_sam(postIMScut, resolution)
postIMSmpre = skimage.filters.median(postIMScut, skimage.morphology.disk( np.floor(((stepsize-pixelsize)/resolution)/3)))

logging.info("Read cropped postIMS mask")
postIMSrcut = readimage_crop(postIMSr_file, [xmin, ymin, xmax, ymax])
postIMSringmask = create_ring_mask(postIMSrcut, (1/resolution)*stepsize*2, (1/resolution)*stepsize*4)
postIMSoutermask = skimage.morphology.isotropic_dilation(postIMSrcut, (1/resolution)*stepsize*2)

# subset and filter postIMS image
# postIMSm = postIMSmpre.copy()
# postIMSm[np.logical_not(postIMSringmask)] = 0

kersize = int(stepsize/2)
kersize = kersize-1 if kersize%2==0 else kersize
kernel = np.zeros((kersize,kersize))
kernel[int((kersize-1)/2),:]=1
kernel[:,int((kersize-1)/2)]=1

tmp1 = skimage.filters.rank.threshold(postIMSmpre, skimage.morphology.disk(kersize))
tmp2 = skimage.filters.rank.mean(tmp1*255, kernel)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(nrows=1, ncols=3)
# ax[0].imshow(postIMScut[250:500,250:500],cmap='gray')
# ax[0].set_title("postIMS")
# ax[1].imshow(tmp1[250:500,250:500],cmap='gray')
# ax[1].set_title("Filtered")
# ax[2].imshow(tmp2[250:500,250:500],cmap='gray')
# ax[2].set_title("Filtered")
# plt.show()

postIMSm = tmp2.copy()
postIMSm[np.logical_not(postIMSringmask)] = 0



logging.info("Find best threshold for IMS laser ablation marks detection")
from sklearn.neighbors import KDTree
def points_from_mask(
        mask: np.ndarray, 
        pixelsize: np.double, 
        resolution: np.double,
        stepsize: np.double):
    
    # filter detected regions to obtain ablation marks
    labs = skimage.measure.label(mask.astype(np.double))
    regs = skimage.measure.regionprops(labs)
    # filter by area (between 0.5 and 1 max pixel area)
    areas = np.asarray([c.area for c in regs])
    area_range = [(pixelsize/(2*resolution))**2,(pixelsize/resolution)**2]
    inran = np.logical_and(areas > area_range[0], areas < area_range[1])
    regsred = np.asarray(regs)[inran]
    cents = np.asarray([r.centroid for r in regsred])

    axis_ratio = np.asarray([c.axis_major_length/c.axis_minor_length for c in regsred])
    inran2 = axis_ratio < 2
    regsred = np.asarray(regsred)[inran2]
    if not isinstance(regsred, np.ndarray) or len(regsred)<6:
        return np.zeros((0,2))
    cents = np.asarray([r.centroid for r in regsred])

    # to IMS scale
    centsred = cents/((1/resolution)*stepsize)
    if centsred.shape[0]<6:
        return centsred
    # filter according to distance to nearest neighbors,
    # expected for a grid are distances close to 1
    kdt = KDTree(centsred, leaf_size=30, metric='euclidean')
    distances, indices = kdt.query(centsred, k=5, return_distance=True)
    distances[distances==0] = np.nan
    distances[np.logical_or(distances<0.75, distances > 1.25)] = np.nan
    all_nan = np.sum(np.isnan(distances),axis=1) == 5
    dists = np.nanmean(distances[np.logical_not(all_nan),:], axis=1)
    point_to_keep = np.logical_and(dists>0.95, dists < 1.05)

    centsred = centsred[np.logical_not(all_nan),:][point_to_keep,:]
    return centsred

# find best threshold by maximizing number of points that fullfill criteria
# grid search
# broad steps
thresholds = list(range(127,250,10))
n_points = []
for th in thresholds:
    # threshold
    postIMSmb = postIMSm>th
    centsred = points_from_mask(postIMSmb, pixelsize=pixelsize, resolution=resolution, stepsize=stepsize)
    n_points.append(centsred.shape[0])

threshold = np.asarray(thresholds)[n_points == np.max(n_points)][0]
# fine steps
thresholds = list(range(threshold-9,threshold+9))
n_points = []
for th in thresholds:
    # threshold
    postIMSmb = postIMSm>th
    centsred = points_from_mask(postIMSmb, pixelsize=pixelsize, resolution=resolution, stepsize=stepsize)
    n_points.append(centsred.shape[0])

threshold = np.asarray(thresholds)[n_points == np.max(n_points)][0]

logging.info("Max number of IMS pixels detected: "+str(np.max(n_points)))
logging.info("Corresponding threshold: "+str(threshold))

logging.info("Apply threshold")
postIMSmb = postIMSm>threshold
centsred = points_from_mask(postIMSmb, pixelsize=pixelsize, resolution=resolution, stepsize=stepsize)
# plt.imshow(postIMSmb)
# plt.scatter(centsred[:,1]*stepsize,centsred[:,0]*stepsize)
# plt.show()
logging.info("Create IMS coordinates")
# create coordsmatrices for IMS
indmatx = np.zeros(imzimg.shape)
for i in range(imzimg.shape[0]):
    indmatx[i,:] = list(range(imzimg.shape[1]))
indmatx = indmatx.astype(np.uint32)
indmaty = np.zeros(imzimg.shape)
for i in range(imzimg.shape[1]):
    indmaty[:,i] = list(range(imzimg.shape[0]))
indmaty = indmaty.astype(np.uint32)

# subset region for IMS
xminimz, yminimz, xmaximz, ymaximz = skimage.measure.regionprops((imzregions == regionimz).astype(np.uint8))[0].bbox

imzringmask = create_ring_mask(imzimg[xminimz:xmaximz,yminimz:ymaximz], 0, 4)


# create coordinates for registration
imzxcoords = indmatx[xminimz:xmaximz,yminimz:ymaximz][imzringmask]
imzycoords = indmaty[xminimz:xmaximz,yminimz:ymaximz][imzringmask]
imzcoords = np.stack([imzycoords, imzxcoords],axis=1)

# reference coordinates, actually in data
imzrefcoords = np.stack([imz.y_coords_min,imz.x_coords_min],axis=1)
# rotate IMS coordinates 
if rotation_imz in [-180, 180]:
    rotmat = np.asarray([[-1, 0], [0, -1]])
elif rotation_imz in [90, -270]:
    rotmat = np.asarray([[0, 1], [-1, 0]])
elif rotation_imz in [-90, 270]:
    rotmat = np.asarray([[0, -1], [1, 0]])
else:
    rotmat = np.asarray([[1, 0], [0, 1]])
center_point=np.max(imzrefcoords,axis=0)/2
imzrefcoords = np.dot(rotmat, (imzrefcoords - center_point).T).T + center_point

logging.info("Filter IMS coordinates")
# filter for coordinates that are in data
in_ref = []
for i in range(imzcoords.shape[0]):
    in_ref.append(np.any(np.logical_and(imzcoords[i,0] == imzrefcoords[:,0],imzcoords[i,1] == imzrefcoords[:,1])))

in_ref = np.array(in_ref)
imzcoords = imzcoords[in_ref,:]

init_translation = -np.min(imzcoords,axis=0).astype(int)
logging.info("Register Points")
# register points using coherent point drift
# target/fixed: postIMS
# source/moving: IMS
reg = pycpd.RigidRegistration(X=centsred, Y=imzcoords+init_translation, w=0.001)
TY, (s_reg, R_reg, t_reg) = reg.register()

logging.info("Find matching points")
# eval matching points
kdt = KDTree(centsred, leaf_size=30, metric='euclidean')
imz_distances, indices = kdt.query(TY, k=1, return_distance=True)
imz_has_match = imz_distances.flatten()<0.75
kdt = KDTree(TY, leaf_size=30, metric='euclidean')
centsred_distances, indices = kdt.query(centsred, k=1, return_distance=True)
centsred_has_match = centsred_distances.flatten()<0.75

centsredfilt = centsred[centsred_has_match,:]
imzcoordsfilt = imzcoords[imz_has_match,:]

logging.info("Register points with matching points only")
# rerun with matching points only
reg = pycpd.RigidRegistration(X=centsredfilt, Y=imzcoordsfilt+init_translation, w=0.001)
TY, (s_reg, R_reg, t_reg) = reg.register()

# TY # transformed points
# R_reg # Rotation
# t_reg # translation
logging.info("Find matching points")
# eval matching points
kdt = KDTree(centsredfilt, leaf_size=30, metric='euclidean')
imz_distances, indices = kdt.query(TY, k=1, return_distance=True)
imz_has_match = imz_distances.flatten()<0.75
kdt = KDTree(TY, leaf_size=30, metric='euclidean')
centsred_distances, indices = kdt.query(centsredfilt, k=1, return_distance=True)
centsred_has_match = centsred_distances.flatten()<0.75

centsredfilt = centsredfilt[centsred_has_match,:]
imzcoordsfilt = imzcoordsfilt[imz_has_match,:]

logging.info("Register points with matching points only")
# rerun with matching points only
reg = pycpd.RigidRegistration(X=centsredfilt, Y=imzcoordsfilt+init_translation, w=0.001)
TY, (s_reg, R_reg, t_reg) = reg.register()

logging.info("Final pycpd registration:")
logging.info(f"Number of points IMS: {imzcoordsfilt.shape[0]}")
logging.info(f"Number of points postIMS: {centsredfilt.shape[0]}")
logging.info(f"Scaling: {s_reg}")
logging.info(f"Translation: {t_reg}")
logging.info(f"Rotation: {R_reg}")

t_reg = t_reg + init_translation
logging.info(f"Translation plus init translation: {t_reg}")
# plt.scatter(TY[:,0]*stepsize, TY[:,1]*stepsize,color="blue")
# plt.scatter(centsredfilt[:,0]*stepsize, centsredfilt[:,1]*stepsize,color="red")
# plt.imshow(postIMSm.T)
# plt.title("matching points")
# plt.show()
# plt.savefig(ims_to_postIMS_matching_points_file, dpi=500)



logging.info("Get mean error after registration")
kdt = KDTree(centsredfilt, leaf_size=30, metric='euclidean')
distances, indices = kdt.query(TY, k=1, return_distance=True)
mean_error = np.mean(distances)*stepsize/resolution
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
# vie.image_transformer.target_pts = centsredfilt
# vie.image_transformer.source_pts = imzcoordsfilt
for regi in imzuqregs[imzuqregs != regionimz]:
    vie.ims_pixel_map.delete_roi(roi_name = str(regi), remove_padding=False)
#     vie.ims_pixel_map.delete_roi(roi_name = str(regi), remove_padding=True)


logging.info("Create Transformation matrix")
# initial rotation of imz
tm1 = sitk.Euler2DTransform()
tm1.SetTranslation([0,0])
transparamtm1 = np.flip((np.asarray(imzimg.shape).astype(np.double))/2*stepsize/resolution-stepsize/resolution/2).astype(np.double)
tm1.SetCenter(transparamtm1)
tm1.SetMatrix(rotmat.flatten().astype(np.double))
logging.info("1. Transformation: Rotation")
logging.info(tm1.GetParameters())

# Translation because of postIMS crop
tm2 = sitk.TranslationTransform(2)
# yshift = ymin-(imzimg.shape[1]-ymaximz)*stepsize/resolution-stepsize/resolution
# yshift = ymin-stepsize/resolution
yshift = ymin
# xshift = xmin-(imzimg.shape[0]-xmaximz)*stepsize/resolution-stepsize/resolution
# xshift = xmin-stepsize/resolution
xshift = xmin
tm2.SetParameters(np.array([yshift,xshift]).astype(np.double))
logging.info("2. Transformation: Translation")
logging.info(tm2.GetParameters())

# Registration of points 
tm3 = sitk.Euler2DTransform()
# tm3.SetCenter(np.array([yshift+t_reg[1]*stepsize/resolution,xshift+t_reg[0]*stepsize/resolution]).astype(np.double))
# tm3.SetCenter(np.array([yshift+t_reg[1]*stepsize/resolution+stepsize/resolution/2,xshift+t_reg[0]*stepsize/resolution+stepsize/resolution/2]).astype(np.double))
tm3.SetCenter(np.array([yshift+stepsize/resolution/2,xshift+stepsize/resolution/2]).astype(np.double))
tm3.SetMatrix(R_reg.flatten().astype(np.double))
tm3.SetTranslation(np.flip(t_reg*stepsize/resolution))
logging.info("3. Transformation: Euler2D")
logging.info(tm3.GetParameters())


# combine transforms to single affine transform
tm = sitk.CompositeTransform(2)
tm.AddTransform(tm3)
tm.AddTransform(tm2)
tm.AddTransform(tm1)
tmfl = composite2affine(tm, [0,0])
logging.info("Combined Transformation: Affine")
logging.info(tmfl.GetParameters())

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


# # check match
# import matplotlib.pyplot as plt
postIMS = skimage.io.imread(postIMS_file)
for i in [-1,0,1]:
    for j in [-1,0,1]:
        postIMS[np.array(pmap_coord_data["y_micro_physical"].to_list()).astype(int)+i,np.array(pmap_coord_data["x_micro_physical"].to_list()).astype(int)+j,:] = [0,0,255]

image_crop = postIMS[xmin:xmax,ymin:ymax]

saveimage_tile(image_crop, ims_to_postIMS_regerror_image, resolution)
# skimage.io.imsave(ims_to_postIMS_regerror_image, image_crop)


# plt.imshow(image_crop)
# plt.show()

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