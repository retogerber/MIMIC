import json
import numpy as np
import sys,os
import logging, traceback
import json
import h5py
from sklearn.neighbors import KDTree
from image_registration_IMS_to_preIMS_utils import readimage_crop
import skimage
import cv2
import shapely
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


logging.info("Error of all points")
# postIMS_ablation_centroids_filename = "/home/retger/Downloads/test_images_ims_to_imc_workflow/Lipid_TMA_37819_042_postIMS_ablation_centroids.csv"
# postIMS_ablation_centroids_filename = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_001_postIMS_ablation_centroids.csv"
postIMS_ablation_centroids_filename = snakemake.input["postIMS_ablation_centroids"]
# metadata_to_save_filename = "/home/retger/Downloads/test_images_ims_to_imc_workflow/Lipid_TMA_37819_042_step1_metadata.json"
# metadata_to_save_filename = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/registration_metric/Cirrhosis-TMA-5_New_Detector_001_step1_metadata.json" 
metadata_to_save_filename = snakemake.input["metadata"]
# imsml_coords_fp = "/home/retger/Downloads/test_images_ims_to_imc_workflow/Lipid_TMA_3781_042-IMSML-coords.h5"
# imsml_coords_fp = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMS/postIMS_to_IMS_test_split_pre-Cirrhosis-TMA-5_New_Detector_001-IMSML-coords.h5"
# imsml_coords_fp = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMS/postIMS_to_IMS_test_split_pre-IMSML-coords.h5"
imsml_coords_fp = snakemake.input["imsml_coords_fp"]
#imc_mask_file = "/home/retger/Downloads/test_images_ims_to_imc_workflow/Lipid_TMA_37819_042_transformed.ome.tiff"
# imc_mask_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_001_transformed_on_postIMS.ome.tiff"
imc_mask_file = snakemake.input["IMCmask"]

# stepsize=10
# stepsize=30
stepsize = float(snakemake.params["IMS_pixelsize"])
# resolution_factor = 0.8
resolution_factor = float(snakemake.params["IMS_shrink_factor"])
# resolution = 0.22537
resolution = float(snakemake.params["IMC_pixelsize"])

logging.info(f"Stepsize: {str(stepsize)}")
logging.info(f"Resolution factor: {str(resolution_factor)}")
logging.info(f"Resolution: {str(resolution)}")

with open(metadata_to_save_filename, 'r') as fp:
    metadata = json.load(fp)
postIMS_bbox = metadata['postIMS_bbox']
# xmin, ymin, xmax, ymax = (np.array(postIMS_bbox)*resolution).astype(int)
xmin, ymin, xmax, ymax = postIMS_bbox
# postIMS_shape = (int(postIMS_bbox[2]*resolution)-int(postIMS_bbox[0]*resolution), int(postIMS_bbox[3]*resolution)-int(postIMS_bbox[1]*resolution))

centsred = np.loadtxt(postIMS_ablation_centroids_filename, delimiter=',')
centsredfilttrans = centsred*stepsize+np.array([xmin*resolution,ymin*resolution])
with h5py.File(imsml_coords_fp, "r") as f:
    print(f["xy_padded"][:])
    print(f["xy_original"][:])
    # if in imsmicrolink IMS was the target
    if "xy_micro_physical" in [key for key, val in f.items()]:
        xy_micro_physical = f["xy_micro_physical"][:]

        micro_x = xy_micro_physical[:,0]
        micro_y = xy_micro_physical[:,1]
    # if the microscopy image was the target
    else:
        padded = f["xy_padded"][:]
        
        micro_x = (padded[:, 0] * stepsize )
        micro_y = (padded[:, 1] * stepsize )
        xsub = np.logical_and(micro_x > ymin*resolution, micro_x < ymax*resolution)
        ysub = np.logical_and(micro_y > xmin*resolution, micro_y < xmax*resolution)
        sub = np.logical_and(xsub, ysub)
        micro_x = micro_x[sub]
        micro_y = micro_y[sub]

imscoords = np.vstack((micro_y,micro_x)).T
# import matplotlib.pyplot as plt
# plt.scatter(imscoords[:,0],imscoords[:,1],s=1)
# plt.scatter(centsredfilttrans[:,0],centsredfilttrans[:,1],s=1)
# plt.show()

kdt = KDTree(imscoords, leaf_size=30, metric='euclidean')
distances, all_indices = kdt.query(centsredfilttrans, k=1, return_distance=True)
np.mean(distances.flatten())


imcmask = readimage_crop(imc_mask_file, [int(xmin), int(ymin), int(xmax), int(ymax)])
if resolution != 1:
    wn = int(imcmask.shape[0]*resolution)
    hn = int(imcmask.shape[1]*resolution)
    imcmask = cv2.resize(imcmask, (hn,wn), interpolation=cv2.INTER_NEAREST_EXACT)
imcmaskch = skimage.morphology.convex_hull_image(imcmask>0)
contours,_ = cv2.findContours(imcmaskch.astype(np.uint8),cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours = np.squeeze(contours[0])
poly = shapely.geometry.Polygon(contours)
poly = poly.buffer(2)
import shapely.affinity
poly = shapely.affinity.translate(poly, xoff=xmin*resolution, yoff=ymin*resolution, zoff=0.0)
# import shapely.plotting
# shapely.plotting.plot_polygon(poly)
# plt.scatter(imscoords[:,0],imscoords[:,1],s=1)
# plt.scatter(centsredfilttrans[:,0],centsredfilttrans[:,1],s=1)
# plt.show()

# check number of points in mask for postIMS
tpls = [shapely.geometry.Point(centsredfilttrans[i,:]) for i in range(centsredfilttrans.shape[0])]
pcontsc = np.array([poly.contains(tpls[i]) for i in range(len(tpls))])
tpls = [shapely.geometry.Point(imscoords[i,:]) for i in range(imscoords.shape[0])]
pcontsi = np.array([poly.contains(tpls[i]) for i in range(len(tpls))])

logging.info("Get mean error after registration")
kdt = KDTree(imscoords, leaf_size=30, metric='euclidean')
distances, indices = kdt.query(centsredfilttrans, k=1, return_distance=True)
mean_error = np.mean(distances)
q95_error = np.quantile(distances,0.95)
q75_error = np.quantile(distances,0.75)
q50_error = np.quantile(distances,0.5)
q25_error = np.quantile(distances,0.25)
q05_error = np.quantile(distances,0.05)

logging.info(f"Mean Error: {str(mean_error)}")
logging.info(f"Number of points: {str(len(distances))} / {str(len(imscoords))}")
logging.info(f"Quantiles: \n\tQ05: {str(q05_error)}\n\tQ25: {str(q25_error)}\n\tQ50: {str(q50_error)}\n\tQ75: {str(q75_error)}\n\tQ95: {str(q95_error)}")

imzcoordstransformedinmask = imscoords[pcontsi]
centsredinmask = centsredfilttrans[pcontsc]
distances_inmask, indices = kdt.query(centsredinmask, k=1, return_distance=True)
mean_error_inmask = np.mean(distances_inmask)
q95_error_inmask = np.quantile(distances_inmask,0.95)
q75_error_inmask = np.quantile(distances_inmask,0.75)
q50_error_inmask = np.quantile(distances_inmask,0.5)
q25_error_inmask = np.quantile(distances_inmask,0.25)
q05_error_inmask = np.quantile(distances_inmask,0.05)

logging.info(f"Mean Error in IMC mask: {str(mean_error_inmask)}")
logging.info(f"Number of points in IMC mask: {str(len(distances_inmask))} / {str(len(imzcoordstransformedinmask))}")
logging.info(f"Quantiles: \n\tQ05: {str(q05_error_inmask)}\n\tQ25: {str(q25_error_inmask)}\n\tQ50: {str(q50_error_inmask)}\n\tQ75: {str(q75_error_inmask)}\n\tQ95: {str(q95_error_inmask)}")


# mean error
reg_measure_dic = {
    "IMS_to_postIMS_mean_error": f"{mean_error:1.4f}",
    "IMS_to_postIMS_quantile05_error": f"{q05_error:1.4f}",
    "IMS_to_postIMS_quantile25_error": f"{q25_error:1.4f}",
    "IMS_to_postIMS_quantile50_error": f"{q50_error:1.4f}",
    "IMS_to_postIMS_quantile75_error": f"{q75_error:1.4f}",
    "IMS_to_postIMS_quantile95_error": f"{q95_error:1.4f}",
    "IMS_to_postIMS_n_points": f"{len(distances):d}",
    "IMS_to_postIMS_n_points_total": f"{len(imscoords):d}",
    "IMS_to_postIMS_inmask_mean_error": f"{mean_error_inmask:1.4f}",
    "IMS_to_postIMS_inmask_quantile05_error": f"{q05_error_inmask:1.4f}",
    "IMS_to_postIMS_inmask_quantile25_error": f"{q25_error_inmask:1.4f}",
    "IMS_to_postIMS_inmask_quantile50_error": f"{q50_error_inmask:1.4f}",
    "IMS_to_postIMS_inmask_quantile75_error": f"{q75_error_inmask:1.4f}",
    "IMS_to_postIMS_inmask_quantile95_error": f"{q95_error_inmask:1.4f}",
    "IMS_to_postIMS_inmask_n_points": f"{len(distances_inmask):d}",
    "IMS_to_postIMS_inmask_n_points_total": f"{len(imzcoordstransformedinmask):d}",
    }


logging.info("Error of landmarks only")
# transform_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMS/postIMS_to_IMS_test_combined-IMSML-meta.json"
# transform_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMS/postIMS_to_IMS_test_combined-IMSML-meta.json"
# transform_file = "/home/retger/Downloads/Lipid_TMA_3781_042-IMSML-meta.json"
# transform_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMS/postIMS_to_IMS_test_split_pre-Cirrhosis-TMA-5_New_Detector_002-IMSML-meta.json"
transform_file = snakemake.input["imsmicrolink_meta"]
logging.info("Read transform json file")
j0 = json.load(open(transform_file, "r"))
IMS = np.asarray(j0["IMS pixel map points (xy, microns)"])
postIMS = np.asarray(j0["PAQ microscopy points (xy, microns)"])

t1 = np.asarray(j0["Affine transformation matrix (xy,microns)"])
t2 = np.asarray(j0["Inverse Affine transformation matrix (xy,microns)"])
t3 = t1.copy()
# t3[:2,:2] = np.array([[1-(t3[0,0]-1),-t3[1,0]],[-t3[0,1],1-(t3[1,1]-1)]])
# t3[:2,2]=-t1[:2,2]
t3[:2,:2] = np.array([[t3[0,0],t3[1,0]],[t3[0,1],t3[1,1]]])
t3[:2,2]=t1[:2,2]
t4 = t2.copy()
t4[:2,:2] = np.array([[1-(t4[0,0]-1),-t4[1,0]],[-t4[0,1],1-(t4[1,1]-1)]])
t4[:2,2]=-t2[:2,2]


def mean_error(A, B, T):
    A_trans = np.matmul(A, T[:2,:2]) + T[:2,2]
    return np.mean(np.sqrt(np.sum((A_trans-B)**2,axis=1)))

logging.info("Apply transformations and calculate error")
e1 = mean_error(IMS, postIMS, t1)
e2 = mean_error(postIMS, IMS, t1)
e3 = mean_error(IMS, postIMS, t2)
e4 = mean_error(postIMS, IMS, t2)
e5 = mean_error(IMS, postIMS, t3)
e6 = mean_error(postIMS, IMS, t3)
e7 = mean_error(IMS, postIMS, t4)
e8 = mean_error(postIMS, IMS, t4)

ee = [e1, e2, e3, e4, e5, e6, e7, e8]

reg_measure_dic['IMS_to_postIMS_landmarks_only_mean_error'] = f"{np.min(ee):1.4f}"
reg_measure_dic['IMS_to_postIMS_n_landmarks'] = f"{IMS.shape[0]:d}"
logging.info("Save json")
json.dump(reg_measure_dic, open(snakemake.output["IMS_to_postIMS_error"],"w"))

logging.info("Finished")


