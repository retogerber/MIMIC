import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
import numpy as np
import json
import tifffile
import re
import numpy as np
import cv2
from PIL import Image
from image_utils import readimage_crop, get_image_shape, subtract_postIMS_grid
from utils import setNThreads, snakeMakeMock
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.threads=32
    snakemake.params["input_spacing_1"] = 0.22537
    snakemake.params["input_spacing_2"] = 0.22537
    snakemake.params["input_spacing_IMC_location"] = 0.22537
    snakemake.params["output_spacing"] = 1
    snakemake.params["max_distance"] = 50
    snakemake.params["min_distance"] = 10
    snakemake.params["remove_postIMS_grid"] = True
    snakemake.input["microscopy_image_1"] = "results/cirrhosis_TMA/data/preIMS/cirrhosis_TMA_preIMS_transformed_on_postIMS.ome.tiff"
    snakemake.input["microscopy_image_2"] = "results/cirrhosis_TMA/data/postIMS/cirrhosis_TMA_postIMS.ome.tiff"
    snakemake.input["IMC_location"] = "results/cirrhosis_TMA/data/IMC_location/cirrhosis_TMA_IMC_mask_on_postIMS_E2.geojson"
    snakemake.input["sam_weights"] = "results/Misc/sam_vit_h_4b8939.pth"
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
import torch
# setting the number of threads results in stalling for some reason
# setNThreads(snakemake.threads)
from romatch import roma_outdoor

# params
input_spacing_1 = snakemake.params["input_spacing_1"]
input_spacing_2 = snakemake.params["input_spacing_2"]
input_spacing_IMC_location = snakemake.params["input_spacing_IMC_location"]
output_spacing = snakemake.params["output_spacing"]
# maximum assumed distance between corresponding points
dmax = snakemake.params["max_distance"]/output_spacing
# minimum distance between points on the same image 
dmin = snakemake.params["min_distance"]/output_spacing
max_landmarks=3000

# inputs
microscopy_file_1 = snakemake.input['microscopy_image_1']
microscopy_file_2 = snakemake.input['microscopy_image_2']
IMC_location=snakemake.input["IMC_location"]
if isinstance(IMC_location, list):
    IMC_location = IMC_location[0]

m = re.search("[a-zA-Z]*(?=.ome.tiff$)",os.path.basename(microscopy_file_1))
comparison_to = m.group(0)
comparison_from = os.path.basename(os.path.dirname(microscopy_file_1))
assert(comparison_to in ["preIMC","preIMS","postIMS"])
assert(comparison_from in ["postIMC","preIMC","preIMS"])

# outputs
microscopy_file_out_1 = snakemake.output['microscopy_image_out_1']
microscopy_file_out_2 = snakemake.output['microscopy_image_out_2']
matching_points_filename_out = snakemake.output['matching_points']

logging.info("core bounding box extraction")
IMC_geojson = json.load(open(IMC_location, "r"))
if isinstance(IMC_geojson,list):
    IMC_geojson=IMC_geojson[0]
boundary_points = np.array(IMC_geojson['geometry']['coordinates'])[0,:,:]
xmin=np.min(boundary_points[:,1])
xmax=np.max(boundary_points[:,1])
ymin=np.min(boundary_points[:,0])
ymax=np.max(boundary_points[:,0])

s1f = input_spacing_1/input_spacing_IMC_location
bb1 = [int(xmin/s1f-201/input_spacing_1),int(ymin/s1f-201/input_spacing_1),int(xmax/s1f+201/input_spacing_1),int(ymax/s1f+201/input_spacing_1)]
logging.info(f"bounding box whole image 1: {bb1}")

s2f = input_spacing_2/input_spacing_IMC_location
bb2 = [int(xmin/s2f-201/input_spacing_2),int(ymin/s2f-201/input_spacing_2),int(xmax/s2f+201/input_spacing_2),int(ymax/s2f+201/input_spacing_2)]
logging.info(f"bounding box whole image 2: {bb2}")

logging.info("load microscopy image 1")
microscopy_image_1 = readimage_crop(microscopy_file_1, bb1)
microscopy_image_1 = microscopy_image_1.astype("uint8")
wn = int(microscopy_image_1.shape[0]*(input_spacing_1/output_spacing))
hn = int(microscopy_image_1.shape[1]*(input_spacing_1/output_spacing))
microscopy_image_1 = cv2.resize(microscopy_image_1, (hn,wn), interpolation=cv2.INTER_AREA)
logging.info(f"microscopy image 1 shape: {microscopy_image_1.shape}")

logging.info("load microscopy image 2")

print(f"snakemake params remove_postIMS_grid: {snakemake.params['remove_postIMS_grid']}")
if snakemake.params["remove_postIMS_grid"]:
    m2full_shape = get_image_shape(microscopy_file_2)
    m2_expansion = 50
    bb3 = [int(bb2[0]-m2_expansion/input_spacing_1),int(bb2[1]-m2_expansion/input_spacing_1),int(bb2[2]+m2_expansion/input_spacing_1),int(bb2[3]+m2_expansion/input_spacing_1)]
    bb3[0] = bb3[0] if bb3[0]>=0 else 0
    bb3[1] = bb3[1] if bb3[1]>=0 else 0
    bb3[2] = bb3[2] if bb3[2]<=m2full_shape[0] else m2full_shape[0]
    bb3[3] = bb3[3] if bb3[3]<=m2full_shape[1] else m2full_shape[1]
    logging.info(f"bounding box mask whole image 2: {bb3}")
    microscopy_image_2 = readimage_crop(microscopy_file_2, bb3)
else:
    microscopy_image_2 = readimage_crop(microscopy_file_2, bb2)
microscopy_image_2 = microscopy_image_2.astype("uint8")
scale_factor_2 = input_spacing_2/output_spacing
wn = int(microscopy_image_2.shape[0]*scale_factor_2)
hn = int(microscopy_image_2.shape[1]*scale_factor_2)
microscopy_image_2 = cv2.resize(microscopy_image_2, (hn,wn), interpolation=cv2.INTER_AREA)
logging.info(f"microscopy image 2 shape: {microscopy_image_2.shape}")
if snakemake.params["remove_postIMS_grid"]:
    logging.info("remove postIMS grid")
    microscopy_image_2_proc = cv2.cvtColor(microscopy_image_2, cv2.COLOR_RGB2GRAY) 
    microscopy_image_2_proc = subtract_postIMS_grid(microscopy_image_2_proc)
    microscopy_image_2 = cv2.cvtColor(microscopy_image_2_proc, cv2.COLOR_GRAY2RGB)
    microscopy_image_2 = microscopy_image_2[m2_expansion:-m2_expansion,m2_expansion:-m2_expansion,:]


logging.info("to PIL")
img1 = Image.fromarray(microscopy_image_1)
img2 = Image.fromarray(microscopy_image_2)
W_A, H_A = img1.size
W_B, H_B = img2.size


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"device: {device}")
logging.info(f"setup RoMa model")
torch.manual_seed(1234)
roma_model = roma_outdoor(device=device, amp_dtype=torch.float32, coarse_res=(700,700))
roma_model.upsample_res = (1400,1400)

logging.info(f"match images")
# Match
warp, certainty = roma_model.match(img1, img2, device=device)
# Sample matches for estimation

logging.info(f"sample matches")
# function sample from: https://github.com/Parskatt/RoMa/blob/64f20c7ee67e7ea5bd1448c3e9468a8c5f2f06b9/romatch/models/matcher.py#L468
# fix error of using hardcoded half (which is not available on CPU)
# matches, certainty = roma_model.sample(warp.type(torch.float32), certainty.type(torch.float32))
logging.info(f"Max number of matches: {max_landmarks}")
if "threshold" in roma_model.sample_mode:
    upper_thresh = roma_model.sample_thresh
    certainty = certainty.clone()
    certainty[certainty > upper_thresh] = 1

matches, certainty = (
    warp.reshape(-1, 4),
    certainty.reshape(-1),
)

expansion_factor = 4
good_samples = torch.multinomial(certainty, 
                  num_samples = min(expansion_factor*max_landmarks, len(certainty)), 
                  replacement=False)
good_matches, good_certainty = matches[good_samples], certainty[good_samples]
scores = (-torch.cdist(good_matches, good_matches)**2/(2*0.1**2)).exp()
density = scores.sum(dim=-1)
p = 1 / (density+1)
p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
balanced_samples = torch.multinomial(p, 
                  num_samples = min(max_landmarks,len(good_certainty)), 
                  replacement=False)
matches =  good_matches[balanced_samples]
certainty = good_certainty[balanced_samples]
logging.info(f"Number of matches: {len(matches)}")

logging.info(f"convert to physical coordinates")
# Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

logging.info(f"estimate affine transformation")
src_pts = kptsA.cpu().numpy()
dst_pts = kptsB.cpu().numpy()
# based on: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
# and https://docs.opencv.org/4.x/d0/d74/md__build_4_x-contrib_docs-lin64_opencv_doc_tutorials_calib3d_usac.html
# Sampling: PROSAC
# Scoring: MAGSAC with low threshold
# Error metric: Sampson distance
# Degeneracy: DEGENSAC
# Local optimization: Graph-cut RANSAC
# Solver: Affine2D
# Graph creation: NEIGH_FLANN_RADIUS
params = cv2.UsacParams()
params.confidence = 0.99999
params.sampler = cv2.SAMPLING_PROSAC
params.score = cv2.SCORE_METHOD_MAGSAC
# params.score = cv2.SCORE_METHOD_RANSAC
params.maxIterations = 1000000
params.neighborsSearch = cv2.NEIGH_FLANN_RADIUS
if comparison_from == "preIMC" and comparison_to == "preIMS":
    # non-linear
    params.threshold = 25
else:
    # linear
    params.threshold = 10
params.loMethod = cv2.LOCAL_OPTIM_GC
params.loIterations = 100
# params.loSampleSize = 20 
M, mask = cv2.estimateAffine2D(src_pts, dst_pts, params)


logging.info(f"filter matches")
dists_real_all = np.sqrt(np.sum((src_pts-dst_pts)**2,axis=1))
src_pts_filt = src_pts[mask.ravel()==1]
dst_pts_filt = dst_pts[mask.ravel()==1]
dists_real = dists_real_all[mask.ravel()==1]
logging.info(f"number of matches: {len(src_pts_filt)}")

combined_output = np.hstack([src_pts,dst_pts,mask.ravel().reshape(-1,1),dists_real_all.reshape(-1,1)])
np.savetxt(matching_points_filename_out,combined_output,header=f"p1x,p1y,p2x,p2y,homography_mask,distance_physical",delimiter=',')

logging.info(f"save matching points")
arrowed_microscopy_image_1 = microscopy_image_1.copy()
for k in range(len(src_pts_filt)):
    arrowed_microscopy_image_1 = cv2.arrowedLine(arrowed_microscopy_image_1, pt1=src_pts_filt[k,:].astype(int), pt2=dst_pts_filt[k,:].astype(int), color=(255,255,255), thickness=int(4), tipLength=0.3, line_type=cv2.LINE_AA)
    arrowed_microscopy_image_1 = cv2.arrowedLine(arrowed_microscopy_image_1, pt1=src_pts_filt[k,:].astype(int), pt2=dst_pts_filt[k,:].astype(int), color=(0,0,255), thickness=int(2), tipLength=0.3, line_type=cv2.LINE_AA)
tifffile.imwrite(microscopy_file_out_1,arrowed_microscopy_image_1)

arrowed_microscopy_image_2 = microscopy_image_2.copy()
for k in range(len(dst_pts_filt)):
    arrowed_microscopy_image_2 = cv2.arrowedLine(arrowed_microscopy_image_2, pt1=dst_pts_filt[k,:].astype(int), pt2=src_pts_filt[k,:].astype(int), color=(255,255,255), thickness=int(4), tipLength=0.3, line_type=cv2.LINE_AA)
    arrowed_microscopy_image_2 = cv2.arrowedLine(arrowed_microscopy_image_2, pt1=dst_pts_filt[k,:].astype(int), pt2=src_pts_filt[k,:].astype(int), color=(0,0,255), thickness=int(2), tipLength=0.3, line_type=cv2.LINE_AA)

tifffile.imwrite(microscopy_file_out_2,arrowed_microscopy_image_2)



tc = np.sum(mask)>=5
mean_error = np.nan if not tc else np.mean(dists_real)
q95_error = np.nan if not tc else np.quantile(dists_real,0.95)
q75_error = np.nan if not tc else np.quantile(dists_real,0.75)
q50_error = np.nan if not tc else np.quantile(dists_real,0.5)
q25_error = np.nan if not tc else np.quantile(dists_real,0.25)
q05_error = np.nan if not tc else np.quantile(dists_real,0.05)
min_error = np.nan if not tc else np.min(dists_real)
max_error = np.nan if not tc else np.max(dists_real)

logging.info(f"median distance: {q50_error:5.3} (min: {min_error:5.3}, max: {max_error:5.3})")

global_x_shift = np.nan if not tc else M[0,2]
global_y_shift = np.nan if not tc else M[1,2]
global_affinemat = np.zeros((2,2))*np.nan if not tc else M[:2,:2]
n_points_in_global_affine = 0 if not tc else np.sum(mask)
n_points_total = 0 if not tc else np.sum(mask)

reg_measure_dic = {
    f"{comparison_from}_to_{comparison_to}_mean_error": str(mean_error),
    f"{comparison_from}_to_{comparison_to}_quantile05_error": str(q05_error),
    f"{comparison_from}_to_{comparison_to}_quantile25_error": str(q25_error),
    f"{comparison_from}_to_{comparison_to}_quantile50_error": str(q50_error),
    f"{comparison_from}_to_{comparison_to}_quantile75_error": str(q75_error),
    f"{comparison_from}_to_{comparison_to}_quantile95_error": str(q95_error),
    f"{comparison_from}_to_{comparison_to}_global_x_shift": str(global_x_shift),
    f"{comparison_from}_to_{comparison_to}_global_y_shift": str(global_y_shift),
    f"{comparison_from}_to_{comparison_to}_global_affine_matrix_00": str(global_affinemat[0,0]),
    f"{comparison_from}_to_{comparison_to}_global_affine_matrix_01": str(global_affinemat[0,1]),
    f"{comparison_from}_to_{comparison_to}_global_affine_matrix_10": str(global_affinemat[1,0]),
    f"{comparison_from}_to_{comparison_to}_global_affine_matrix_11": str(global_affinemat[1,1]),
    f"{comparison_from}_to_{comparison_to}_n_points_in_global_affine": str(n_points_in_global_affine),
    f"{comparison_from}_to_{comparison_to}_n_points_total": str(n_points_total)
    }

logging.info("Save json")
json.dump(reg_measure_dic, open(snakemake.output["error_stats"],"w"))

logging.info("Finished")
