import numpy as np
import json
import skimage
import skimage.metrics
import re
import cv2
from image_utils import readimage_crop, convert_and_scale_image, get_image_shape
from utils import setNThreads, snakeMakeMock
import sys,os
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["input_spacing_1"] = 1 
    snakemake.params["input_spacing_2"] = 0.22537
    snakemake.params["input_spacing_IMC_location"] = 0.22537
    snakemake.params["output_spacing"] = 1
    snakemake.input["microscopy_image_1"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/preIMS/Cirrhosis-TMA-5_New_Detector_001_transformed_on_postIMS.ome.tiff"
    snakemake.input["microscopy_image_2"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/postIMS/test_combined_postIMS.ome.tiff"
    snakemake.input["IMC_location"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMC_location/test_combined_IMC_mask_on_postIMS_A1.geojson"
    snakemake.output["error_stats"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMC_location/test_combined_IMC_mask_on_postIMS_A1.geojson"

    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
input_spacing_1 = snakemake.params["input_spacing_1"]
input_spacing_2 = snakemake.params["input_spacing_2"]
input_spacing_IMC_location = snakemake.params["input_spacing_IMC_location"]
output_spacing = snakemake.params["output_spacing"]
# inputs
microscopy_file_1 = snakemake.input['microscopy_image_1']
microscopy_file_2 = snakemake.input['microscopy_image_2']
IMC_location=snakemake.input["IMC_location"]
if isinstance(IMC_location, list):
    IMC_location = IMC_location[0]
# outputs
error_stats = snakemake.output["error_stats"]

m = re.search("[a-zA-Z]*(?=.ome.tiff$)",os.path.basename(microscopy_file_1))
comparison_to = m.group(0)
comparison_from = os.path.basename(os.path.dirname(microscopy_file_1))
assert(comparison_to in ["preIMC","preIMS","postIMS"])
assert(comparison_from in ["postIMC","preIMC","preIMS"])


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

m2full_shape = get_image_shape(microscopy_file_1)
bb3 = [int(xmin/s1f-1251/input_spacing_1),int(ymin/s1f-1251/input_spacing_1),int(xmax/s1f+1251/input_spacing_1),int(ymax/s1f+1251/input_spacing_1)]
bb3[0] = bb3[0] if bb3[0]>=0 else 0
bb3[1] = bb3[1] if bb3[1]>=0 else 0
bb3[2] = bb3[2] if bb3[2]<=m2full_shape[0] else m2full_shape[0]
bb3[3] = bb3[3] if bb3[3]<=m2full_shape[1] else m2full_shape[1]
logging.info(f"bounding box mask whole image 1: {bb3}")


logging.info("load microscopy image 1")
microscopy_image_1 = readimage_crop(microscopy_file_1, bb1)
microscopy_image_1 = convert_and_scale_image(microscopy_image_1, input_spacing_1/output_spacing)
logging.info(f"  shape: {microscopy_image_1.shape}")

logging.info("load microscopy image 2")
microscopy_image_2 = readimage_crop(microscopy_file_2, bb2)
microscopy_image_2 = convert_and_scale_image(microscopy_image_2, input_spacing_2/output_spacing)
logging.info(f"  shape: {microscopy_image_2.shape}")

logging.info("resize image 2 to image 1")
microscopy_image_2 = cv2.resize(microscopy_image_2, microscopy_image_1.shape[::-1])

assert(microscopy_image_1.shape == microscopy_image_2.shape)

logging.info("Structural Similarity Index")
ssim = skimage.metrics.structural_similarity(microscopy_image_1, microscopy_image_2, data_range=1)
logging.info(f"  SSIM: {ssim}")
logging.info("Mean Squared Error")
mse = skimage.metrics.mean_squared_error(microscopy_image_1, microscopy_image_2)
logging.info(f"  MSE: {mse}")
logging.info("Normalized Mutual Information")
nmi = skimage.metrics.normalized_mutual_information(microscopy_image_1, microscopy_image_2)
logging.info(f"  NMI: {nmi}")
logging.info("Normalized Root Mean Squared Error")
nrmse = skimage.metrics.normalized_root_mse(microscopy_image_1, microscopy_image_2)
logging.info(f"  NRMSE: {nrmse}")




logging.info("Save json")
reg_measure_dic = {
    f"{comparison_from}_to_{comparison_to}_structural_similarity": str(ssim),
    f"{comparison_from}_to_{comparison_to}_mean_squared_error": str(mse),
    f"{comparison_from}_to_{comparison_to}_normalized_mutual_information": str(nmi),
    f"{comparison_from}_to_{comparison_to}_normalized_root_mean_squared_error": str(nrmse)
    }
json.dump(reg_measure_dic, open(error_stats,"w"))

logging.info("Finished")