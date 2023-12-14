import imageio.v3 as iio
import numpy as np
import json
import re
import numpy as np
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
    snakemake.params["pixel_expansion"] = 101
    # snakemake.input['microscopy_image_1'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMC/Cirrhosis-TMA-5_New_Detector_001_transformed_on_preIMS.ome.tiff"
    # snakemake.input['microscopy_image_1'] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/postIMC/NASH_HCC_TMA-2_030_transformed_on_preIMC.ome.tiff"
    snakemake.input['microscopy_image_1'] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/preIMC/NASH_HCC_TMA-2_030_transformed_on_preIMS.ome.tiff"
    # snakemake.input['microscopy_image_2'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/preIMS/test_split_pre_preIMS.ome.tiff"
    # snakemake.input['microscopy_image_2'] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/preIMC/NASH_HCC_TMA_preIMC.ome.tiff"
    snakemake.input['microscopy_image_2'] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/preIMS/NASH_HCC_TMA_preIMS.ome.tiff"
    # snakemake.input['IMC_location'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_location/test_split_pre_IMC_mask_on_preIMS_A1.geojson"
    # snakemake.input['IMC_location'] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/IMC_location/NASH_HCC_TMA_IMC_mask_on_preIMC_E6.geojson"
    snakemake.input['IMC_location'] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/IMC_location/NASH_HCC_TMA_IMC_mask_on_preIMS_E6.geojson"
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
input_spacing_1 = snakemake.params["input_spacing_1"]
input_spacing_2 = snakemake.params["input_spacing_2"]
input_spacing_IMC_location = snakemake.params["input_spacing_IMC_location"]
output_spacing = snakemake.params["output_spacing"]
pixel_expansion = snakemake.params["pixel_expansion"]

# inputs
microscopy_file_1 = snakemake.input['microscopy_image_1']
microscopy_file_2 = snakemake.input['microscopy_image_2']
IMC_location=snakemake.input["IMC_location"]
if isinstance(IMC_location, list):
    IMC_location = IMC_location[0]

# outputs
gif_out = snakemake.output['gif_file']

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
bb1 = [int(xmin/s1f-pixel_expansion/input_spacing_1),int(ymin/s1f-pixel_expansion/input_spacing_1),int(xmax/s1f+pixel_expansion/input_spacing_1),int(ymax/s1f+pixel_expansion/input_spacing_1)]
imxmax, imymax, _ = get_image_shape(microscopy_file_1)
imxmax=int(imxmax/input_spacing_1)
imymax=int(imymax/input_spacing_1)
bb1[0] = bb1[0] if bb1[0]>=0 else 0
bb1[1] = bb1[1] if bb1[1]>=0 else 0
bb1[2] = bb1[2] if bb1[2]<=imxmax else imxmax
bb1[3] = bb1[3] if bb1[3]<=imymax else imymax
logging.info(f"bounding box whole image 1: {bb1}")

s2f = input_spacing_2/input_spacing_IMC_location
bb2 = [int(xmin/s2f-pixel_expansion/input_spacing_2),int(ymin/s2f-pixel_expansion/input_spacing_2),int(xmax/s2f+pixel_expansion/input_spacing_2),int(ymax/s2f+pixel_expansion/input_spacing_2)]
imxmax, imymax, _ = get_image_shape(microscopy_file_2)
imxmax=int(imxmax/input_spacing_1)
imymax=int(imymax/input_spacing_1)
bb2[0] = bb2[0] if bb2[0]>=0 else 0
bb2[1] = bb2[1] if bb2[1]>=0 else 0
bb2[2] = bb2[2] if bb2[2]<=imxmax else imxmax
bb2[3] = bb2[3] if bb2[3]<=imymax else imymax
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

logging.info("load microscopy image 2")
microscopy_image_2 = readimage_crop(microscopy_file_2, bb2)
microscopy_image_2 = convert_and_scale_image(microscopy_image_2, input_spacing_2/output_spacing)

logging.info("crop images to same size")
xmax = min([microscopy_image_1.shape[0],microscopy_image_2.shape[0]])
ymax = min([microscopy_image_1.shape[1],microscopy_image_2.shape[1]])
imcbbox_outer = [0,0,xmax,ymax]
microscopy_image_1 = microscopy_image_1[imcbbox_outer[0]:imcbbox_outer[2],imcbbox_outer[1]:imcbbox_outer[3]].copy()
microscopy_image_2 = microscopy_image_2[imcbbox_outer[0]:imcbbox_outer[2],imcbbox_outer[1]:imcbbox_outer[3]].copy()

def label_image(image,label):
    # Add the label "IMS" to the top right corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_position = (image.shape[1] - text_size[0] - 10, text_size[1] + 10)

    # Add black border around label
    border_thickness = 3
    border_size = (text_size[0] + border_thickness * 2, text_size[1] + border_thickness * 2)
    border_position = (text_position[0] - border_thickness, text_position[1] + border_thickness)
    image = cv2.rectangle(image, border_position, (border_position[0] + border_size[0], border_position[1] - border_size[1]), (0, 0, 0), -1)

    image = cv2.putText(image, label, text_position, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return image

logging.info("add label to images")
image1 = label_image(microscopy_image_1, comparison_from)
image2 = label_image(microscopy_image_2, comparison_to)

logging.info("images to 4 bit")
image1 = cv2.convertScaleAbs(cv2.convertScaleAbs(image1, alpha=(15/255)),alpha=(255/15))
image2 = cv2.convertScaleAbs(cv2.convertScaleAbs(image2, alpha=(15/255)),alpha=(255/15))

logging.info("Create frames for gif")
num_frames = 8
blend_factor = np.linspace(0, 1, num_frames)
frames = []
# Blend microscopy_image_1 into microscopy_image_2 and back again
for factor in blend_factor:
    blended_image = factor * image1 + (1 - factor) * image2
    frames.append(blended_image.astype(np.uint8))
# Append the frames in reverse order to create the back transition
frames += frames[1:-1][::-1]
# durations in ms for each frame
durations = [1000]+[200]*(num_frames-2)+[1000]+[200]*(num_frames-2)

logging.info("save gif")
iio.imwrite(gif_out, np.stack(frames), duration=durations, loop=0, palettesize=4, subrectangles=True, plugin="pillow", optimize=True)

logging.info("Finished")