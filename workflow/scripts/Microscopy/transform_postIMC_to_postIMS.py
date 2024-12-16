import sys,os
# sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "workflow","scripts","utils")))
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
import cv2
from wsireg.writers.ome_tiff_writer import OmeTiffWriter
from wsireg.reg_transforms.reg_transform_seq import RegTransformSeq
from wsireg.reg_images.loader import reg_image_loader
import SimpleITK as sitk
import numpy as np
from shapely.geometry import shape
import json
from image_utils import get_image_shape, readimage_crop, saveimage_tile
from utils import setNThreads, snakeMakeMock
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["input_spacing"] = 0.22537
    snakemake.params["TMA_location_spacing"] = 0.22537
    snakemake.params["output_spacing"] = 0.22537
    snakemake.params["transform_target"] = "preIMC"
    snakemake.params["transform_source"] = "postIMC"
    # snakemake.input["postIMC_to_postIMS_transform"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/registrations/postIMC_to_postIMS/B5/NASH_HCC_TMA_B5-postIMC_to_postIMS_transformations_mod.json"
    snakemake.input["postIMC_to_postIMS_transform"] = [f"/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/registrations/postIMC_to_postIMS/{core}/test_split_pre_{core}-postIMC_to_postIMS_transformations_mod.json" for core in ["A1","B1"]]
    # snakemake.input["postIMC"] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/postIMC/NASH_HCC_TMA_postIMC.ome.tiff"
    snakemake.input["postIMC"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/test_split_pre_postIMC.ome.tiff"
    # snakemake.input['IMC_location'] = "/home/retger/IMC/data/complete_analysis_imc_workflow/imc_to_ims_workflow/results/NASH_HCC_TMA/data/IMC_location/NASH_HCC_TMA_IMC_mask_on_postIMC_B5.geojson"
    snakemake.input["TMA_location_source"] = [f"/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/TMA_location/test_split_pre_TMA_location_on_postIMC_{core}.geojson" for core in ["A1","B1"]]
    snakemake.input["TMA_location_target"] = [f"/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/TMA_location/test_split_pre_TMA_location_on_preIMC_{core}.geojson" for core in ["A1","B1"]]
    snakemake.output["postIMC_transformed"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/test_split_pre_postIMC_on_preIMC.ome.tiff"
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
input_spacing = snakemake.params["input_spacing"]
output_spacing = snakemake.params["output_spacing"]
TMA_location_spacing = snakemake.params["TMA_location_spacing"]
transform_target = snakemake.params["transform_target"]
transform_source = snakemake.params["transform_source"]

# inputs
transform_file_postIMC_to_postIMS=snakemake.input["postIMC_to_postIMS_transform"]
if isinstance(transform_file_postIMC_to_postIMS, str):
    transform_file_postIMC_to_postIMS = [transform_file_postIMC_to_postIMS]
img_file = snakemake.input["postIMC"]
TMA_source_geojson_file=snakemake.input['TMA_location_source']
if isinstance(TMA_source_geojson_file, str):
    TMA_source_geojson_file = [TMA_source_geojson_file]
TMA_target_geojson_file=snakemake.input['TMA_location_target']
if isinstance(TMA_target_geojson_file, str):
    TMA_target_geojson_file = [TMA_target_geojson_file]


# outputs
img_filename_out = snakemake.output["postIMC_transformed"]
img_basename = os.path.basename(img_filename_out).split(".")[0]
img_dirname = os.path.dirname(img_filename_out)

# get info of IMC location
TMA_source_geojson_polygon_ls = list()
for single_TMA_geojson_file in TMA_source_geojson_file:
    TMA_geojson = json.load(open(single_TMA_geojson_file, "r"))
    if isinstance(TMA_geojson,list):
        TMA_geojson=TMA_geojson[0]
    TMA_source_geojson_polygon_ls.append(shape(TMA_geojson['geometry']))

TMA_target_geojson_polygon_ls = list()
for single_TMA_geojson_file in TMA_target_geojson_file:
    TMA_geojson = json.load(open(single_TMA_geojson_file, "r"))
    if isinstance(TMA_geojson,list):
        TMA_geojson=TMA_geojson[0]
    TMA_target_geojson_polygon_ls.append(shape(TMA_geojson['geometry']))

logging.info("Create bounding box")
bb_source_ls = list()
for TMA_geojson_polygon in TMA_source_geojson_polygon_ls:
    # bounding box
    bb1 = TMA_geojson_polygon.bounds
    # reorder axis
    bb1 = np.array([bb1[1],bb1[0],bb1[3],bb1[2]])/(TMA_location_spacing/input_spacing)
    bb1 = bb1.astype(int)
    bb_source_ls.append(bb1)

logging.info("Create bounding box")
bb_target_ls = list()
for TMA_geojson_polygon in TMA_target_geojson_polygon_ls:
    # bounding box
    bb1 = TMA_geojson_polygon.bounds
    # reorder axis
    bb1 = np.array([bb1[1],bb1[0],bb1[3],bb1[2]])/(TMA_location_spacing/input_spacing)
    bb1 = bb1.astype(int)
    bb_target_ls.append(bb1)


logging.info("Load transform")
rtlsls = list()
for single_transform_file_postIMC_to_postIMS in transform_file_postIMC_to_postIMS:
    logging.info(f"Setup transformation for {os.path.basename(single_transform_file_postIMC_to_postIMS)}")
    rtsn=RegTransformSeq(single_transform_file_postIMC_to_postIMS)
    rtsn.set_output_spacing((float(output_spacing),float(output_spacing)))

    rtls = rtsn.reg_transforms
    all_linear = np.array([r.is_linear for r in rtls]).all()
    if all_linear:
        assert(len(rtls)==5 or len(rtls)==3)
        is_split_transform = len(rtls)==5
    else:
        # len=4 : direct registration
        # len=6 : additional separate registration between preIMC and preIMS
        assert(len(rtls)==6 or len(rtls)==4)
        is_split_transform = len(rtls)==6


    if transform_target == "preIMC":
        n_end = 1
    elif transform_target == "preIMS":
        if all_linear:
            n_end = 4 if is_split_transform else 2
        else:
            n_end = 5 if is_split_transform else 3
    elif transform_target == "postIMS":
        if all_linear:
            n_end = 5 if is_split_transform else 3
        else:
            n_end = 6 if is_split_transform else 4
    else:
        raise ValueError("Unknown transform target: " + transform_target)

    if transform_source == "postIMC":
        n_start = 0
    elif transform_source == "preIMC":
        n_start = 1
    elif transform_source == "preIMS":
        if all_linear:
            n_start = 4 if is_split_transform else 2
        else:
            n_start = 5 if is_split_transform else 3
    else:
        raise ValueError("Unknown transform source: " + transform_source)

    rtls = rtsn.reg_transforms
    rtls = rtls[n_start:n_end]
    rtsn = RegTransformSeq(rtls, transform_seq_idx=list(range(len(rtls))))
    if len(rtls)>0:
        rtsn.set_output_spacing((float(output_spacing),float(output_spacing)))
    rtlsls.append(rtsn)


def get_resampler(transform, fixed, default_value=0.0):
    resampler = sitk.ResampleImageFilter()
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(transform)
    resampler.SetReferenceImage(fixed)
    resampler.SetOutputSpacing(fixed.GetSpacing())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return resampler

def resample_image(resampler, moving):
    return sitk.GetArrayFromImage(resampler.Execute(moving))

n_channels = get_image_shape(img_file)[2]
img_dtype = readimage_crop(img_file, [0,0,1,1]).dtype
output_shape = [rtsn.output_size[1], rtsn.output_size[0], n_channels]
out_image = np.zeros(output_shape, dtype=img_dtype)    
logging.info(f"Output shape: {output_shape}")
for i in range(len(rtlsls)):
    logging.info(f"Transform image {i}")
    logging.info(f"\tbb_source: {bb_source_ls[i]}")
    logging.info(f"\tbb_target: {bb_target_ls[i]}")
    logging.info(f"\tcreate composite transform")
    rtsn = rtlsls[i]
    composite = sitk.CompositeTransform(2)
    # add translation to source
    composite.AddTransform(sitk.TranslationTransform(2, [-float(bb_source_ls[i][1]*input_spacing), -float(bb_source_ls[i][0]*input_spacing)]))
    # add all transforms between source and target
    trls = [rtsn.composite_transform.GetNthTransform(j) for j in range(rtsn.composite_transform.GetNumberOfTransforms())]
    for j,trl in enumerate(trls):
        composite.AddTransform(trl)
    # add translation to target
    composite.AddTransform(sitk.TranslationTransform(2, [float(bb_target_ls[i][1]*input_spacing), float(bb_target_ls[i][0]*input_spacing)]))

    logging.info(f"\tnumber of transforms: {composite.GetNumberOfTransforms()}")
    logging.info(f"\ttransforms:")
    trls = [composite.GetNthTransform(j) for j in range(composite.GetNumberOfTransforms())]
    for j,trl in enumerate(trls):
        if trl.IsLinear():
            logging.info(f"\t\t{j}: linear, {trl.GetParameters()}")
        else:
            logging.info(f"\t\t{j}: non-linear")

    logging.info(f"\tcreate resampler")
    resampler = rtsn.resampler
    logging.info(f"\torigin: {resampler.GetOutputOrigin()}")
    logging.info(f"\tspacing: {resampler.GetOutputSpacing()}")
    logging.info(f"\tprevious size: {resampler.GetSize()}")
    newsize = np.array([bb_target_ls[i][3]-bb_target_ls[i][1], bb_target_ls[i][2]-bb_target_ls[i][0]], dtype='int').tolist()
    resampler.SetSize(newsize)
    logging.info(f"\tcropped size: {resampler.GetSize()}")
    resampler.SetTransform(composite)

    logging.info(f"\t\tread image")
    moving_np = readimage_crop(img_file, bb_source_ls[i]).astype(np.float32)
    logging.info(f"\tloop over channels")
    for ch in range(n_channels):
        logging.info(f"\t\tchannel {ch}")
        moving = sitk.GetImageFromArray(moving_np[:,:,ch])
        moving.SetSpacing([input_spacing, input_spacing])

        logging.info(f"\t\tresample image")
        source_image_trans = resample_image(resampler, moving)
        prop_nonzero = np.sum(source_image_trans>0)/np.prod(source_image_trans.shape)
        logging.info(f"\t\tproportion of non-zero pixels: {prop_nonzero}")
        assert prop_nonzero > 0
    
        out_image[bb_target_ls[i][0]:bb_target_ls[i][2], bb_target_ls[i][1]:bb_target_ls[i][3],ch] = source_image_trans

# prop_nonzero = np.sum(out_image>0)/np.prod(out_image.shape)
# logging.info(f"proportion of non-zero pixels: {prop_nonzero}")

logging.info(f"Save image")
saveimage_tile(out_image, filename= img_filename_out, resolution=output_spacing)

# wn = int(out_image.shape[0]*0.5)
# hn = int(out_image.shape[1]*0.5)
# out_image_small = cv2.resize(out_image, (hn,wn), interpolation=cv2.INTER_NEAREST)

# import tifffile
# postIMS = tifffile.imread("/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMS/test_split_pre_postIMS.ome.tiff")
# wn = int(postIMS.shape[0]*0.5)
# hn = int(postIMS.shape[1]*0.5)
# postIMS_small = cv2.resize(postIMS, (hn,wn), interpolation=cv2.INTER_NEAREST)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1,3, figsize=(60,30))
# ax[0].imshow(out_image_small)
# ax[1].imshow(postIMS_small)
# ax[2].imshow(postIMS_small/255-out_image_small/255)
# plt.savefig("test.png")

logging.info("Finished")
