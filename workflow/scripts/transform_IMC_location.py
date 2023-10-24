import cv2
import SimpleITK as sitk
from wsireg.reg_shapes import RegShapes
from wsireg.reg_transforms.reg_transform_seq import RegTransformSeq
import json
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

cv2.setNumThreads(snakemake.threads)
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(snakemake.threads)

# output_spacing=0.22537
output_spacing = snakemake.params["output_spacing"]
input_spacing = snakemake.params["input_spacing"]

logging.info(f"output_spacing: {output_spacing}")
logging.info(f"input_spacing: {input_spacing}")

# transform_target = "preIMS"
transform_target = snakemake.params["transform_target"]
assert(transform_target in ["preIMC", "preIMS", "postIMS"])
# transform_file_postIMC_to_postIMS = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/registrations/postIMC_to_postIMS/test_split_pre-postIMC_to_postIMS_transformations.json"
# transform_file_postIMC_to_postIMS = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/registrations/postIMC_to_postIMS/A1/test_split_pre_A1-postIMC_to_postIMS_transformations.json"
transform_file_postIMC_to_postIMS=snakemake.input["postIMC_to_postIMS_transform"]

# IMC_geojson_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_location/test_split_pre_IMC_mask_on_postIMC_A1.geojson"
IMC_geojson_file=snakemake.input['IMC_location_on_postIMC']

out_mask = snakemake.output["IMC_location_transformed"]



logging.info("Load transform")
# setup transformation sequence
rtsn=RegTransformSeq(transform_file_postIMC_to_postIMS)
#rtsn.set_output_spacing((microscopy_pixelsize,microscopy_pixelsize))
#rtsn.set_output_spacing((1.0,1.0))
rtsn.set_output_spacing((float(output_spacing),float(output_spacing)))

# if transform_target != "postIMS":
rtls = rtsn.reg_transforms
is_split_transform = len(rtls)==6


logging.info("Setup transformation for image")
if transform_target == "preIMC":
    n_end = 1
elif transform_target == "preIMS":
    n_end = 5 if is_split_transform else 3
elif transform_target == "postIMS":
    n_end = 6 if is_split_transform else 4
else:
    raise ValueError("Unknown transform target: " + transform_target)

rtls = rtsn.reg_transforms
rtls = rtls[:n_end]

rtsngeo = RegTransformSeq(rtls, transform_seq_idx=list(range(len(rtls))))
if len(rtls)>0:
    rtsngeo.set_output_spacing((float(output_spacing),float(output_spacing)))


logging.info("Read json, transform and create shape")
assert(len(rtls)>0)
rs = RegShapes(IMC_geojson_file, source_res=input_spacing, target_res=output_spacing)
rs.transform_shapes(rtsngeo)

rs.save_shape_data(out_mask)
