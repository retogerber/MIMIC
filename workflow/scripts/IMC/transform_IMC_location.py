import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
from wsireg.reg_shapes import RegShapes
from wsireg.reg_transforms.reg_transform_seq import RegTransformSeq
import numpy as np
from utils import setNThreads, snakeMakeMock
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["input_spacing"] = 0.22537
    snakemake.params["output_spacing"] = 0.22537
    snakemake.params["transform_target"] = "preIMS"
    snakemake.input["postIMC_to_postIMS_transform"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/registrations/postIMC_to_postIMS/test_combined-postIMC_to_postIMS_transformations_mod.json"
    snakemake.input['IMC_location_on_postIMC'] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_location/test_split_pre_IMC_mask_on_postIMC_A1.geojson"
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
output_spacing = snakemake.params["output_spacing"]
input_spacing = snakemake.params["input_spacing"]
transform_target = snakemake.params["transform_target"]
assert(transform_target in ["preIMC", "preIMS", "postIMS"])

# inputs
transform_file_postIMC_to_postIMS=snakemake.input["postIMC_to_postIMS_transform"]
IMC_geojson_file=snakemake.input['IMC_location_on_postIMC']

# outputs
out_mask = snakemake.output["IMC_location_transformed"]

logging.info("Load transform")
# setup transformation sequence
rtsn=RegTransformSeq(transform_file_postIMC_to_postIMS)
#rtsn.set_output_spacing((microscopy_pixelsize,microscopy_pixelsize))
#rtsn.set_output_spacing((1.0,1.0))
rtsn.set_output_spacing((float(output_spacing),float(output_spacing)))

# if transform_target != "postIMS":
rtls = rtsn.reg_transforms
logging.info(f"Number of transforms: {len(rtls)}")
all_linear = np.array([r.is_linear for r in rtls]).all()
if all_linear:
    assert(len(rtls)==5 or len(rtls)==3)
    is_split_transform = len(rtls)==5
else:
    # len=4 : direct registration
    # len=6 : additional separate registration between preIMC and preIMS
    assert(len(rtls)==6 or len(rtls)==4)
    is_split_transform = len(rtls)==6


logging.info("Setup transformation for image")
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

rtls = rtsn.reg_transforms
rtls = rtls[:n_end]

rtsngeo = RegTransformSeq(rtls, transform_seq_idx=list(range(len(rtls))))
assert(len(rtls)>0)
rtsngeo.set_output_spacing((float(output_spacing),float(output_spacing)))


logging.info("Read json, transform and create shape")
rs = RegShapes(IMC_geojson_file, source_res=input_spacing, target_res=output_spacing)
rs.transform_shapes(rtsngeo)

tmpout = rs.transformed_shape_data[0]['array']
xmin = np.min(tmpout[:,0])
assert(xmin>0)
xmax = np.max(tmpout[:,0])
assert(xmax<=rtsngeo.output_size[0])
ymin = np.min(tmpout[:,1])
assert(ymin>0)
ymax = np.max(tmpout[:,1])
assert(ymax<=rtsngeo.output_size[1])

rs.save_shape_data(out_mask)

logging.info(f"Finished")