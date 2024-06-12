import SimpleITK as sitk
import numpy as np
from wsireg.writers.ome_tiff_writer import OmeTiffWriter
from wsireg.reg_transforms.reg_transform import RegTransform
from wsireg.reg_transforms.reg_transform_seq import RegTransformSeq
from wsireg.reg_images.loader import reg_image_loader
import json
from utils import setNThreads, snakeMakeMock
import sys,os
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["input_spacing"] = 0.22537
    snakemake.params["output_spacing"] = 0.22537
    snakemake.params["transform_target"] = "preIMS"
    snakemake.input["IMC_to_preIMC_transform"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/registrations/IMC_to_preIMC/test_split_pre_B1/test_split_pre_B1-precise_IMC_to_preIMC_transformations.json"
    snakemake.input["preIMC_orig_size_transform"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/registrations/IMC_to_preIMC/test_split_pre_B1/test_split_pre_B1_precise_preIMC_orig_size_tform.json"
    snakemake.input["preIMC_to_postIMS_transform"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/registrations/preIMC_to_preIMS/B1/test_split_pre_B1-preIMC_to_postIMS_transformations.json"
    snakemake.input["IMC"] = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_combined/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_002.tiff"
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)


# params
input_spacing = snakemake.params["input_spacing"]
output_spacing = snakemake.params["output_spacing"]
transform_target = snakemake.params["transform_target"]

# inputs
transform_file_IMC_to_preIMC=snakemake.input["IMC_to_preIMC_transform"]
orig_size_tform_IMC_to_preIMC=snakemake.input["preIMC_orig_size_transform"]
transform_file_preIMC_to_postIMS=snakemake.input["preIMC_to_postIMS_transform"]
img_file = snakemake.input["IMC"]

# outputs
img_out = snakemake.output["IMC_transformed"]
img_basename = os.path.basename(img_out).split('.')[0]
img_dirname = os.path.dirname(img_out)

logging.info("Read Transformation 1")
if os.path.getsize(transform_file_IMC_to_preIMC)>0:
# transform sequence IMC to preIMC
    try:
        rts = RegTransformSeq(transform_file_IMC_to_preIMC)
        read_rts_error=False
    except:
        read_rts_error=True
    try:
        tmptform = json.load(open(transform_file_IMC_to_preIMC, "r"))
        print("tmptform")
        print(tmptform)
        tmprt = RegTransform(tmptform)
        rts=RegTransformSeq([tmprt], transform_seq_idx=[0])
        read_rts_error=False
    except:
        read_rts_error=True
    if read_rts_error:
        exit("Could not read transform data transform_file_IMC_to_preIMC: " + transform_file_IMC_to_preIMC)

else:
    print("Empty File!")
    rts = RegTransformSeq()
    
logging.info("Read Transformation 2")
# read transform sequence preIMC
osize_tform = json.load(open(orig_size_tform_IMC_to_preIMC, "r"))
osize_tform_rt = RegTransform(osize_tform)
osize_rts = RegTransformSeq([osize_tform_rt], transform_seq_idx=[0])
rts.append(osize_rts)

rts.set_output_spacing((float(output_spacing),float(output_spacing)))
  
logging.info("Read Transformation 3")
# read in transformation sequence from preIMC to postIMS
rtsn=RegTransformSeq(transform_file_preIMC_to_postIMS)
rtsn.set_output_spacing((float(output_spacing),float(output_spacing)))

logging.info("Setup transformation")
# setup transformation sequence
rtsn=RegTransformSeq(transform_file_preIMC_to_postIMS)
rtsn.set_output_spacing((float(output_spacing),float(output_spacing)))

if transform_target != "postIMS":

    rtls = rtsn.reg_transforms
    all_linear = np.array([r.is_linear for r in rtls]).all()
    if all_linear:
        assert(len(rtls)==4 or len(rtls)==2)
        is_split_transform = len(rtls)==4
    else:
        # len=4 : direct registration
        # len=6 : additional separate registration between preIMC and preIMS
        assert(len(rtls)==5 or len(rtls)==3)
        is_split_transform = len(rtls)==5

    if transform_target == "preIMC":
        n_end = 0
    elif transform_target == "preIMS":
        if all_linear:
            n_end = 3 if is_split_transform else 1
        else:
            n_end = 4 if is_split_transform else 2
    else:
        raise ValueError("Unknown transform target: " + transform_target)

    rtls = rtsn.reg_transforms
    rtls = rtls[:n_end]
    rtsn = RegTransformSeq(rtls, transform_seq_idx=list(range(len(rtls))))
    if len(rtls)>0:
        rtsn.set_output_spacing((float(output_spacing),float(output_spacing)))

# combine transformations
rts.append(rtsn)
rts.set_output_spacing((float(output_spacing),float(output_spacing)))

logging.info("Transform and save image")
ri = reg_image_loader(img_file, float(input_spacing))#,preprocessing=ipp)
writer = OmeTiffWriter(ri, reg_transform_seq=rts)
writer.write_image_by_plane(img_basename, output_dir=img_dirname, tile_size=1024)

logging.info("Finished")
