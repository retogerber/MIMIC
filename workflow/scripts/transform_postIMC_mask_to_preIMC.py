from wsireg.reg_shapes import RegShapes
from wsireg.reg_transforms.reg_transform_seq import RegTransformSeq
from utils import setNThreads, snakeMakeMock
import sys,os
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["microscopy_pixelsize"] = 1
    snakemake.input["IMC_location_on_postIMC"] = ""
    snakemake.input["postIMC_to_postIMS_transform"] = ""
    snakemake.output["IMC_location_on_preIMC"] = ""
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params
microscopy_pixelsize = snakemake.params["microscopy_pixelsize"]
# inputs
mask_file = snakemake.input["IMC_location_on_postIMC"]
transform_file = snakemake.input["postIMC_to_postIMS_transform"]
# outputs
out_mask = snakemake.output["IMC_location_on_preIMC"]

logging.info("Read Mask")
# read in mask
rs = RegShapes(mask_file)

logging.info("Read Transformation")
# read in transform
rts = RegTransformSeq(transform_file)
# only use first transform which is postIMC to preIMC
rt = rts.reg_transforms[0]
rts = RegTransformSeq([rt], transform_seq_idx=[0])
#rts.set_output_spacing((1.0, 1.0)) # only do this for .ome.tiff converted data, comment for .ndpi file
rts.set_output_spacing((float(microscopy_pixelsize), float(microscopy_pixelsize))) # only do this for .ome.tiff converted data, comment for .ndpi file

logging.info("Transform and save mask")
# do transformation
rs.transform_shapes(rts)

rs.save_shape_data(out_mask)

logging.info("Finished")