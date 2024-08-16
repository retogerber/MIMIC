import sys,os
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], "..","..","workflow","scripts","utils")))
from wsireg.writers.ome_tiff_writer import OmeTiffWriter
from wsireg.reg_transforms.reg_transform_seq import RegTransform, RegTransformSeq
from wsireg.parameter_maps.transformations import BASE_AFF_TFORM
from wsireg.reg_images.loader import reg_image_loader
from image_utils import get_image_shape 
from utils import setNThreads, snakeMakeMock
import logging, traceback
import logging_utils

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    snakemake = snakeMakeMock()
    snakemake.params["input_spacing"] = 0.22537
    snakemake.params["output_spacing"] = 0.22537
    snakemake.input["postIMS"] = ""
    snakemake.output["postIMS_downscaled"] = ""
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        raise Exception("Running in interactive mode!!")
# logging setup
logging_utils.logging_setup(snakemake.log['stdout'])
logging_utils.log_snakemake_info(snakemake)
setNThreads(snakemake.threads)

# params 
input_spacing = snakemake.params["input_spacing"]
output_spacing = snakemake.params["output_spacing"]
scale_factor = input_spacing / output_spacing

# inputs
img_file = snakemake.input["postIMS"]
# outputs
img_out = snakemake.output["postIMS_downscaled"]
img_basename = os.path.basename(img_out).split(".")[0]
img_dirname = os.path.dirname(img_out)

logging.info("Create Transformation")
# get image dimensions
imgshape = get_image_shape(img_file)

# setup transformation sequence
empty_transform = BASE_AFF_TFORM.copy()
empty_transform['Spacing'] = (str(input_spacing),str(input_spacing))
empty_transform['Size'] = (imgshape[1], imgshape[0])
empty_transform['TransformParameters'] = (1/scale_factor,0,0,1/scale_factor,0,0)
rt = RegTransform(empty_transform)
rt.output_size = (int(imgshape[1]*scale_factor), int(imgshape[0]*scale_factor))
rt.output_spacing = (output_spacing, output_spacing)
rts = RegTransformSeq(rt,[0])

logging.info("Transform and Save")
# transform and save image
ri = reg_image_loader(img_file, output_spacing)
writer = OmeTiffWriter(ri, reg_transform_seq=rts)
writer.write_image_by_plane(img_basename, output_dir=img_dirname, tile_size=1024)


logging.info("Finished")