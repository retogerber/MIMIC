from wsireg.writers.ome_tiff_writer import OmeTiffWriter
from wsireg.reg_transforms.reg_transform_seq import RegTransform, RegTransformSeq
from wsireg.parameter_maps.transformations import BASE_RIG_TFORM
from wsireg.reg_images.loader import reg_image_loader
from tifffile import imread
import sys,os
import logging, traceback
logging.basicConfig(filename=snakemake.log["stdout"],
                    level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    )
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

# microscopy_pixelsize = 0.22537
input_spacing = snakemake.params["input_spacing"]
output_spacing = snakemake.params["output_spacing"]

# img_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/test_split_pre_postIMC.ome.tiff"
img_file = snakemake.input["postIMS"]

imcmask_file = snakemake.input["IMC_transformed"][0]
# img_out = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMC/A1_postIMC_transformed.ome.tiff"
img_out = snakemake.output["postIMS_downscaled"]
img_basename = os.path.basename(img_out).split(".")[0]
img_dirname = os.path.dirname(img_out)


logging.info("Read image")
# read image
img=imread(img_file)

logging.info("Read Mask")
# mask for final image size
mask = imread(imcmask_file)

logging.info("Create Transformation")
# setup transformation sequence
empty_transform = BASE_RIG_TFORM
empty_transform['Spacing'] = (str(input_spacing),str(input_spacing))
empty_transform['Size'] = (mask.shape[1], mask.shape[0])
# empty_transform['Size'] = (img.shape[1], img.shape[0])
rt = RegTransform(empty_transform)
rts = RegTransformSeq(rt,[0])

logging.info("Transform and Save")
# transform and save image
ri = reg_image_loader(img, output_spacing)
writer = OmeTiffWriter(ri, reg_transform_seq=rts)
writer.write_image_by_plane(img_basename, output_dir=img_dirname, tile_size=1024)


logging.info("Finished")