from rembg import remove, new_session
import skimage
import numpy as np
# from imc_to_ims_workflow.workflow.scripts.image_registration_IMS_to_preIMS_utils import *
from image_registration_IMS_to_preIMS_utils import *
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

# prepare model for rembg
model_name = "isnet-general-use"
os.environ["OMP_NUM_THREADS"] = str(snakemake.threads)
rembg_session = new_session(model_name)

# parameters
stepsize = 30
stepsize = float(snakemake.params["IMS_pixelsize"])
pixelsize = 24
pixelsize = stepsize*float(snakemake.params["IMS_shrink_factor"])
resolution = 1
resolution = float(snakemake.params["IMC_pixelsize"])

# postIMS_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMS/test_split_pre_postIMS_reduced.ome.tiff"
postIMS_file = snakemake.input["postIMS_downscaled"]
# postIMSr_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMS/test_split_pre_postIMS_reduced_mask.ome.tiff"
postIMSr_file = snakemake.output["postIMSmask_downscaled"]

logging.info("Read and process image")
# read postIMS image
postIMS = skimage.io.imread(postIMS_file)
postIMS = prepare_image_for_sam(postIMS, resolution)
# postIMSmpre = skimage.filters.median(postIMS, skimage.morphology.disk( np.floor(((stepsize-pixelsize)/resolution)/3)))
postIMS2r = skimage.filters.median(postIMS, skimage.morphology.disk(int((1/resolution) * (pixelsize/4))))
postIMS2r = np.stack([postIMS2r, postIMS2r, postIMS2r], axis=2)

logging.info("Remove background")
# remove background, i.e. detect cores
postIMSr = remove(postIMS2r, only_mask=True, session=rembg_session,post_process_mask=True)
postIMSr = postIMSr>127
# fill holes in tissue mask
logging.info("Fill holes in mask")
postIMSr = skimage.morphology.remove_small_holes(postIMSr,1000**2*np.pi*(1/resolution))

logging.info("Save mask")
# save mask
skimage.io.imsave(postIMSr_file, postIMSr)


logging.info("Finished")