# from wsireg.writers.ome_tiff_writer import OmeTiffWriter
# from wsireg.reg_transforms.reg_transform_seq import RegTransform, RegTransformSeq
# from wsireg.parameter_maps.transformations import BASE_RIG_TFORM
# from wsireg.reg_images.loader import reg_image_loader
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

lbs = skimage.measure.label(postIMSr)
rps = skimage.measure.regionprops(lbs)
cvi = lbs*0
for i in range(len(rps)):
    print(i)
    tbb = rps[i].bbox
    ti = skimage.morphology.convex_hull_image(lbs[tbb[0]:tbb[2],tbb[1]:tbb[3]]==rps[i].label)
    cvi[tbb[0]:tbb[2],tbb[1]:tbb[3]] = np.logical_or(ti,cvi[tbb[0]:tbb[2],tbb[1]:tbb[3]])

logging.info("Save mask")

saveimage_tile(cvi, postIMSr_file, resolution)
# empty_transform = BASE_RIG_TFORM
# empty_transform['Spacing'] = (str(resolution),str(resolution))
# empty_transform['Size'] = (cvi.shape[1], cvi.shape[0])
# rt = RegTransform(empty_transform)
# rts = RegTransformSeq(rt,[0])
# ri = reg_image_loader(cvi.astype(np.uint8), resolution)
# writer = OmeTiffWriter(ri, reg_transform_seq=rts)
# img_basename = os.path.basename(postIMSr_file).split(".")[0]
# img_dirname = os.path.dirname(postIMSr_file)
# writer.write_image_by_plane(img_basename, output_dir=img_dirname, tile_size=1024)


logging.info("Finished")