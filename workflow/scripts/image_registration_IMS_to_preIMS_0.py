# from wsireg.writers.ome_tiff_writer import OmeTiffWriter
# from wsireg.reg_transforms.reg_transform_seq import RegTransform, RegTransformSeq
# from wsireg.parameter_maps.transformations import BASE_RIG_TFORM
# from wsireg.reg_images.loader import reg_image_loader
from rembg import remove, new_session
from segment_anything import sam_model_registry, SamPredictor
import torch
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
# CHECKPOINT_PATH = "/home/retger/Downloads/sam_vit_h_4b8939.pth"
CHECKPOINT_PATH = snakemake.input["sam_weights"]
DEVICE = 'cpu'
MODEL_TYPE = "vit_h"
torch.set_num_threads(snakemake.threads)

# parameters
stepsize = 30
stepsize = float(snakemake.params["IMS_pixelsize"])
pixelsize = 24
pixelsize = stepsize*float(snakemake.params["IMS_shrink_factor"])
resolution = 1
resolution = float(snakemake.params["IMC_pixelsize"])

# postIMS_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMS/test_split_pre_postIMS_reduced.ome.tiff"
# postIMS_file = "/home/retger/Downloads/Lipid_TMA_3781_postIMS_reduced.ome.tiff"
postIMS_file = snakemake.input["postIMS_downscaled"]
# postIMSr_file = "/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/postIMS/test_split_pre_postIMS_reduced_mask.ome.tiff"
postIMSr_file = snakemake.output["postIMSmask_downscaled"]

# imc_mask_files = [f"/home/retger/Downloads/Lipid_TMA_37819_0{i}_transformed.ome.tiff" for i in ["09"] + list(range(10,43))]
# imc_mask_files = [ f"/home/retger/Nextcloud/Projects/test_imc_to_ims_workflow/imc_to_ims_workflow/results/test_split_pre/data/IMC_mask/Cirrhosis-TMA-5_New_Detector_00{i}_transformed.ome.tiff" for i in [1,2]]
imc_mask_files = snakemake.input["IMCmask"]
logging.info("Read and process image")
# read postIMS image
postIMS = skimage.io.imread(postIMS_file)
postIMS = prepare_image_for_sam(postIMS, resolution)
# postIMSmpre = skimage.filters.median(postIMS, skimage.morphology.disk( np.floor(((stepsize-pixelsize)/resolution)/3)))
# postIMS2r = skimage.filters.median(postIMS, skimage.morphology.disk(int((1/resolution) * (pixelsize/4))))
# postIMS2r = np.stack([postIMS, postIMS, postIMS], axis=2)



logging.info("Read IMC masks")
imcbboxls = list()
for imcmaskfile in imc_mask_files:
    imc=skimage.io.imread(imcmaskfile)
    imc[imc>0]=255
    imc = imc.astype(np.uint8)
    imcbboxls.append(skimage.measure.regionprops(imc)[0].bbox)

med_radius = int((1/resolution) * (stepsize/4))
logging.info(f"Median filter postIMS with radius: {med_radius}px")
postIMS2r = skimage.filters.median(postIMS, skimage.morphology.disk(med_radius))
# postIMS2r = np.stack([postIMS2r, postIMS2r, postIMS2r], axis=2)


logging.info("Remove background individually for each IMC location")
postIMSstitch = postIMS2r[:,:]*0
postIMSmask = postIMS2r[:,:]*0
for i in range(len(imcbboxls)):
    #for expand in [750,1000,1250,1500,1750]:
    for expand in [int(r/resolution) for r in [1500]]:
        xmin = int(imcbboxls[i][0]-expand)
        xmin = xmin if xmin>=0 else 0
        ymin = int(imcbboxls[i][1]-expand)
        ymin = ymin if ymin>=0 else 0
        xmax = int(imcbboxls[i][2]+expand)
        xmax = xmax if xmax<=postIMS2r.shape[0] else postIMS2r.shape[0]
        ymax = int(imcbboxls[i][3]+expand)
        ymax = ymax if ymax<=postIMS2r.shape[1] else postIMS2r.shape[1]
        print(f"i: {i}, {os.path.basename(imc_mask_files[i])}, coords:[{xmin}:{xmax},{ymin}:{ymax}]")
        postIMSmask[(xmin+expand):(xmax-expand),(ymin+expand):(ymax-expand)] = 255
        tmpimg = remove(postIMS2r[xmin:xmax,ymin:ymax], only_mask=True, session=rembg_session,post_process_mask=True)
        tmpimg2 = np.zeros(np.array(list(tmpimg.shape))+20, dtype=np.uint8)
        tmpimg2[10:(tmpimg2.shape[0]-10),10:(tmpimg2.shape[1]-10)] = tmpimg.astype(np.uint8)
        tmpimg2 = skimage.morphology.convex_hull_image(tmpimg2>127)
        tmpimg = tmpimg2[10:(tmpimg2.shape[0]-10),10:(tmpimg2.shape[1]-10)]
        tmpimg = preprocess_mask(tmpimg,1) 
        tmpimg = tmpimg.astype(np.uint8)*255
        postIMSstitch[xmin:xmax,ymin:ymax] = np.max(np.stack([postIMSstitch[xmin:xmax,ymin:ymax],tmpimg], axis=0),axis=0)

postIMSr = postIMSstitch>127
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].imshow(postIMSstitch, cmap='gray')
# ax[0].set_title("postIMS")
# ax[1].imshow(postIMSmask, cmap='gray')
# ax[1].set_title("postIMSmask")
# plt.show()


logging.info("Check mask")
IMC_filled = list()
for i in range(len(imcbboxls)):
    xmin = int(imcbboxls[i][0])
    ymin = int(imcbboxls[i][1])
    xmax = int(imcbboxls[i][2])
    ymax = int(imcbboxls[i][3])
    IMC_filled.append(np.min(postIMSr[xmin:xmax,ymin:ymax]) == 1)
IMC_filled

if np.sum(IMC_filled)<len(IMC_filled):
    inds = np.array(list(range(len(imcbboxls))))[np.logical_not(np.array(IMC_filled))]
    logging.info(f"Number of incomplete masks found: {len(inds)}")
    logging.info("Run segment anything model on missing masks")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    for i in inds:
        xmin = int(imcbboxls[i][0]-1500/resolution)
        xmin = xmin if xmin>=0 else 0
        ymin = int(imcbboxls[i][1]-1500/resolution)
        ymin = ymin if ymin>=0 else 0
        xmax = int(imcbboxls[i][2]+1500/resolution)
        xmax = xmax if xmax<=postIMS2r.shape[0] else postIMS2r.shape[0]
        ymax = int(imcbboxls[i][3]+1500/resolution)
        ymax = ymax if ymax<=postIMS2r.shape[1] else postIMS2r.shape[1]
        print(f"i: {i}, {os.path.basename(imc_mask_files[i])}, coords:[{xmin}:{xmax},{ymin}:{ymax}]")
        saminp = np.stack([postIMS2r[xmin:xmax,ymin:ymax],postIMS2r[xmin:xmax,ymin:ymax],postIMS2r[xmin:xmax,ymin:ymax]], axis=2)
        postIMSmasks, scores1 = sam_core(saminp, sam)
        postIMSmasks = np.stack([preprocess_mask(msk,1) for msk in postIMSmasks ])
        tmpareas = np.array([np.sum(im) for im in postIMSmasks])
        imcarea = (imcbboxls[i][2]-imcbboxls[i][0])*(imcbboxls[i][2]-imcbboxls[i][1])
        tmpinds = np.array(list(range(3)))
        tmpinds = tmpinds[tmpareas > imcarea*1.02]
        tmpind = tmpinds[tmpareas[tmpinds]==np.min(tmpareas[tmpinds])]
        tmpimg = postIMSmasks[tmpind,:,:][0,:,:].astype(np.uint8)*255
        postIMSstitch[xmin:xmax,ymin:ymax] = np.max(np.stack([postIMSstitch[xmin:xmax,ymin:ymax],tmpimg], axis=0),axis=0)

postIMSr = postIMSstitch>127
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].imshow(postIMSstitch, cmap='gray')
# ax[0].set_title("postIMS")
# ax[1].imshow(postIMSmask, cmap='gray')
# ax[1].set_title("postIMSmask")
# plt.show()

logging.info(f"Get convex hull")
lbs = skimage.measure.label(postIMSr)
rps = skimage.measure.regionprops(lbs)
cvi = lbs*0
for i in range(len(rps)):
    tbb = rps[i].bbox
    ti = skimage.morphology.convex_hull_image(lbs[tbb[0]:tbb[2],tbb[1]:tbb[3]]==rps[i].label)
    cvi[tbb[0]:tbb[2],tbb[1]:tbb[3]] = np.logical_or(ti,cvi[tbb[0]:tbb[2],tbb[1]:tbb[3]])
# plt.imshow(cvi)
# plt.show()
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